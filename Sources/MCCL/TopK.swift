import Foundation

// MARK: - Streams

/// Identifies an independent sequence of collectives on one communicator.
///
/// Only stateful compression needs this: `.topK` carries an error-feedback
/// residual from one call to the next, and two unrelated tensors reduced on the
/// same communicator must not share that residual. Everything else ignores it.
///
/// The C shim maps `mcclStream_t` onto this, which is why the payload is a
/// plain integer rather than an object identity.
public struct StreamID: Hashable, Sendable, CustomStringConvertible {
    public let rawValue: UInt32

    public init(_ rawValue: UInt32) { self.rawValue = rawValue }

    /// The stream every collective uses unless the caller says otherwise.
    public static let `default` = StreamID(0)

    public var description: String { "stream(\(rawValue))" }
}

// MARK: - Residual state

/// Per-stream error-feedback residuals, owned by a `Communicator`.
///
/// The residual is the part of a gradient that top-k declined to send. Keeping
/// it and adding it back on the next call is what makes sparsified all-reduce
/// unbiased in the limit: every element eventually accumulates enough magnitude
/// to be selected, so nothing is dropped permanently, it is only delayed.
///
/// Residuals are always `Float`, whatever the caller's dtype. An fp16 residual
/// would flush the small values it exists to preserve straight back to zero.
final class ResidualStore: @unchecked Sendable {
    private let lock = NSLock()
    private var slots: [StreamID: [Float]] = [:]

    /// The residual for `stream`, or a fresh zero vector when there is none or
    /// the caller changed the element count (a different tensor on the same
    /// stream is a different history).
    func load(_ stream: StreamID, count: Int) -> [Float] {
        lock.lock(); defer { lock.unlock() }
        if let existing = slots[stream], existing.count == count { return existing }
        return [Float](repeating: 0, count: count)
    }

    func store(_ stream: StreamID, _ values: [Float]) {
        lock.lock(); defer { lock.unlock() }
        slots[stream] = values
    }

    func snapshot(_ stream: StreamID) -> [Float]? {
        lock.lock(); defer { lock.unlock() }
        return slots[stream]
    }

    /// Drops one stream's residual, or every stream's when `stream` is nil.
    func reset(_ stream: StreamID?) {
        lock.lock(); defer { lock.unlock() }
        if let stream { slots[stream] = nil } else { slots.removeAll() }
    }

    var streams: [StreamID] {
        lock.lock(); defer { lock.unlock() }
        return Array(slots.keys)
    }
}

// MARK: - Selection

enum TopK {
    /// How many elements a fraction selects. At least one — a collective that
    /// sends nothing makes no progress — and never more than the whole buffer.
    static func count(elementCount: Int, fraction: Double) -> Int {
        guard elementCount > 0 else { return 0 }
        let raw = (Double(elementCount) * fraction).rounded(.up)
        guard raw.isFinite else { return elementCount }
        return Swift.max(1, Swift.min(elementCount, Int(raw)))
    }

    /// Indices of the `k` largest magnitudes, returned in ascending index order.
    ///
    /// Quickselect, not a sort: selection is on the hot path of every
    /// compressed all-reduce and `n` is the whole tensor, so O(n) average
    /// beats O(n log n) by a wide margin at 16M elements.
    static func selectIndices(magnitudes: [Float], k: Int) -> [Int] {
        let n = magnitudes.count
        guard k > 0, n > 0 else { return [] }
        guard k < n else { return Array(0..<n) }

        var indices = Array(0..<n)
        indices.withUnsafeMutableBufferPointer { idx in
            magnitudes.withUnsafeBufferPointer { mag in
                var lo = 0
                var hi = n - 1
                let target = k - 1
                // Hoare partition, descending. Loop until the k-th largest sits
                // at position `target`; everything before it is >= it.
                while lo < hi {
                    let pivot = mag[idx[lo + (hi - lo) / 2]]
                    var i = lo
                    var j = hi
                    while i <= j {
                        while mag[idx[i]] > pivot { i += 1 }
                        while mag[idx[j]] < pivot { j -= 1 }
                        if i <= j {
                            idx.swapAt(i, j)
                            i += 1
                            j -= 1
                        }
                    }
                    if target <= j { hi = j } else if target >= i { lo = i } else { break }
                }
            }
        }
        var selected = Array(indices[0..<k])
        selected.sort()
        return selected
    }
}

// MARK: - Sparse block layout

/// The on-the-wire form of one rank's sparsified contribution.
///
/// ```
///  0..3                     nnz (little-endian UInt32)
///  4 ..< 4+4·nnz            element indices, ascending
///  4+4·nnz ..< end          values, in the caller's dtype
/// ```
///
/// Self-describing: a receiver never has to be told `k` out of band. Every rank
/// in a collective produces exactly the same `nnz`, so the blocks are uniform
/// and an ordinary ring all-gather can move them.
enum TopKBlock {
    static let headerBytes = 4

    static func byteCount(nonZeros k: Int, valueWidth: Int) -> Int {
        headerBytes + k * (MemoryLayout<UInt32>.size + valueWidth)
    }

    /// Writes the entries at `indices`, reading their values from `values`.
    @discardableResult
    static func write(
        indices: [Int], values: [Float], dataType: DataType, into dst: UnsafeMutableRawPointer
    ) -> Int {
        let width = dataType.byteWidth
        dst.storeBytes(of: UInt32(indices.count).littleEndian, toByteOffset: 0, as: UInt32.self)
        var offset = headerBytes
        for index in indices {
            dst.storeBytes(of: UInt32(index).littleEndian, toByteOffset: offset, as: UInt32.self)
            offset += MemoryLayout<UInt32>.size
        }
        for index in indices {
            ElementIO.storeFloat(values[index], into: dst, byteOffset: offset, dataType)
            offset += width
        }
        return offset
    }

    /// Adds a block's entries into a dense destination buffer.
    ///
    /// Summation, not assignment: reducing `n` sparse contributions is exactly
    /// `n` of these onto a zeroed buffer, and two ranks may well have selected
    /// the same index.
    static func accumulate(
        payload: UnsafeRawPointer, payloadBytes: Int, elementCount: Int,
        dataType: DataType, into dst: UnsafeMutableRawPointer
    ) throws {
        guard payloadBytes >= headerBytes else {
            throw MCCLError.protocolViolation("top-k block of \(payloadBytes) bytes has no header")
        }
        let nnz = Int(UInt32(littleEndian: payload.loadUnaligned(fromByteOffset: 0, as: UInt32.self)))
        let width = dataType.byteWidth
        guard nnz >= 0, payloadBytes >= byteCount(nonZeros: nnz, valueWidth: width) else {
            throw MCCLError.protocolViolation(
                "top-k block claims \(nnz) entries but carries only \(payloadBytes) bytes")
        }
        let valueBase = headerBytes + nnz * MemoryLayout<UInt32>.size
        for entry in 0..<nnz {
            let index = Int(UInt32(littleEndian: payload.loadUnaligned(
                fromByteOffset: headerBytes + entry * MemoryLayout<UInt32>.size, as: UInt32.self)))
            guard index < elementCount else {
                throw MCCLError.protocolViolation(
                    "top-k index \(index) out of range for \(elementCount) elements")
            }
            let value = ElementIO.loadFloat(payload, byteOffset: valueBase + entry * width, dataType)
            let existing = ElementIO.loadFloat(dst, byteOffset: index * width, dataType)
            ElementIO.storeFloat(existing + value, into: dst, byteOffset: index * width, dataType)
        }
    }

    /// Number of entries a block declares. Used by tests and by the bench to
    /// check that the wire really is as sparse as the fraction promised.
    static func nonZeroCount(payload: UnsafeRawPointer, payloadBytes: Int) throws -> Int {
        guard payloadBytes >= headerBytes else {
            throw MCCLError.protocolViolation("top-k block of \(payloadBytes) bytes has no header")
        }
        return Int(UInt32(littleEndian: payload.loadUnaligned(fromByteOffset: 0, as: UInt32.self)))
    }
}
