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
    ///
    /// The select runs over the *magnitudes*, not over a permutation of
    /// indices. Partitioning an index array means every comparison is a random
    /// read into the magnitudes — a cache miss per comparison, and quickselect
    /// makes ~2n of them — where partitioning the values themselves is a
    /// sequential scan the hardware prefetches. The k-th largest magnitude then
    /// becomes a threshold, and one final pass collects the indices at or above
    /// it. That pass walks in index order, so the result is already ascending
    /// and the old `sort()` of the selection is gone too.
    static func selectIndices(magnitudes: [Float], k: Int) -> [Int] {
        let n = magnitudes.count
        guard k > 0, n > 0 else { return [] }
        guard k < n else { return Array(0..<n) }

        var scratch = magnitudes
        let threshold = scratch.withUnsafeMutableBufferPointer { values in
            kthLargest(values, position: k - 1)
        }
        return magnitudes.withUnsafeBufferPointer { mag in
            let strict = CodecKernels.countAboveFP32(mag.baseAddress!, count: n, threshold: threshold)
            return collect(count: n, k: k, strictCount: strict, threshold: threshold) { mag[$0] }
        }
    }

    /// Indices of the `k` entries whose score is at or above `threshold`, in
    /// ascending order. `score` must be the same function the threshold was
    /// derived from, and `strictCount` the number of entries strictly above it.
    ///
    /// Everything strictly above the threshold is in — at most k-1 elements, by
    /// definition of the k-th largest — and elements equal to it fill the rest.
    /// Knowing the strict count up front is what lets this pass walk in index
    /// order and stop at exactly k, so the result needs no sort. Without it an
    /// early run of ties could fill the block and displace a larger element
    /// further down the buffer.
    ///
    /// The tie test is `!(m < threshold)` rather than `m == threshold` so a NaN
    /// score (which loses every comparison, including to itself) still counts
    /// as a tie. Selection then degrades to "the first k", where `==` would
    /// have returned fewer than k indices and left the block short of what its
    /// own header claims.
    @inline(__always)
    static func collect(
        count n: Int, k: Int, strictCount: Int, threshold: Float, score: (Int) -> Float
    ) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(k)
        var tieQuota = k - strictCount
        for i in 0..<n {
            let m = score(i)
            if m > threshold {
                selected.append(i)
            } else if tieQuota > 0, !(m < threshold) {
                selected.append(i)
                tieQuota -= 1
            }
            if selected.count == k { break }
        }
        return selected
    }

    /// The value that would sit at `position` if `values` were sorted
    /// descending. Hoare partition, average O(n), and `values` is left
    /// partially ordered — the caller owns the scratch copy.
    static func kthLargest(_ values: UnsafeMutableBufferPointer<Float>, position: Int) -> Float {
        var lo = 0
        var hi = values.count - 1
        while lo < hi {
            let pivot = values[lo + (hi - lo) / 2]
            var i = lo
            var j = hi
            while i <= j {
                while values[i] > pivot { i += 1 }
                while values[j] < pivot { j -= 1 }
                if i <= j {
                    values.swapAt(i, j)
                    i += 1
                    j -= 1
                }
            }
            // Landing between the partitions means `position` is inside the
            // run of elements equal to the pivot, which is the answer.
            if position <= j { hi = j } else if position >= i { lo = i } else { break }
        }
        return values[position]
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
        // The dtype is loop-invariant; taking it out of the packing loop is
        // worth a branch per entry, and the fp32 store then needs no dispatch
        // at all.
        if dataType == .float32 {
            values.withUnsafeBufferPointer { source in
                for index in indices {
                    dst.storeBytes(of: source[index], toByteOffset: offset, as: Float.self)
                    offset += 4
                }
            }
            return offset
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
        let isFP32 = dataType == .float32
        for entry in 0..<nnz {
            let index = Int(UInt32(littleEndian: payload.loadUnaligned(
                fromByteOffset: headerBytes + entry * MemoryLayout<UInt32>.size, as: UInt32.self)))
            guard index < elementCount else {
                throw MCCLError.protocolViolation(
                    "top-k index \(index) out of range for \(elementCount) elements")
            }
            if isFP32 {
                let value = payload.loadUnaligned(fromByteOffset: valueBase + entry * 4, as: Float.self)
                let existing = dst.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
                dst.storeBytes(of: existing + value, toByteOffset: index * 4, as: Float.self)
                continue
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

// MARK: - The sending half of a top-k all-reduce

enum TopKKernels {
    /// Steps 1 and 2 of a top-k all-reduce: fold the residual back into the
    /// caller's buffer, select the `k` largest magnitudes of the sum, write
    /// them as a sparse block, and leave everything not sent in `residual`.
    ///
    /// `residual` comes in as the previous call's leftovers and goes out as
    /// this call's — the whole point of error feedback is that the two are the
    /// same vector. Returns the bytes written to `destination`.
    @discardableResult
    static func encodeBlock(
        source: UnsafeRawPointer, count: Int, dataType: DataType, fraction: Double,
        residual: inout [Float], into destination: UnsafeMutableRawPointer
    ) -> Int {
        var magnitudes = [Float](repeating: 0, count: count)
        foldResidual(source: source, count: count, dataType: dataType,
                     work: &residual, magnitudes: &magnitudes)

        let k = TopK.count(elementCount: count, fraction: fraction)
        let selected = TopK.selectIndices(magnitudes: magnitudes, k: k)
        let bytes = TopKBlock.write(
            indices: selected, values: residual, dataType: dataType, into: destination)
        // What went on the wire is no longer owed to anyone; what stayed is.
        for index in selected { residual[index] = 0 }
        return bytes
    }

    /// `work += source`, and `magnitudes = |work|` with non-finite elements
    /// scored zero so they can never be selected.
    static func foldResidual(
        source: UnsafeRawPointer, count: Int, dataType: DataType,
        work: inout [Float], magnitudes: inout [Float]
    ) {
        if dataType == .float32 {
            work.withUnsafeMutableBufferPointer { w in
                magnitudes.withUnsafeMutableBufferPointer { m in
                    CodecKernels.foldResidualFP32(
                        source: source, count: count,
                        work: w.baseAddress!, magnitudes: m.baseAddress!)
                }
            }
            return
        }
        let width = dataType.byteWidth
        for i in 0..<count {
            let value = ElementIO.loadFloat(source, byteOffset: i * width, dataType) + work[i]
            work[i] = value
            magnitudes[i] = value.isFinite ? abs(value) : 0
        }
    }
}
