import Foundation

/// Fixed 24-byte frame header. Every byte mccl puts on a wire is preceded by
/// one of these — collectives, bootstrap handshakes and the probe all share it.
///
/// Layout (little-endian):
/// ```
///  0..3  magic 'MCCL'
///  4     version
///  5     wire codec id
///  6     dtype code
///  7     flags
///  8..11 element count
/// 12..15 payload bytes (post-compression)
/// 16..19 codec block size
/// 20..23 tag (collective step / probe opcode)
/// ```
struct WireHeader {
    static let magic: UInt32 = 0x4C43_434D   // 'M','C','C','L'
    static let byteCount = 24
    static let currentVersion: UInt8 = 1

    var version: UInt8 = WireHeader.currentVersion
    var codec: UInt8 = 0
    var dataType: UInt8 = 0
    var flags: UInt8 = 0
    var elementCount: UInt32 = 0
    var payloadBytes: UInt32 = 0
    var blockSize: UInt32 = 0
    var tag: UInt32 = 0

    func encode(into p: UnsafeMutableRawPointer) {
        p.storeBytes(of: WireHeader.magic.littleEndian, toByteOffset: 0, as: UInt32.self)
        p.storeBytes(of: version, toByteOffset: 4, as: UInt8.self)
        p.storeBytes(of: codec, toByteOffset: 5, as: UInt8.self)
        p.storeBytes(of: dataType, toByteOffset: 6, as: UInt8.self)
        p.storeBytes(of: flags, toByteOffset: 7, as: UInt8.self)
        p.storeBytes(of: elementCount.littleEndian, toByteOffset: 8, as: UInt32.self)
        p.storeBytes(of: payloadBytes.littleEndian, toByteOffset: 12, as: UInt32.self)
        p.storeBytes(of: blockSize.littleEndian, toByteOffset: 16, as: UInt32.self)
        p.storeBytes(of: tag.littleEndian, toByteOffset: 20, as: UInt32.self)
    }

    static func decode(from p: UnsafeRawPointer) throws -> WireHeader {
        let magic = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
        guard magic == WireHeader.magic else {
            throw MCCLError.protocolViolation(String(format: "bad frame magic 0x%08x", magic))
        }
        var h = WireHeader()
        h.version = p.loadUnaligned(fromByteOffset: 4, as: UInt8.self)
        guard h.version == WireHeader.currentVersion else {
            throw MCCLError.protocolViolation("frame version \(h.version) != \(WireHeader.currentVersion)")
        }
        h.codec = p.loadUnaligned(fromByteOffset: 5, as: UInt8.self)
        h.dataType = p.loadUnaligned(fromByteOffset: 6, as: UInt8.self)
        h.flags = p.loadUnaligned(fromByteOffset: 7, as: UInt8.self)
        h.elementCount = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
        h.payloadBytes = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
        h.blockSize = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 16, as: UInt32.self))
        h.tag = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 20, as: UInt32.self))
        return h
    }
}

/// A reusable, 16-byte-aligned scratch buffer. The collectives hold a few of
/// these for the duration of an operation rather than allocating per ring step.
final class ScratchBuffer {
    private(set) var storage: UnsafeMutableRawBufferPointer

    init(capacity: Int) {
        storage = UnsafeMutableRawBufferPointer.allocate(byteCount: max(capacity, 16), alignment: 16)
    }

    func ensure(_ byteCount: Int) {
        guard byteCount > storage.count else { return }
        storage.deallocate()
        storage = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: 16)
    }

    var base: UnsafeMutableRawPointer { storage.baseAddress! }

    deinit { storage.deallocate() }
}

// MARK: - Framed send/receive

extension Channel {
    /// Sends `header` followed by `payloadBytes` from `payload`, as one write.
    func sendFrame(_ header: WireHeader, payload: UnsafeRawPointer?, payloadBytes: Int, using scratch: ScratchBuffer) throws {
        scratch.ensure(WireHeader.byteCount + payloadBytes)
        var h = header
        h.payloadBytes = UInt32(payloadBytes)
        h.encode(into: scratch.base)
        if payloadBytes > 0, let payload {
            (scratch.base + WireHeader.byteCount).copyMemory(from: payload, byteCount: payloadBytes)
        }
        try sendBytes(UnsafeRawBufferPointer(start: scratch.base, count: WireHeader.byteCount + payloadBytes))
    }

    /// Sends a frame whose payload the caller already wrote into `scratch` at
    /// `WireHeader.byteCount`. Saves the extra copy on the collective hot path.
    func sendPreparedFrame(_ header: WireHeader, payloadBytes: Int, in scratch: ScratchBuffer) throws {
        var h = header
        h.payloadBytes = UInt32(payloadBytes)
        h.encode(into: scratch.base)
        try sendBytes(UnsafeRawBufferPointer(start: scratch.base, count: WireHeader.byteCount + payloadBytes))
    }

    /// Reads one frame; the payload lands at offset 0 of `scratch`.
    @discardableResult
    func receiveFrame(into scratch: ScratchBuffer, maxPayloadBytes: Int) throws -> WireHeader {
        var headerBytes = [UInt8](repeating: 0, count: WireHeader.byteCount)
        try headerBytes.withUnsafeMutableBytes { try receiveBytes(into: $0) }
        let header = try headerBytes.withUnsafeBytes { try WireHeader.decode(from: $0.baseAddress!) }
        let payloadBytes = Int(header.payloadBytes)
        guard payloadBytes <= maxPayloadBytes else {
            throw MCCLError.protocolViolation("frame payload \(payloadBytes) exceeds expected max \(maxPayloadBytes)")
        }
        if payloadBytes > 0 {
            scratch.ensure(payloadBytes)
            try receiveBytes(into: UnsafeMutableRawBufferPointer(start: scratch.base, count: payloadBytes))
        }
        return header
    }
}
