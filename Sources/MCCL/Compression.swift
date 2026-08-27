import Foundation

/// On-the-wire encodings. The id travels in the frame header, so a receiver
/// always decodes with the scheme the sender actually used — a rank never has
/// to be told out of band.
enum WireCodecID: UInt8 {
    case raw = 0
    case fp16Downcast = 1
    case int8Blockwise = 2
}

/// Applies a `WireCompression` scheme to one buffer of `dataType`.
///
/// Two invariants hold everywhere in mccl:
/// 1. The caller's dtype never changes. Compression is per-hop only.
/// 2. A scheme that cannot help a given dtype silently degrades to `.raw`
///    rather than erroring — `.downcast` on an fp16 buffer is already done.
struct WireCodec {
    let dataType: DataType
    let codec: WireCodecID
    let blockSize: Int

    init(dataType: DataType, compression: WireCompression) throws {
        self.dataType = dataType
        switch compression {
        case .none:
            codec = .raw
            blockSize = 0
        case .downcast:
            // Only fp32 has anything to give up; 16-bit and integer payloads
            // pass through untouched.
            codec = (dataType == .float32) ? .fp16Downcast : .raw
            blockSize = 0
        case .int8Blockwise(let bs):
            guard bs > 0 else { throw MCCLError.invalidArgument("int8Blockwise blockSize must be > 0") }
            // Quantising integers would be lossy for no bandwidth win on int8
            // and semantically wrong for int32 accumulators.
            codec = dataType.isFloatingPoint ? .int8Blockwise : .raw
            blockSize = bs
        case .topK(let fraction):
            // Deliberately not a per-hop codec. Top-k carries communicator-scoped
            // error-feedback residual state and is only unbiased when the
            // sparsification happens once per operation, at the caller's buffer —
            // re-sparsifying a partially reduced chunk on every hop would drop
            // mass that no residual accounts for. `Communicator.allReduce` routes
            // `.topK` to a sparse all-gather instead; see `topKAllReduceSync`.
            throw MCCLError.unsupportedCompression(
                "topK(fraction: \(fraction)) is not a per-hop wire codec: it is an all-reduce-only scheme whose "
                + "error-feedback residual is scoped to a communicator and stream. Call "
                + "Communicator.allReduce(_:count:dataType:op:compression:stream:) with .topK, or use "
                + ".none, .downcast or .int8Blockwise here."
            )
        }
    }

    /// Upper bound on the encoded size for `elementCount` elements.
    func encodedByteCount(elementCount: Int) -> Int {
        switch codec {
        case .raw:
            return elementCount * dataType.byteWidth
        case .fp16Downcast:
            return elementCount * 2
        case .int8Blockwise:
            let blocks = (elementCount + blockSize - 1) / blockSize
            return blocks * MemoryLayout<Float>.size + elementCount
        }
    }

    /// Encodes `elementCount` elements read from `src` into `dst`.
    /// Returns the number of bytes written.
    @discardableResult
    func encode(from src: UnsafeRawPointer, elementCount: Int, into dst: UnsafeMutableRawPointer) -> Int {
        let width = dataType.byteWidth
        switch codec {
        case .raw:
            let n = elementCount * width
            if n > 0 { dst.copyMemory(from: src, byteCount: n) }
            return n

        case .fp16Downcast:
            // The codec id is only ever chosen for fp32 (see `init`), so the
            // stride is known and the kernel can be a straight vector loop.
            CodecKernels.encodeFP16(from: src, count: elementCount, into: dst)
            return elementCount * 2

        case .int8Blockwise:
            let blocks = (elementCount + blockSize - 1) / blockSize
            let codesOffset = blocks * MemoryLayout<Float>.size
            let vectorable = dataType == .float32
            for b in 0..<blocks {
                let start = b * blockSize
                let end = min(start + blockSize, elementCount)
                let span = end - start
                var absmax: Float = 0
                if vectorable {
                    absmax = CodecKernels.absmaxFP32(src + start * 4, count: span)
                } else {
                    for i in start..<end {
                        let v = abs(ElementIO.loadFloat(src, byteOffset: i * width, dataType))
                        if v.isFinite, v > absmax { absmax = v }
                    }
                }
                let scale = absmax / 127.0
                dst.storeBytes(of: scale, toByteOffset: b * MemoryLayout<Float>.size, as: Float.self)
                if scale == 0 {
                    memset(dst + codesOffset + start, 0, span)
                } else if vectorable {
                    CodecKernels.quantizeFP32(src + start * 4, count: span, inverseScale: 1.0 / scale,
                                              into: dst + codesOffset + start)
                } else {
                    let inv = 1.0 / scale
                    for i in start..<end {
                        let v = ElementIO.loadFloat(src, byteOffset: i * width, dataType)
                        var q = (v * inv).rounded()
                        if q > 127 { q = 127 } else if q < -127 { q = -127 } else if q.isNaN { q = 0 }
                        dst.storeBytes(of: Int8(q), toByteOffset: codesOffset + i, as: Int8.self)
                    }
                }
            }
            return codesOffset + elementCount
        }
    }

    /// Decodes a payload back into the caller's dtype. `codec`/`blockSize` come
    /// from the received frame header, not from the local configuration.
    static func decode(
        payload: UnsafeRawPointer,
        payloadBytes: Int,
        elementCount: Int,
        codec: WireCodecID,
        blockSize: Int,
        dataType: DataType,
        into dst: UnsafeMutableRawPointer
    ) throws {
        let width = dataType.byteWidth
        switch codec {
        case .raw:
            let expected = elementCount * width
            guard payloadBytes == expected else {
                throw MCCLError.protocolViolation("raw payload \(payloadBytes) != expected \(expected)")
            }
            if expected > 0 { dst.copyMemory(from: payload, byteCount: expected) }

        case .fp16Downcast:
            guard payloadBytes == elementCount * 2 else {
                throw MCCLError.protocolViolation("fp16 payload \(payloadBytes) != expected \(elementCount * 2)")
            }
            guard dataType == .float32 else {
                throw MCCLError.protocolViolation("fp16 downcast frame decoded as \(dataType)")
            }
            CodecKernels.decodeFP16(from: payload, count: elementCount, into: dst)

        case .int8Blockwise:
            guard blockSize > 0 else { throw MCCLError.protocolViolation("int8 frame with blockSize 0") }
            let blocks = (elementCount + blockSize - 1) / blockSize
            let codesOffset = blocks * MemoryLayout<Float>.size
            guard payloadBytes == codesOffset + elementCount else {
                throw MCCLError.protocolViolation(
                    "int8 payload \(payloadBytes) != expected \(codesOffset + elementCount)")
            }
            let vectorable = dataType == .float32
            for b in 0..<blocks {
                let scale = payload.loadUnaligned(fromByteOffset: b * MemoryLayout<Float>.size, as: Float.self)
                let start = b * blockSize
                let end = min(start + blockSize, elementCount)
                if vectorable {
                    CodecKernels.dequantizeFP32(payload + codesOffset + start, count: end - start,
                                                scale: scale, into: dst + start * 4)
                    continue
                }
                for i in start..<end {
                    let code = payload.loadUnaligned(fromByteOffset: codesOffset + i, as: Int8.self)
                    ElementIO.storeFloat(Float(code) * scale, into: dst, byteOffset: i * width, dataType)
                }
            }
        }
    }
}

extension WireCompression: CustomStringConvertible {
    public var description: String {
        switch self {
        case .none: return "none"
        case .downcast: return "downcast"
        case .int8Blockwise(let b): return "int8Blockwise(\(b))"
        case .topK(let f): return "topK(\(f))"
        }
    }
}

extension WireCompression: Equatable {
    public static func == (lhs: WireCompression, rhs: WireCompression) -> Bool {
        switch (lhs, rhs) {
        case (.none, .none), (.downcast, .downcast): return true
        case (.int8Blockwise(let a), .int8Blockwise(let b)): return a == b
        case (.topK(let a), .topK(let b)): return a == b
        default: return false
        }
    }
}
