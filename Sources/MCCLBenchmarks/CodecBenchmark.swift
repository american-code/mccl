import Foundation
import MCCL

// `mcclbench --codec-bench`: what each wire codec costs per byte, with no
// network under it at all.
//
// The collective sweep answers "is this codec worth it on this cable?"; this
// mode answers the half of that question that belongs to the machine rather
// than the fabric. docs/ARCHITECTURE.md §Measured states the rule the two
// combine into — *a codec pays only when the fabric's uncompressed all-reduce
// rate is below the codec's own encode/decode ceiling* — and until this mode
// existed the ceilings were read off the plateaus of a distributed sweep, which
// is an inference and not a measurement.
//
// The numbers are per *payload* byte: a codec that halves the wire still has to
// look at every byte the caller handed it, so payload bytes over encode time is
// the rate that competes with the link. `round-trip` is the sustained rate
// through both halves, `1/(1/encode + 1/decode)`, which is what a rank
// achieves when it is encoding what it sends and decoding what it receives.

public struct CodecBenchRow: Sendable {
    public let payloadBytes: Int
    public let elementCount: Int
    public let codec: String
    /// Bytes the codec put on the wire for one call.
    public let encodedBytes: Int
    public let iterations: Int
    /// Best per-call encode/decode times, seconds.
    public let encodeSeconds: Double
    public let decodeSeconds: Double
    public let failure: String?

    public var encodeBytesPerSecond: Double {
        encodeSeconds > 0 ? Double(payloadBytes) / encodeSeconds : 0
    }
    public var decodeBytesPerSecond: Double {
        decodeSeconds > 0 ? Double(payloadBytes) / decodeSeconds : 0
    }
    /// The ceiling the compression rule is stated against.
    public var roundTripBytesPerSecond: Double {
        let total = encodeSeconds + decodeSeconds
        return total > 0 ? Double(payloadBytes) / total : 0
    }
    /// Payload bytes per wire byte — 2.00x for downcast, and so on.
    public var compressionRatio: Double {
        encodedBytes > 0 ? Double(payloadBytes) / Double(encodedBytes) : 0
    }
    /// Nanoseconds of CPU per element, both halves. The unit the scalar loops
    /// were originally costed in.
    public var nanosecondsPerElement: Double {
        elementCount > 0 ? (encodeSeconds + decodeSeconds) / Double(elementCount) * 1e9 : 0
    }
}

public enum CodecBenchRunner {

    /// Sweeps every (size, codec) point. No fabric, no ranks: one buffer is
    /// encoded and decoded in this thread.
    ///
    /// Every point is run `repeats` times and the *best* is kept, for the same
    /// reason the distributed sweep is a best-of — a shared machine's worst
    /// case measures the other workload, not the codec.
    public static func run(
        _ options: BenchOptions,
        repeats: Int = 5,
        onRow: ((CodecBenchRow) -> Void)? = nil
    ) -> [CodecBenchRow] {
        var rows: [CodecBenchRow] = []
        for size in options.sizes {
            for codec in options.codecs {
                guard codec.applies(to: .allReduce, dataType: options.dataType) else { continue }
                let row = measure(size: size, codec: codec, options: options, repeats: repeats)
                rows.append(row)
                onRow?(row)
            }
        }
        return rows
    }

    static func measure(
        size: Int, codec: BenchCodec, options: BenchOptions, repeats: Int
    ) -> CodecBenchRow {
        let width = options.dataType.byteWidth
        let count = max(1, size / width)
        let payloadBytes = count * width
        let iterations = options.iterations(forMessageBytes: size)

        func failed(_ message: String) -> CodecBenchRow {
            CodecBenchRow(payloadBytes: payloadBytes, elementCount: count, codec: codec.description,
                          encodedBytes: 0, iterations: 0, encodeSeconds: 0, decodeSeconds: 0,
                          failure: message)
        }

        let source = UnsafeMutableRawBufferPointer.allocate(byteCount: payloadBytes, alignment: 64)
        let destination = UnsafeMutableRawBufferPointer.allocate(byteCount: payloadBytes, alignment: 64)
        defer { source.deallocate(); destination.deallocate() }
        fill(source, count: count, dataType: options.dataType)
        destination.initializeMemory(as: UInt8.self, repeating: 0)

        do {
            let harness = try makeHarness(codec: codec, count: count, options: options)
            defer { harness.release() }

            var bestEncode = Double.infinity
            var bestDecode = Double.infinity
            for _ in 0..<max(1, repeats) {
                for _ in 0..<max(1, options.warmup) {
                    try harness.encode(source.baseAddress!)
                    try harness.decode(destination.baseAddress!)
                }
                var start = DispatchTime.now().uptimeNanoseconds
                for _ in 0..<iterations { try harness.encode(source.baseAddress!) }
                bestEncode = Swift.min(
                    bestEncode,
                    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 / Double(iterations))

                start = DispatchTime.now().uptimeNanoseconds
                for _ in 0..<iterations { try harness.decode(destination.baseAddress!) }
                bestDecode = Swift.min(
                    bestDecode,
                    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 / Double(iterations))
            }
            return CodecBenchRow(
                payloadBytes: payloadBytes, elementCount: count, codec: codec.description,
                encodedBytes: harness.wireBytes, iterations: iterations,
                encodeSeconds: bestEncode, decodeSeconds: bestDecode, failure: nil)
        } catch {
            return failed("\(error)")
        }
    }

    // MARK: - Per-codec harnesses

    /// What one codec needs to be exercised end to end: a wire buffer, an
    /// encode, and a decode. Top-k needs a residual and a gathered block per
    /// rank as well, so it gets its own implementation rather than a special
    /// case in the timing loop.
    final class Harness {
        let wireBytes: Int
        let encode: (UnsafeRawPointer) throws -> Void
        let decode: (UnsafeMutableRawPointer) throws -> Void
        let release: () -> Void

        init(wireBytes: Int,
             encode: @escaping (UnsafeRawPointer) throws -> Void,
             decode: @escaping (UnsafeMutableRawPointer) throws -> Void,
             release: @escaping () -> Void) {
            self.wireBytes = wireBytes
            self.encode = encode
            self.decode = decode
            self.release = release
        }
    }

    static func makeHarness(codec: BenchCodec, count: Int, options: BenchOptions) throws -> Harness {
        let dataType = options.dataType
        if case .topK(let fraction) = codec.compression {
            // Top-k's receiving half sums one block per rank, so the world size
            // is part of the cost. The cluster runs n = 2.
            let ranks = Swift.max(2, options.worldSize)
            let probe = TopKKernelProbe(elementCount: count, dataType: dataType, fraction: fraction)
            let wire = UnsafeMutableRawBufferPointer.allocate(
                byteCount: Swift.max(16, probe.blockByteCount * ranks), alignment: 64)
            wire.initializeMemory(as: UInt8.self, repeating: 0)
            return Harness(
                wireBytes: probe.blockByteCount * (ranks - 1),
                encode: { source in
                    let bytes = probe.encode(from: source, into: wire.baseAddress!)
                    // Every rank sends the same shape; copy it into the other
                    // ranks' slots so the decode half has something to sum.
                    for peer in 1..<ranks {
                        (wire.baseAddress! + peer * probe.blockByteCount)
                            .copyMemory(from: wire.baseAddress!, byteCount: bytes)
                    }
                },
                decode: { destination in
                    try probe.decode(blockCount: ranks, from: wire.baseAddress!, into: destination)
                },
                release: { wire.deallocate() })
        }

        let compression = codec.compression
        let capacity = try CodecKernelProbe.encodedByteCount(
            compression, dataType: dataType, elementCount: count)
        let wire = UnsafeMutableRawBufferPointer.allocate(
            byteCount: Swift.max(16, capacity), alignment: 64)
        wire.initializeMemory(as: UInt8.self, repeating: 0)
        var written = capacity
        return Harness(
            wireBytes: capacity,
            encode: { source in
                written = try CodecKernelProbe.encode(
                    compression, dataType: dataType, from: source,
                    elementCount: count, into: wire.baseAddress!)
            },
            decode: { destination in
                try CodecKernelProbe.decode(
                    compression, dataType: dataType, payload: wire.baseAddress!,
                    payloadBytes: written, elementCount: count, into: destination)
            },
            release: { wire.deallocate() })
    }

    /// Values with a wide dynamic range and no repeating block structure: a
    /// buffer of one constant would let int8's per-block absmax scan predict
    /// itself, and top-k would have no real choice to make.
    static func fill(_ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType) {
        guard let base = buffer.baseAddress else { return }
        let width = dataType.byteWidth
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(state >> 40) / Float(1 << 24) - 0.5      // -0.5 ..< 0.5
            let value = unit * Float(1 << ((i / 4096) % 8))            // varied block scales
            ElementIOBridge.store(value, into: base, byteOffset: i * width, dataType)
        }
    }
}

/// `ElementIO` is internal to MCCL and rightly so; the bench only needs to
/// write a float in the caller's dtype, which is four lines.
enum ElementIOBridge {
    static func store(_ value: Float, into p: UnsafeMutableRawPointer, byteOffset: Int, _ dt: DataType) {
        switch dt {
        case .float32: p.storeBytes(of: value, toByteOffset: byteOffset, as: Float.self)
        case .float16: p.storeBytes(of: Float16(value), toByteOffset: byteOffset, as: Float16.self)
        case .bfloat16:
            p.storeBytes(of: UInt16(truncatingIfNeeded: value.bitPattern >> 16),
                         toByteOffset: byteOffset, as: UInt16.self)
        case .int32: p.storeBytes(of: Int32(value.rounded()), toByteOffset: byteOffset, as: Int32.self)
        case .int8: p.storeBytes(of: Int8(truncatingIfNeeded: Int(value.rounded())),
                                 toByteOffset: byteOffset, as: Int8.self)
        }
    }
}

// MARK: - Table rendering

public enum CodecBenchTable {
    public static func header(_ options: BenchOptions) -> String {
        options.csv
            ? "payload_bytes,codec,wire_bytes,ratio,iterations,encode_GB_s,decode_GB_s,"
              + "roundtrip_GB_s,ns_per_element"
            : BenchTable.pad("size", 12, left: false) + "  " + BenchTable.pad("codec", 14)
              + BenchTable.pad("ratio", 8, left: false) + "  "
              + BenchTable.pad("GB/s enc", 10, left: false) + "  "
              + BenchTable.pad("GB/s dec", 10, left: false) + "  "
              + BenchTable.pad("GB/s r/t", 10, left: false) + "  "
              + BenchTable.pad("ns/elem", 9, left: false)
    }

    public static func rule(_ options: BenchOptions) -> String? {
        options.csv ? nil : String(repeating: "-", count: 12 + 2 + 14 + 8 + 2 + 10 + 2 + 10 + 2 + 10 + 2 + 9)
    }

    public static func render(_ row: CodecBenchRow, options: BenchOptions) -> String {
        if options.csv {
            guard row.failure == nil else {
                return "\(row.payloadBytes),\(row.codec),0,,0,,,,\"\(row.failure!)\""
            }
            return [
                String(row.payloadBytes), row.codec, String(row.encodedBytes),
                String(format: "%.3f", row.compressionRatio), String(row.iterations),
                String(format: "%.4f", row.encodeBytesPerSecond / 1e9),
                String(format: "%.4f", row.decodeBytesPerSecond / 1e9),
                String(format: "%.4f", row.roundTripBytesPerSecond / 1e9),
                String(format: "%.4f", row.nanosecondsPerElement),
            ].joined(separator: ",")
        }
        let prefix = BenchTable.pad(BenchTable.bytes(row.payloadBytes), 12, left: false) + "  "
            + BenchTable.pad(row.codec, 14)
        guard row.failure == nil else { return prefix + "—  " + row.failure! }
        return prefix
            + BenchTable.pad(String(format: "%.2fx", row.compressionRatio), 8, left: false) + "  "
            + BenchTable.pad(String(format: "%.3f", row.encodeBytesPerSecond / 1e9), 10, left: false) + "  "
            + BenchTable.pad(String(format: "%.3f", row.decodeBytesPerSecond / 1e9), 10, left: false) + "  "
            + BenchTable.pad(String(format: "%.3f", row.roundTripBytesPerSecond / 1e9), 10, left: false) + "  "
            + BenchTable.pad(String(format: "%.2f", row.nanosecondsPerElement), 9, left: false)
    }

    /// The line the compression rule is actually stated in: one ceiling per
    /// codec, taken at the size where the codec does best.
    public static func ceilings(_ rows: [CodecBenchRow]) -> [(codec: String, ceiling: Double, at: Int)] {
        var order: [String] = []
        for row in rows where row.failure == nil {
            if !order.contains(row.codec) { order.append(row.codec) }
        }
        return order.compactMap { codec in
            let candidates = rows.filter { $0.codec == codec && $0.failure == nil }
            guard let best = candidates.max(by: { $0.roundTripBytesPerSecond < $1.roundTripBytesPerSecond })
            else { return nil }
            return (codec, best.roundTripBytesPerSecond, best.payloadBytes)
        }
    }
}
