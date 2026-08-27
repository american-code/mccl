import Foundation

// A measurement seam for the wire codecs.
//
// The per-byte cost of encoding and decoding is not a detail: it is the number
// that decides whether compression pays at all on a given fabric. The rule in
// docs/ARCHITECTURE.md §Measured — *a codec is worth applying only when the
// fabric's uncompressed all-reduce rate is below that codec's own encode/decode
// ceiling* — is useless without a way to measure the ceiling on the machine
// that will run the collective.
//
// The codec kernels themselves stay internal; a caller has no business reaching
// around the collectives to encode a buffer. What is public here is only the
// timing seam `mcclbench --codec-bench` drives, and it is public rather than
// `@testable` because the machines whose ceilings matter are lab nodes that run
// executables, not XCTest.

/// One-shot access to a per-hop wire codec, for measuring its throughput.
///
/// `.topK` is not a per-hop codec (see `WireCodec`), so it is measured through
/// `TopKKernelProbe` instead.
public enum CodecKernelProbe {

    /// Upper bound on the encoded size, so a caller can size a wire buffer.
    public static func encodedByteCount(
        _ compression: WireCompression, dataType: DataType, elementCount: Int
    ) throws -> Int {
        try WireCodec(dataType: dataType, compression: compression)
            .encodedByteCount(elementCount: elementCount)
    }

    /// Encodes exactly as a ring step would, and returns the bytes written.
    @discardableResult
    public static func encode(
        _ compression: WireCompression, dataType: DataType,
        from source: UnsafeRawPointer, elementCount: Int, into destination: UnsafeMutableRawPointer
    ) throws -> Int {
        try WireCodec(dataType: dataType, compression: compression)
            .encode(from: source, elementCount: elementCount, into: destination)
    }

    /// Decodes a payload the way a receiver would, taking the codec parameters
    /// from `compression` rather than from a frame header.
    public static func decode(
        _ compression: WireCompression, dataType: DataType,
        payload: UnsafeRawPointer, payloadBytes: Int, elementCount: Int,
        into destination: UnsafeMutableRawPointer
    ) throws {
        let codec = try WireCodec(dataType: dataType, compression: compression)
        try WireCodec.decode(
            payload: payload, payloadBytes: payloadBytes, elementCount: elementCount,
            codec: codec.codec, blockSize: codec.blockSize, dataType: dataType,
            into: destination)
    }
}

/// One-shot access to the top-k sparsifier, for measuring its throughput.
///
/// Top-k replaces the algorithm rather than the payload, so "encode" here is
/// what a rank does once per all-reduce — fold the residual back in, select the
/// `k` largest magnitudes, pack them, keep the rest — and "decode" is the
/// accumulation of the gathered blocks onto the caller's buffer. The residual
/// lives in the probe, exactly as it lives in a `Communicator`, because
/// measuring the first call only would measure a zero residual.
public final class TopKKernelProbe {
    public let elementCount: Int
    public let dataType: DataType
    public let fraction: Double
    /// Entries a block carries — `⌈fraction·count⌉`.
    public let nonZeros: Int
    /// Bytes one rank's sparse block occupies.
    public let blockByteCount: Int

    private var residual: [Float]

    public init(elementCount: Int, dataType: DataType, fraction: Double) {
        self.elementCount = elementCount
        self.dataType = dataType
        self.fraction = fraction
        self.nonZeros = TopK.count(elementCount: elementCount, fraction: fraction)
        self.blockByteCount = TopKBlock.byteCount(
            nonZeros: nonZeros, valueWidth: dataType.byteWidth)
        self.residual = [Float](repeating: 0, count: elementCount)
    }

    /// Sparsifies `source` into `destination`, updating the residual. Returns
    /// the bytes written.
    @discardableResult
    public func encode(
        from source: UnsafeRawPointer, into destination: UnsafeMutableRawPointer
    ) -> Int {
        TopKKernels.encodeBlock(
            source: source, count: elementCount, dataType: dataType,
            fraction: fraction, residual: &residual, into: destination)
    }

    /// Sums `blockCount` gathered blocks onto a zeroed `destination` — the
    /// receiving half of a top-k all-reduce over `blockCount` ranks.
    public func decode(
        blockCount: Int, from payload: UnsafeRawPointer, into destination: UnsafeMutableRawPointer
    ) throws {
        memset(destination, 0, elementCount * dataType.byteWidth)
        for block in 0..<blockCount {
            try TopKBlock.accumulate(
                payload: payload + block * blockByteCount, payloadBytes: blockByteCount,
                elementCount: elementCount, dataType: dataType, into: destination)
        }
    }
}
