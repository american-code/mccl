import XCTest
@testable import MCCL

final class CompressionTests: XCTestCase {

    /// Encodes then decodes through the real wire codec and returns the result.
    private func roundTrip(
        _ values: [Float], dataType: DataType, compression: WireCompression
    ) throws -> [Float] {
        let codec = try WireCodec(dataType: dataType, compression: compression)
        let count = values.count
        let source = Buf.allocate(count: count, dataType: dataType)
        defer { source.deallocate() }
        Buf.fill(source, count: count, dataType: dataType) { values[$0] }

        let wire = ScratchBuffer(capacity: max(16, codec.encodedByteCount(elementCount: count)))
        let bytes = codec.encode(from: source.baseAddress!, elementCount: count, into: wire.base)
        XCTAssertLessThanOrEqual(bytes, codec.encodedByteCount(elementCount: count))

        let destination = Buf.allocate(count: count, dataType: dataType)
        defer { destination.deallocate() }
        try WireCodec.decode(
            payload: wire.base, payloadBytes: bytes, elementCount: count,
            codec: codec.codec, blockSize: codec.blockSize,
            dataType: dataType, into: destination.baseAddress!)
        return Buf.read(destination, count: count, dataType: dataType)
    }

    // MARK: - none

    func testNoneIsExactForEveryDataType() throws {
        for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
            let values = (0..<300).map { Float(($0 % 61) - 30) }
            let decoded = try roundTrip(values, dataType: dataType, compression: .none)
            XCTAssertEqual(decoded, values, "\(dataType) must round-trip exactly with .none")
        }
    }

    // MARK: - downcast

    func testDowncastHalvesTheWireAndStaysWithinFP16Precision() throws {
        let codec = try WireCodec(dataType: .float32, compression: .downcast)
        XCTAssertEqual(codec.codec, .fp16Downcast)
        XCTAssertEqual(codec.encodedByteCount(elementCount: 1024), 2048,
                       "fp32 -> fp16 must halve the bytes on the wire")

        var generator = SystemRandomNumberGenerator()
        let values = (0..<4096).map { _ in Float.random(in: -1000...1000, using: &generator) }
        let decoded = try roundTrip(values, dataType: .float32, compression: .downcast)

        // fp16 carries 10 explicit mantissa bits: relative error <= 2^-11.
        let bound: Float = 1.0 / 2048.0
        for (original, result) in zip(values, decoded) {
            XCTAssertLessThanOrEqual(abs(result - original), abs(original) * bound + 1e-7,
                                     "fp16 downcast error out of bound for \(original)")
        }
    }

    func testDowncastIsANoOpForNonFP32DataTypes() throws {
        for dataType in [DataType.float16, .bfloat16, .int32, .int8] {
            let codec = try WireCodec(dataType: dataType, compression: .downcast)
            XCTAssertEqual(codec.codec, .raw, "\(dataType) has nothing to downcast to")
            let values = (0..<128).map { Float($0 % 17) }
            XCTAssertEqual(try roundTrip(values, dataType: dataType, compression: .downcast), values)
        }
    }

    // MARK: - int8 blockwise

    func testInt8BlockwiseStaysWithinHalfAQuantumOfBlockAbsmax() throws {
        let blockSize = 64
        var generator = SystemRandomNumberGenerator()
        // Deliberately varied block scales: block k is ~10^(k mod 3) in size,
        // which is exactly the case a single global scale would ruin.
        let count = 64 * 12 + 17
        let values = (0..<count).map { i -> Float in
            let magnitude = Float(pow(10.0, Double((i / blockSize) % 3)))
            return Float.random(in: -magnitude...magnitude, using: &generator)
        }
        let decoded = try roundTrip(values, dataType: .float32,
                                    compression: .int8Blockwise(blockSize: blockSize))

        let blocks = (count + blockSize - 1) / blockSize
        for b in 0..<blocks {
            let range = (b * blockSize)..<min((b + 1) * blockSize, count)
            let absmax = range.map { abs(values[$0]) }.max() ?? 0
            // scale = absmax/127, so nearest-integer rounding is <= scale/2.
            let bound = absmax / 254.0 * 1.001 + 1e-6
            for i in range {
                XCTAssertLessThanOrEqual(abs(decoded[i] - values[i]), bound,
                                         "block \(b) element \(i): \(values[i]) -> \(decoded[i])")
            }
        }
    }

    func testInt8BlockwiseWireSize() throws {
        let codec = try WireCodec(dataType: .float32, compression: .int8Blockwise(blockSize: 256))
        // 1024 elements -> 4 scales (16 bytes) + 1024 codes, vs 4096 raw.
        XCTAssertEqual(codec.encodedByteCount(elementCount: 1024), 16 + 1024)
    }

    func testInt8BlockwiseHandlesAllZeroBlocks() throws {
        let values = [Float](repeating: 0, count: 200)
        let decoded = try roundTrip(values, dataType: .float32, compression: .int8Blockwise(blockSize: 64))
        XCTAssertEqual(decoded, values)
    }

    func testInt8BlockwiseWorksForHalfPrecisionDataTypes() throws {
        for dataType in [DataType.float16, .bfloat16] {
            let values = (0..<256).map { Float($0 % 33) - 16 }
            let decoded = try roundTrip(values, dataType: dataType,
                                        compression: .int8Blockwise(blockSize: 64))
            for (original, result) in zip(values, decoded) {
                XCTAssertLessThanOrEqual(abs(result - original), 16.0 / 254.0 + 0.15,
                                         "\(dataType): \(original) -> \(result)")
            }
        }
    }

    func testInt8BlockwiseIsANoOpForIntegerDataTypes() throws {
        for dataType in [DataType.int32, .int8] {
            let codec = try WireCodec(dataType: dataType, compression: .int8Blockwise(blockSize: 64))
            XCTAssertEqual(codec.codec, .raw, "quantising an integer accumulator would be wrong")
            let values = (0..<100).map { Float($0 - 50) }
            XCTAssertEqual(try roundTrip(values, dataType: dataType,
                                         compression: .int8Blockwise(blockSize: 64)), values)
        }
    }

    func testInvalidBlockSizeIsRejected() {
        XCTAssertThrowsError(try WireCodec(dataType: .float32, compression: .int8Blockwise(blockSize: 0)))
    }

    // MARK: - topK

    func testTopKThrowsInsteadOfCrashing() {
        XCTAssertThrowsError(try WireCodec(dataType: .float32, compression: .topK(fraction: 0.01))) { error in
            guard case MCCLError.unsupportedCompression(let message) = error else {
                return XCTFail("expected .unsupportedCompression, got \(error)")
            }
            XCTAssertTrue(message.contains("error-feedback"),
                          "the error should say why top-k is unavailable: \(message)")
        }
    }

    /// `.topK` is reachable through the ordinary collective API — it is routed
    /// to the sparse all-gather path rather than the per-hop codec. The
    /// error-feedback behaviour itself lives in `TopKTests`.
    func testTopKSurfacesThroughTheCollectiveAPI() async throws {
        try await Ranks.run(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 16, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: 16, dataType: .float32) { Float(comm.rank + 1) * Float($0 + 1) }
            try await comm.allReduce(buffer, count: 16, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 1.0))
            for (i, value) in Buf.read(buffer, count: 16, dataType: .float32).enumerated() {
                XCTAssertEqual(value, 3 * Float(i + 1), accuracy: 1e-4)
            }
        }
    }

    // MARK: - dtype conversions

    func testBFloat16RoundsToNearestEven() {
        for value in [Float(0), 1, -1, 3.5, -2.25, 65504, 1e-8, 12345.0] {
            let recovered = BFloat16.toFloat(BFloat16.fromFloat(value))
            XCTAssertLessThanOrEqual(abs(recovered - value), abs(value) / 256.0 + 1e-12,
                                     "bf16 round-trip of \(value) gave \(recovered)")
        }
        XCTAssertTrue(BFloat16.toFloat(BFloat16.fromFloat(Float.nan)).isNaN)
    }
}
