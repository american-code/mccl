import XCTest
@testable import MCCL

/// The wire format is a compatibility contract, and vectorising the codecs was
/// not allowed to touch it: a rank running the SIMD encoder has to be
/// understood by a peer still running the scalar one, in both directions and
/// down to the byte.
///
/// `Reference` below is the pre-vectorisation encoder and decoder, copied
/// verbatim from the scalar implementation. It is deliberately naive — it is
/// the thing the fast path is checked against, so any cleverness in it would
/// weaken the check.
final class CompressionInteropTests: XCTestCase {

    // MARK: - The scalar reference

    enum Reference {
        static func encodeFP16(_ src: UnsafeRawPointer, count: Int, into dst: UnsafeMutableRawPointer) {
            for i in 0..<count {
                let v = src.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                dst.storeBytes(of: Float16(v), toByteOffset: i * 2, as: Float16.self)
            }
        }

        static func decodeFP16(_ payload: UnsafeRawPointer, count: Int, into dst: UnsafeMutableRawPointer) {
            for i in 0..<count {
                let h = payload.loadUnaligned(fromByteOffset: i * 2, as: Float16.self)
                dst.storeBytes(of: Float(h), toByteOffset: i * 4, as: Float.self)
            }
        }

        static func encodeInt8(
            _ src: UnsafeRawPointer, count: Int, blockSize: Int, dataType: DataType,
            into dst: UnsafeMutableRawPointer
        ) -> Int {
            let width = dataType.byteWidth
            let blocks = (count + blockSize - 1) / blockSize
            let codesOffset = blocks * MemoryLayout<Float>.size
            for b in 0..<blocks {
                let start = b * blockSize
                let end = min(start + blockSize, count)
                var absmax: Float = 0
                for i in start..<end {
                    let v = abs(ElementIO.loadFloat(src, byteOffset: i * width, dataType))
                    if v.isFinite, v > absmax { absmax = v }
                }
                let scale = absmax / 127.0
                dst.storeBytes(of: scale, toByteOffset: b * MemoryLayout<Float>.size, as: Float.self)
                if scale == 0 {
                    for i in start..<end {
                        dst.storeBytes(of: Int8(0), toByteOffset: codesOffset + i, as: Int8.self)
                    }
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
            return codesOffset + count
        }

        static func decodeInt8(
            _ payload: UnsafeRawPointer, count: Int, blockSize: Int, dataType: DataType,
            into dst: UnsafeMutableRawPointer
        ) {
            let width = dataType.byteWidth
            let blocks = (count + blockSize - 1) / blockSize
            let codesOffset = blocks * MemoryLayout<Float>.size
            for b in 0..<blocks {
                let scale = payload.loadUnaligned(
                    fromByteOffset: b * MemoryLayout<Float>.size, as: Float.self)
                let start = b * blockSize
                let end = min(start + blockSize, count)
                for i in start..<end {
                    let code = payload.loadUnaligned(fromByteOffset: codesOffset + i, as: Int8.self)
                    ElementIO.storeFloat(Float(code) * scale, into: dst, byteOffset: i * width, dataType)
                }
            }
        }

        /// The pre-vectorisation top-k packer: indices, then values, through
        /// the per-element dtype switch.
        static func writeTopKBlock(
            indices: [Int], values: [Float], dataType: DataType, into dst: UnsafeMutableRawPointer
        ) -> Int {
            let width = dataType.byteWidth
            dst.storeBytes(of: UInt32(indices.count).littleEndian, toByteOffset: 0, as: UInt32.self)
            var offset = TopKBlock.headerBytes
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
    }

    // MARK: - Fixtures

    /// Deterministic values spanning several orders of magnitude, with the
    /// awkward cases seeded in: exact ties for the quantiser to round, zeros,
    /// a NaN and both infinities.
    private func values(count: Int, seed: UInt64 = 0x243F_6A88_85A3_08D3) -> [Float] {
        var state = seed
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(state >> 40) / Float(1 << 24) - 0.5
            switch i % 23 {
            case 7: out[i] = 0
            case 11: out[i] = Float((i % 250) - 125) + 0.5    // lands on a rounding tie
            default: out[i] = unit * Float(1 << ((i / 64) % 7))
            }
        }
        if count > 40 {
            out[3] = .nan
            out[17] = .infinity
            out[29] = -.infinity
        }
        return out
    }

    private func buffer(_ values: [Float]) -> UnsafeMutableRawBufferPointer {
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: max(16, values.count * 4), alignment: 64)
        for (i, v) in values.enumerated() {
            buffer.baseAddress!.storeBytes(of: v, toByteOffset: i * 4, as: Float.self)
        }
        return buffer
    }

    private func bytes(_ pointer: UnsafeRawPointer, _ count: Int) -> [UInt8] {
        Array(UnsafeRawBufferPointer(start: pointer, count: count))
    }

    /// Element counts that straddle every SIMD boundary the kernels have: the
    /// empty tail, a partial vector, exactly one vector, a partial chunk, and
    /// sizes either side of the quantiser's 4096-element scratch chunk.
    private let tailLengths = [1, 2, 7, 8, 9, 15, 16, 17, 63, 255, 256, 257, 1000, 4095, 4096, 4097, 9999]

    // MARK: - Byte-for-byte equality with the scalar encoder

    func testVectorFP16EncoderProducesTheScalarEncodersBytes() throws {
        for count in tailLengths {
            let source = buffer(values(count: count))
            defer { source.deallocate() }
            let codec = try WireCodec(dataType: .float32, compression: .downcast)

            let fast = ScratchBuffer(capacity: max(16, count * 2))
            let slow = ScratchBuffer(capacity: max(16, count * 2))
            let written = codec.encode(from: source.baseAddress!, elementCount: count, into: fast.base)
            Reference.encodeFP16(source.baseAddress!, count: count, into: slow.base)

            XCTAssertEqual(written, count * 2)
            XCTAssertEqual(bytes(fast.base, count * 2), bytes(slow.base, count * 2),
                           "fp16 frame differs from the scalar encoder at count \(count)")
        }
    }

    func testVectorInt8EncoderProducesTheScalarEncodersBytes() throws {
        for blockSize in [1, 7, 64, 256, 4096] {
            for count in tailLengths {
                let source = buffer(values(count: count, seed: UInt64(blockSize) &* 0x9E37_79B9))
                defer { source.deallocate() }
                let codec = try WireCodec(dataType: .float32,
                                          compression: .int8Blockwise(blockSize: blockSize))
                let size = codec.encodedByteCount(elementCount: count)
                let fast = ScratchBuffer(capacity: max(16, size))
                let slow = ScratchBuffer(capacity: max(16, size))

                let written = codec.encode(from: source.baseAddress!, elementCount: count, into: fast.base)
                let reference = Reference.encodeInt8(source.baseAddress!, count: count,
                                                     blockSize: blockSize, dataType: .float32,
                                                     into: slow.base)
                XCTAssertEqual(written, reference)
                XCTAssertEqual(bytes(fast.base, written), bytes(slow.base, reference),
                               "int8/\(blockSize) frame differs from the scalar encoder at count \(count)")
            }
        }
    }

    /// The scale is the part of the int8 frame a receiver cannot sanity-check,
    /// so the rule that produces it — largest *finite* magnitude, non-finite
    /// elements skipped — has to survive vectorisation exactly.
    func testInt8BlockScaleStillIgnoresNonFiniteElements() throws {
        let count = 64
        var raw = [Float](repeating: 2, count: count)
        raw[5] = .infinity
        raw[6] = .nan
        raw[7] = -8            // the largest finite magnitude
        let source = buffer(raw)
        defer { source.deallocate() }

        let codec = try WireCodec(dataType: .float32, compression: .int8Blockwise(blockSize: count))
        let wire = ScratchBuffer(capacity: codec.encodedByteCount(elementCount: count))
        codec.encode(from: source.baseAddress!, elementCount: count, into: wire.base)

        let scale = wire.base.loadUnaligned(as: Float.self)
        XCTAssertEqual(scale, 8.0 / 127.0, accuracy: 1e-9,
                       "an infinity in the block must not become the block's scale")
        XCTAssertEqual(CodecKernels.absmaxFP32(source.baseAddress!, count: count), 8)
        // NaN quantises to 0, ±infinity saturates, exactly as the scalar path did.
        let codesOffset = MemoryLayout<Float>.size
        XCTAssertEqual(wire.base.loadUnaligned(fromByteOffset: codesOffset + 6, as: Int8.self), 0)
        XCTAssertEqual(wire.base.loadUnaligned(fromByteOffset: codesOffset + 5, as: Int8.self), 127)
    }

    // MARK: - Cross-version decode, both directions

    func testScalarEncodedFramesDecodeThroughTheVectorDecoder() throws {
        for count in tailLengths {
            let original = values(count: count)
            let source = buffer(original)
            defer { source.deallocate() }

            // fp16: a scalar sender's frame, decoded here.
            let half = ScratchBuffer(capacity: max(16, count * 2))
            Reference.encodeFP16(source.baseAddress!, count: count, into: half.base)
            let mine = UnsafeMutableRawBufferPointer.allocate(byteCount: max(16, count * 4), alignment: 64)
            let theirs = UnsafeMutableRawBufferPointer.allocate(byteCount: max(16, count * 4), alignment: 64)
            defer { mine.deallocate(); theirs.deallocate() }
            try WireCodec.decode(payload: half.base, payloadBytes: count * 2, elementCount: count,
                                 codec: .fp16Downcast, blockSize: 0, dataType: .float32,
                                 into: mine.baseAddress!)
            Reference.decodeFP16(half.base, count: count, into: theirs.baseAddress!)
            XCTAssertEqual(bytes(mine.baseAddress!, count * 4), bytes(theirs.baseAddress!, count * 4),
                           "fp16 decode disagrees with the scalar decoder at count \(count)")

            // int8: same, through a block size that leaves a ragged last block.
            let blockSize = 64
            let blocks = (count + blockSize - 1) / blockSize
            let payloadBytes = blocks * 4 + count
            let quantised = ScratchBuffer(capacity: max(16, payloadBytes))
            _ = Reference.encodeInt8(source.baseAddress!, count: count, blockSize: blockSize,
                                     dataType: .float32, into: quantised.base)
            try WireCodec.decode(payload: quantised.base, payloadBytes: payloadBytes, elementCount: count,
                                 codec: .int8Blockwise, blockSize: blockSize, dataType: .float32,
                                 into: mine.baseAddress!)
            Reference.decodeInt8(quantised.base, count: count, blockSize: blockSize,
                                 dataType: .float32, into: theirs.baseAddress!)
            XCTAssertEqual(bytes(mine.baseAddress!, count * 4), bytes(theirs.baseAddress!, count * 4),
                           "int8 decode disagrees with the scalar decoder at count \(count)")
        }
    }

    func testVectorEncodedFramesDecodeThroughTheScalarDecoder() throws {
        let count = 5000
        let original = values(count: count)
        let source = buffer(original)
        defer { source.deallocate() }
        let decoded = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        defer { decoded.deallocate() }

        let downcast = try WireCodec(dataType: .float32, compression: .downcast)
        let half = ScratchBuffer(capacity: count * 2)
        downcast.encode(from: source.baseAddress!, elementCount: count, into: half.base)
        Reference.decodeFP16(half.base, count: count, into: decoded.baseAddress!)
        for i in 0..<count where original[i].isFinite {
            let result = decoded.baseAddress!.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
            XCTAssertLessThanOrEqual(abs(result - original[i]), abs(original[i]) / 2048 + 1e-7,
                                     "element \(i) outside the fp16 bound after a scalar decode")
        }

        let blockSize = 256
        let int8 = try WireCodec(dataType: .float32, compression: .int8Blockwise(blockSize: blockSize))
        let quantised = ScratchBuffer(capacity: int8.encodedByteCount(elementCount: count))
        int8.encode(from: source.baseAddress!, elementCount: count, into: quantised.base)
        Reference.decodeInt8(quantised.base, count: count, blockSize: blockSize,
                             dataType: .float32, into: decoded.baseAddress!)
        for b in 0..<((count + blockSize - 1) / blockSize) {
            let range = (b * blockSize)..<min((b + 1) * blockSize, count)
            let absmax = range.compactMap { original[$0].isFinite ? abs(original[$0]) : nil }.max() ?? 0
            for i in range where original[i].isFinite {
                let result = decoded.baseAddress!.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                XCTAssertLessThanOrEqual(abs(result - original[i]), absmax / 254 * 1.001 + 1e-6,
                                         "element \(i) outside the int8 bound after a scalar decode")
            }
        }
    }

    // MARK: - Round-trip bounds through the vectorised path

    func testVectorRoundTripHoldsTheDocumentedBoundsAtEverySIMDTail() throws {
        for count in tailLengths {
            let original = values(count: count)
            let source = buffer(original)
            let decoded = UnsafeMutableRawBufferPointer.allocate(
                byteCount: max(16, count * 4), alignment: 64)
            defer { source.deallocate(); decoded.deallocate() }

            let downcast = try WireCodec(dataType: .float32, compression: .downcast)
            let half = ScratchBuffer(capacity: max(16, count * 2))
            let written = downcast.encode(from: source.baseAddress!, elementCount: count, into: half.base)
            try WireCodec.decode(payload: half.base, payloadBytes: written, elementCount: count,
                                 codec: .fp16Downcast, blockSize: 0, dataType: .float32,
                                 into: decoded.baseAddress!)
            for i in 0..<count where original[i].isFinite && abs(original[i]) < 65504 {
                let result = decoded.baseAddress!.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                XCTAssertLessThanOrEqual(abs(result - original[i]), abs(original[i]) / 2048 + 1e-7,
                                         "count \(count), element \(i): fp16 bound")
            }

            for blockSize in [7, 256] {
                let int8 = try WireCodec(dataType: .float32,
                                         compression: .int8Blockwise(blockSize: blockSize))
                let wire = ScratchBuffer(capacity: max(16, int8.encodedByteCount(elementCount: count)))
                let size = int8.encode(from: source.baseAddress!, elementCount: count, into: wire.base)
                try WireCodec.decode(payload: wire.base, payloadBytes: size, elementCount: count,
                                     codec: .int8Blockwise, blockSize: blockSize, dataType: .float32,
                                     into: decoded.baseAddress!)
                for b in 0..<((count + blockSize - 1) / blockSize) {
                    let range = (b * blockSize)..<min((b + 1) * blockSize, count)
                    let absmax = range.compactMap {
                        original[$0].isFinite ? abs(original[$0]) : nil
                    }.max() ?? 0
                    for i in range where original[i].isFinite {
                        let result = decoded.baseAddress!.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                        XCTAssertLessThanOrEqual(abs(result - original[i]), absmax / 254 * 1.001 + 1e-6,
                                                 "count \(count) block \(blockSize), element \(i)")
                    }
                }
            }
        }
    }

    // MARK: - Top-k blocks

    func testVectorPackedTopKBlockIsByteIdenticalToTheScalarPacker() throws {
        let count = 4000
        let source = values(count: count).map { $0.isFinite ? $0 : 0 }
        let indices = TopK.selectIndices(magnitudes: source.map { abs($0) }, k: 40)

        for dataType in [DataType.float32, .float16, .bfloat16] {
            let size = TopKBlock.byteCount(nonZeros: indices.count, valueWidth: dataType.byteWidth)
            let fast = ScratchBuffer(capacity: size)
            let slow = ScratchBuffer(capacity: size)
            let written = TopKBlock.write(indices: indices, values: source,
                                          dataType: dataType, into: fast.base)
            let reference = Reference.writeTopKBlock(indices: indices, values: source,
                                                     dataType: dataType, into: slow.base)
            XCTAssertEqual(written, reference)
            XCTAssertEqual(bytes(fast.base, written), bytes(slow.base, reference),
                           "\(dataType) top-k block differs from the scalar packer")
        }
    }

    /// A block packed by one build must accumulate identically on the other,
    /// which is the property that lets a mixed-version world run top-k at all.
    func testScalarPackedTopKBlockAccumulatesIdentically() throws {
        let count = 512
        let source = values(count: count).map { $0.isFinite ? $0 : 0 }
        let indices = TopK.selectIndices(magnitudes: source.map { abs($0) }, k: 24)
        let size = TopKBlock.byteCount(nonZeros: indices.count, valueWidth: 4)
        let block = ScratchBuffer(capacity: size)
        _ = Reference.writeTopKBlock(indices: indices, values: source, dataType: .float32,
                                     into: block.base)

        let destination = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        defer { destination.deallocate() }
        destination.initializeMemory(as: UInt8.self, repeating: 0)
        try TopKBlock.accumulate(payload: block.base, payloadBytes: size, elementCount: count,
                                 dataType: .float32, into: destination.baseAddress!)

        for i in 0..<count {
            let expected = indices.contains(i) ? source[i] : 0
            let actual = destination.baseAddress!.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
            XCTAssertEqual(actual, expected, "element \(i) of a scalar-packed block")
        }
    }

    // MARK: - Selection

    /// The rewritten selection partitions magnitudes instead of an index
    /// permutation, so the tie handling is new code. At scale, with ties made
    /// deliberately common, it still has to return exactly `k` ascending
    /// indices whose magnitudes dominate every rejected one.
    func testSelectionHoldsAtScaleWithHeavyTies() {
        var state: UInt64 = 0xDEAD_BEEF_CAFE_F00D
        for trial in 0..<6 {
            let n = 100_000
            var magnitudes = [Float](repeating: 0, count: n)
            for i in 0..<n {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                // Only 64 distinct values, so the k-th largest is almost
                // always a tie shared by thousands of elements.
                magnitudes[i] = Float((state >> 48) % 64)
            }
            let k = [1, 7, 1000, 40_000, n - 1][trial % 5]
            let selected = TopK.selectIndices(magnitudes: magnitudes, k: k)

            XCTAssertEqual(selected.count, k)
            XCTAssertEqual(selected, selected.sorted())
            XCTAssertEqual(Set(selected).count, k, "selection must not repeat an index")
            let chosen = Set(selected)
            let smallestSelected = selected.map { magnitudes[$0] }.min() ?? 0
            var largestRejected = -Float.infinity
            for i in 0..<n where !chosen.contains(i) {
                largestRejected = max(largestRejected, magnitudes[i])
            }
            XCTAssertGreaterThanOrEqual(smallestSelected, largestRejected)
        }
    }

    func testCountAboveMatchesANaiveCount() {
        for count in tailLengths {
            let sample = values(count: count).map { $0.isFinite ? abs($0) : 0 }
            for threshold in [Float(0), 0.25, 1, 8] {
                let expected = sample.filter { $0 > threshold }.count
                let actual = sample.withUnsafeBufferPointer {
                    CodecKernels.countAboveFP32($0.baseAddress!, count: count, threshold: threshold)
                }
                XCTAssertEqual(actual, expected, "count \(count), threshold \(threshold)")
            }
        }
    }

    // MARK: - The probe measures the collective's own kernels

    /// `mcclbench --codec-bench` reports ceilings through `CodecKernelProbe`.
    /// If that seam ever stopped running the same code as a ring step, every
    /// number in docs/ARCHITECTURE.md §Measured would be measuring nothing.
    func testCodecProbeEncodesExactlyWhatARingStepWould() throws {
        let count = 1024
        let source = buffer(values(count: count))
        defer { source.deallocate() }

        for compression in [WireCompression.none, .downcast, .int8Blockwise(blockSize: 128)] {
            let codec = try WireCodec(dataType: .float32, compression: compression)
            let direct = ScratchBuffer(capacity: codec.encodedByteCount(elementCount: count))
            let throughProbe = ScratchBuffer(capacity: try CodecKernelProbe.encodedByteCount(
                compression, dataType: .float32, elementCount: count))

            let a = codec.encode(from: source.baseAddress!, elementCount: count, into: direct.base)
            let b = try CodecKernelProbe.encode(compression, dataType: .float32,
                                                from: source.baseAddress!, elementCount: count,
                                                into: throughProbe.base)
            XCTAssertEqual(a, b)
            XCTAssertEqual(bytes(direct.base, a), bytes(throughProbe.base, b),
                           "\(compression) through the probe differs from the collective's encoder")
        }
    }

    func testTopKProbeCarriesItsResidualLikeACommunicator() throws {
        let count = 256
        let source = buffer((0..<count).map { Float($0 % 17) - 8 })
        defer { source.deallocate() }
        let probe = TopKKernelProbe(elementCount: count, dataType: .float32, fraction: 0.05)
        let block = ScratchBuffer(capacity: probe.blockByteCount * 2)

        let first = probe.encode(from: source.baseAddress!, into: block.base)
        XCTAssertEqual(first, probe.blockByteCount)
        XCTAssertEqual(try TopKBlock.nonZeroCount(payload: block.base, payloadBytes: first),
                       TopK.count(elementCount: count, fraction: 0.05))

        // The second call must see the first call's leftovers: with a constant
        // input the residual grows, so the values on the wire grow too.
        let firstValues = topKValues(block.base, nonZeros: probe.nonZeros)
        _ = probe.encode(from: source.baseAddress!, into: block.base)
        let secondValues = topKValues(block.base, nonZeros: probe.nonZeros)
        XCTAssertGreaterThan(secondValues.map { abs($0) }.max() ?? 0,
                             firstValues.map { abs($0) }.max() ?? 0,
                             "the probe dropped its residual between calls")
    }

    private func topKValues(_ payload: UnsafeRawPointer, nonZeros: Int) -> [Float] {
        let base = TopKBlock.headerBytes + nonZeros * MemoryLayout<UInt32>.size
        return (0..<nonZeros).map {
            payload.loadUnaligned(fromByteOffset: base + $0 * 4, as: Float.self)
        }
    }
}
