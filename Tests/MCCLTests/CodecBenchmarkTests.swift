import XCTest
@testable import MCCL
@testable import MCCLBenchmarks

/// `mcclbench --codec-bench`. The ceilings it prints are half of the rule in
/// docs/ARCHITECTURE.md §Measured — the half that decides whether compression is
/// worth switching on — so what is tested here is the bookkeeping around the
/// timings: that every point is measured, that the ratios are the wire formats'
/// real ratios, and that the derived figures are consistent with the times they
/// come from. The timings themselves are the machine's business.
final class CodecBenchmarkTests: XCTestCase {

    private func tinyOptions() -> BenchOptions {
        var options = BenchOptions()
        options.worldSize = 2
        options.codecBench = true
        options.explicitSizes = [4096, 16384]
        options.codecs = [.none, .downcast, .int8Blockwise(blockSize: 256), .topK(fraction: 0.01)]
        options.warmup = 1
        options.maxIterations = 4
        options.byteBudget = 1 << 16
        return options
    }

    func testEveryPointIsMeasuredAndProducesUsableNumbers() {
        let options = tinyOptions()
        var streamed = 0
        let rows = CodecBenchRunner.run(options, repeats: 1) { _ in streamed += 1 }

        XCTAssertEqual(rows.count, options.sizes.count * options.codecs.count)
        XCTAssertEqual(streamed, rows.count, "every row must reach the caller as it is produced")
        for row in rows {
            XCTAssertNil(row.failure, "\(row.codec) at \(row.payloadBytes) failed: \(row.failure ?? "")")
            XCTAssertGreaterThan(row.encodeSeconds, 0)
            XCTAssertGreaterThan(row.decodeSeconds, 0)
            XCTAssertGreaterThan(row.encodeBytesPerSecond, 0)
            XCTAssertGreaterThan(row.decodeBytesPerSecond, 0)
            XCTAssertEqual(row.elementCount, row.payloadBytes / 4)
        }
    }

    /// The round-trip figure is what the compression rule is stated against, so
    /// it has to be the two halves in series and not, say, their mean.
    func testRoundTripIsTheHarmonicOfTheTwoHalves() {
        for row in CodecBenchRunner.run(tinyOptions(), repeats: 1) {
            let expected = Double(row.payloadBytes) / (row.encodeSeconds + row.decodeSeconds)
            XCTAssertEqual(row.roundTripBytesPerSecond, expected, accuracy: expected * 1e-9)
            XCTAssertLessThanOrEqual(row.roundTripBytesPerSecond,
                                     min(row.encodeBytesPerSecond, row.decodeBytesPerSecond) + 1,
                                     "\(row.codec): the pair cannot beat its slower half")
            XCTAssertEqual(row.nanosecondsPerElement,
                           (row.encodeSeconds + row.decodeSeconds) / Double(row.elementCount) * 1e9,
                           accuracy: 1e-6)
        }
    }

    /// The ratios are the wire formats' own, and the whole point of reporting
    /// them next to the rates: past the crossover it is the ratio that decides.
    func testReportedRatiosAreTheWireFormatsRatios() {
        let rows = CodecBenchRunner.run(tinyOptions(), repeats: 1)
        func ratio(_ codec: String, at bytes: Int) -> Double {
            rows.first { $0.codec == codec && $0.payloadBytes == bytes }?.compressionRatio ?? 0
        }
        XCTAssertEqual(ratio("none", at: 16384), 1.0, accuracy: 1e-9)
        XCTAssertEqual(ratio("downcast", at: 16384), 2.0, accuracy: 1e-9)
        // 4096 elements at block 256: 16 scales + 4096 codes.
        XCTAssertEqual(ratio("int8/256", at: 16384), 16384.0 / (16 * 4 + 4096), accuracy: 1e-9)
        // top-k sends (index, value) pairs for 1% of the elements, plus a header,
        // and its "wire" is one block per peer.
        XCTAssertGreaterThan(ratio("topk/0.01", at: 16384), 40)
    }

    func testTopKHarnessChargesForOneBlockPerPeer() throws {
        var options = tinyOptions()
        options.worldSize = 4
        options.explicitSizes = [16384]
        options.codecs = [.topK(fraction: 0.01)]
        let rows = CodecBenchRunner.run(options, repeats: 1)
        let probe = TopKKernelProbe(elementCount: 4096, dataType: .float32, fraction: 0.01)
        XCTAssertEqual(rows.first?.encodedBytes, probe.blockByteCount * 3,
                       "a rank receives a block from every peer but itself")
    }

    func testCodecsThatCannotHelpADataTypeAreSkipped() {
        var options = tinyOptions()
        options.dataType = .int32
        let rows = CodecBenchRunner.run(options, repeats: 1)
        XCTAssertFalse(rows.contains { $0.codec != "none" },
                       "int32 has nothing to downcast, quantise or sparsify")
        XCTAssertEqual(rows.count, options.sizes.count)
    }

    func testCeilingsTakeEachCodecsBestRoundTripPoint() {
        let rows = CodecBenchRunner.run(tinyOptions(), repeats: 1)
        let ceilings = CodecBenchTable.ceilings(rows)
        XCTAssertEqual(ceilings.map(\.codec), ["none", "downcast", "int8/256", "topk/0.01"],
                       "codecs keep the order they were swept in")
        for entry in ceilings {
            let best = rows.filter { $0.codec == entry.codec }.map(\.roundTripBytesPerSecond).max()
            XCTAssertEqual(entry.ceiling, best)
            XCTAssertTrue(rows.contains { $0.codec == entry.codec && $0.payloadBytes == entry.at })
        }
    }

    func testTableRendersAlignedColumnsAndCSV() {
        var options = tinyOptions()
        let row = CodecBenchRow(payloadBytes: 1 << 20, elementCount: 1 << 18, codec: "downcast",
                                encodedBytes: 1 << 19, iterations: 7,
                                encodeSeconds: 0.001, decodeSeconds: 0.003, failure: nil)

        let text = CodecBenchTable.render(row, options: options)
        XCTAssertTrue(text.contains("1 MiB"))
        XCTAssertTrue(text.contains("2.00x"), "the ratio column: \(text)")
        XCTAssertTrue(text.contains("1.049"), "1 MiB / 1 ms is 1.049 GB/s: \(text)")
        XCTAssertEqual(CodecBenchTable.rule(options)?.isEmpty, false)

        options.csv = true
        let fields = CodecBenchTable.render(row, options: options).split(separator: ",")
        XCTAssertEqual(fields.count, CodecBenchTable.header(options).split(separator: ",").count)
        XCTAssertEqual(String(fields[0]), "1048576")
        XCTAssertEqual(String(fields[1]), "downcast")
        XCTAssertEqual(String(fields[3]), "2.000")
        XCTAssertNil(CodecBenchTable.rule(options))
    }

    func testAFailedPointIsReportedRatherThanDropped() {
        let row = CodecBenchRow(payloadBytes: 4096, elementCount: 1024, codec: "downcast",
                                encodedBytes: 0, iterations: 0, encodeSeconds: 0,
                                decodeSeconds: 0, failure: "boom")
        var options = tinyOptions()
        XCTAssertTrue(CodecBenchTable.render(row, options: options).contains("boom"))
        options.csv = true
        XCTAssertTrue(CodecBenchTable.render(row, options: options).contains("\"boom\""))
        XCTAssertEqual(row.roundTripBytesPerSecond, 0)
        XCTAssertEqual(row.compressionRatio, 0)
        XCTAssertEqual(CodecBenchTable.ceilings([row]).count, 0, "a failed point has no ceiling")
    }

    func testTheFlagParsesAndRefusesToJoinAWorld() throws {
        let options = try XCTUnwrap(try BenchArguments.parse(["--codec-bench", "--sizes", "64K"]))
        XCTAssertTrue(options.codecBench)
        XCTAssertEqual(options.sizes, [65536])

        XCTAssertThrowsError(try BenchArguments.parse(["--codec-bench", "--rank", "1"])) { error in
            XCTAssertTrue("\(error)".contains("no world to join"), "\(error)")
        }
        XCTAssertFalse(try XCTUnwrap(try BenchArguments.parse([])).codecBench,
                       "the collective sweep stays the default")
    }

    /// The generated buffer has to give every codec something real to do: a
    /// constant buffer would quantise perfectly and leave top-k with no choice.
    func testTheGeneratedBufferHasBlockStructureAndSpread() {
        let count = 8192
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        defer { buffer.deallocate() }
        CodecBenchRunner.fill(buffer, count: count, dataType: .float32)
        let values = (0..<count).map {
            buffer.baseAddress!.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self)
        }
        XCTAssertTrue(values.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(Set(values).count, count / 2, "the values must not repeat")
        let firstBlock = values[0..<4096].map { abs($0) }.max() ?? 0
        let secondBlock = values[4096..<8192].map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(secondBlock / max(firstBlock, .leastNormalMagnitude), 1.5,
                             "blocks must differ in scale, or per-block quantisation proves nothing")
    }
}
