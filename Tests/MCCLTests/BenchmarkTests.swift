import XCTest
@testable import MCCL
@testable import MCCLBenchmarks

/// The `mcclbench` harness. A benchmark whose numbers are wrong is worse than
/// no benchmark, so what is tested here is the arithmetic and the sweep
/// bookkeeping — not the timings, which are the machine's business.
final class BenchmarkTests: XCTestCase {

    private func tinyOptions() -> BenchOptions {
        var options = BenchOptions()
        options.worldSize = 4
        options.minBytes = 1 << 10
        options.maxBytes = 1 << 12
        options.sizeStep = 4
        options.warmup = 1
        options.minIterations = 1
        options.maxIterations = 2
        options.byteBudget = 1 << 12
        options.transport = .loopback
        return options
    }

    // MARK: - Sweep

    func testSweepCoversEveryPointAndProducesUsableNumbers() async throws {
        let options = tinyOptions()
        let rows = try await BenchRunner.run(options)

        let expected = options.sizes.count * options.algorithms.count * options.codecs.count
        XCTAssertEqual(rows.count, expected,
                       "3 sizes x \(options.algorithms.count) algorithms x \(options.codecs.count) codecs")
        for row in rows {
            XCTAssertNil(row.failure, "\(row.algorithm) / \(row.codec) at \(row.messageBytes)B: "
                         + "\(row.failure ?? "")")
            XCTAssertGreaterThan(row.seconds, 0)
            XCTAssertGreaterThan(row.algorithmicBytesPerSecond, 0)
            XCTAssertGreaterThan(row.iterations, 0)
            // Bus bandwidth for a 4-rank all-reduce is 1.5x the algorithmic
            // figure: each link carries 2(n-1)/n of the payload.
            XCTAssertEqual(row.busBytesPerSecond, row.algorithmicBytesPerSecond * 1.5,
                           accuracy: row.algorithmicBytesPerSecond * 1e-9)
        }
    }

    func testEveryCollectiveRunsUnderEveryAlgorithm() async throws {
        for collective in BenchCollective.allCases {
            var options = tinyOptions()
            options.collective = collective
            options.maxBytes = 1 << 10
            options.codecs = [.none]
            let rows = try await BenchRunner.run(options)
            XCTAssertEqual(rows.count, options.algorithms.count)
            for row in rows {
                XCTAssertNil(row.failure, "\(collective) on \(row.algorithm): \(row.failure ?? "")")
            }
        }
    }

    func testTopKIsSweptOnlyWhereItIsDefined() async throws {
        var options = tinyOptions()
        options.maxBytes = 1 << 10
        options.algorithms = [.ring]
        options.codecs = [.topK(fraction: 0.1)]

        let allReduce = try await BenchRunner.run(options)
        XCTAssertEqual(allReduce.count, 1)
        XCTAssertNil(allReduce[0].failure)
        XCTAssertNotNil(allReduce[0].wireBytesPerRank,
                        "the top-k row should report what it actually put on the wire")

        // Top-k does not apply to the other collectives, so those points are
        // skipped rather than run and reported as errors.
        options.collective = .broadcast
        let broadcastRows = try await BenchRunner.run(options)
        XCTAssertTrue(broadcastRows.isEmpty)

        // Nor to integer dtypes.
        options.collective = .allReduce
        options.dataType = .int32
        let integerRows = try await BenchRunner.run(options)
        XCTAssertTrue(integerRows.isEmpty)
    }

    func testDowncastIsSkippedForDataTypesWithNothingToGiveUp() async throws {
        var options = tinyOptions()
        options.maxBytes = 1 << 10
        options.algorithms = [.ring]
        options.codecs = [.none, .downcast]
        options.dataType = .float16

        let rows = try await BenchRunner.run(options)
        XCTAssertEqual(rows.map(\.codec), ["none"], "fp16 is already downcast")
    }

    func testAWorldOfOneIsRejected() async {
        var options = tinyOptions()
        options.worldSize = 1
        await XCTAssertThrowsMCCLError({ _ = try await BenchRunner.run(options) })
    }

    func testHierarchicalIsSkippedWhenTheWorldCannotFormIslands() async throws {
        var options = tinyOptions()
        options.worldSize = 2
        options.maxBytes = 1 << 10
        options.codecs = [.none]
        let rows = try await BenchRunner.run(options)
        XCTAssertEqual(Set(rows.map(\.algorithm.rawValue)), ["ring", "tree"],
                       "two ranks cannot make two islands of two")
    }

    // MARK: - Dimensions

    func testSizesAreAGeometricSweep() {
        var options = BenchOptions()
        options.minBytes = 1024
        options.maxBytes = 64 << 20
        options.sizeStep = 4
        XCTAssertEqual(options.sizes.first, 1024)
        XCTAssertEqual(options.sizes.last, 64 << 20)
        XCTAssertEqual(options.sizes.count, 9, "1 KiB to 64 MiB in steps of 4x")

        options.maxBytes = 1024
        XCTAssertEqual(options.sizes, [1024])
        options.maxBytes = 512
        XCTAssertEqual(options.sizes, [1024], "never sweep nothing at all")
    }

    func testIterationCountTracksTheByteBudget() {
        var options = BenchOptions()
        options.byteBudget = 1 << 20
        options.minIterations = 3
        options.maxIterations = 50
        XCTAssertEqual(options.iterations(forMessageBytes: 1 << 10), 50, "clamped by the ceiling")
        XCTAssertEqual(options.iterations(forMessageBytes: 1 << 16), 16)
        XCTAssertEqual(options.iterations(forMessageBytes: 1 << 26), 3, "clamped by the floor")
    }

    func testPlansHaveTheShapeTheirNamePromises() {
        XCTAssertEqual(BenchAlgorithm.ring.plan(worldSize: 4), .ring(order: [0, 1, 2, 3]))
        guard case .tree(let root, let children)? = BenchAlgorithm.tree.plan(worldSize: 4) else {
            return XCTFail("tree plan missing")
        }
        XCTAssertEqual(root, 0)
        XCTAssertEqual(TopologyPlanner.parents(of: children).count, 3, "every non-root has a parent")

        guard case .hierarchical(let islands, _)? = BenchAlgorithm.hierarchical.plan(worldSize: 4) else {
            return XCTFail("hierarchical plan missing")
        }
        XCTAssertEqual(islands, [[0, 1], [2, 3]])
        XCTAssertNil(BenchAlgorithm.hierarchical.plan(worldSize: 3))
        XCTAssertNil(BenchAlgorithm.hierarchical.plan(worldSize: 2))
    }

    func testBusFactorsMatchTheAlgorithmsTraffic() {
        XCTAssertEqual(BenchCollective.allReduce.busFactor(worldSize: 4), 1.5, accuracy: 1e-12)
        XCTAssertEqual(BenchCollective.allGather.busFactor(worldSize: 4), 0.75, accuracy: 1e-12)
        XCTAssertEqual(BenchCollective.reduceScatter.busFactor(worldSize: 4), 0.75, accuracy: 1e-12)
        XCTAssertEqual(BenchCollective.broadcast.busFactor(worldSize: 4), 1.0, accuracy: 1e-12)
        XCTAssertEqual(BenchCollective.allReduce.busFactor(worldSize: 1), 1.0, accuracy: 1e-12)
    }

    func testGatherAndScatterSplitTheMessageAcrossRanks() {
        XCTAssertEqual(BenchCollective.allReduce.elementCount(
            messageBytes: 4096, dataType: .float32, worldSize: 4), 1024)
        XCTAssertEqual(BenchCollective.allGather.elementCount(
            messageBytes: 4096, dataType: .float32, worldSize: 4), 256,
            "the gathered buffer, not the contribution, is the message size")
        XCTAssertEqual(BenchCollective.allReduce.elementCount(
            messageBytes: 4096, dataType: .float16, worldSize: 4), 2048)
    }

    // MARK: - Arguments

    func testDefaultsCoverTheDocumentedSweep() throws {
        let options = try XCTUnwrap(try BenchArguments.parse([]))
        XCTAssertEqual(options.worldSize, 4)
        XCTAssertEqual(options.minBytes, 1 << 10)
        XCTAssertEqual(options.maxBytes, 64 << 20)
        XCTAssertEqual(options.algorithms, BenchAlgorithm.allCases)
        XCTAssertEqual(options.codecs.count, 4)
        XCTAssertEqual(options.collective, .allReduce)
    }

    func testHelpReturnsNothingToRun() throws {
        XCTAssertNil(try BenchArguments.parse(["--help"]))
        XCTAssertNil(try BenchArguments.parse(["-h"]))
    }

    func testFlagsParse() throws {
        let options = try XCTUnwrap(try BenchArguments.parse([
            "--ranks", "8", "--min-bytes", "512", "--max-bytes", "8192", "--step", "2",
            "--collective", "reducescatter", "--algorithms", "ring,tree",
            "--codecs", "none,int8:128,topk:0.05", "--dtype", "bf16", "--op", "avg",
            "--transport", "loopback", "--warmup", "1", "--min-iters", "2",
            "--max-iters", "9", "--budget", "4096", "--csv",
        ]))
        XCTAssertEqual(options.worldSize, 8)
        XCTAssertEqual(options.sizes, [512, 1024, 2048, 4096, 8192])
        XCTAssertEqual(options.collective, .reduceScatter)
        XCTAssertEqual(options.algorithms, [.ring, .tree])
        XCTAssertEqual(options.codecs.map(\.description), ["none", "int8/128", "topk/0.05"])
        XCTAssertEqual(options.dataType, .bfloat16)
        XCTAssertEqual(options.op, .avg)
        XCTAssertEqual(options.transport, .loopback)
        XCTAssertEqual(options.byteBudget, 4096)
        XCTAssertTrue(options.csv)
    }

    func testQuickTrimsTheSweep() throws {
        let options = try XCTUnwrap(try BenchArguments.parse(["--quick"]))
        XCTAssertEqual(options.maxBytes, 1 << 20)
        XCTAssertEqual(options.algorithms, [.ring, .tree])
    }

    func testCodecShorthandDefaultsAndBounds() throws {
        XCTAssertEqual(try BenchArguments.parseCodecs("int8").map(\.description), ["int8/256"])
        XCTAssertEqual(try BenchArguments.parseCodecs("topk").map(\.description), ["topk/0.01"])
        XCTAssertEqual(try BenchArguments.parseCodecs("fp16").map(\.description), ["downcast"])
        XCTAssertThrowsError(try BenchArguments.parseCodecs("topk:0"))
        XCTAssertThrowsError(try BenchArguments.parseCodecs("topk:2"))
        XCTAssertThrowsError(try BenchArguments.parseCodecs("int8:0"))
        XCTAssertThrowsError(try BenchArguments.parseCodecs("lz4"))
    }

    func testBadArgumentsAreRejectedWithAReason() {
        for arguments in [["--ranks"], ["--ranks", "one"], ["--ranks", "1"],
                          ["--collective", "scatter"], ["--dtype", "fp8"],
                          ["--op", "median"], ["--transport", "rdma"],
                          ["--algorithms", "mesh"], ["--min-bytes", "0"],
                          ["--min-bytes", "4096", "--max-bytes", "1024"],
                          ["--nonsense"]] {
            XCTAssertThrowsError(try BenchArguments.parse(arguments),
                                 "\(arguments) should not parse") { error in
                XCTAssertFalse("\(error)".isEmpty, "an error needs to say what went wrong")
            }
        }
    }

    // MARK: - Table

    func testTableRendersAlignedColumnsAndCSV() {
        var options = BenchOptions()
        let row = BenchRow(
            messageBytes: 1 << 20, algorithm: .ring, codec: "int8/256", iterations: 12,
            seconds: 0.0025, algorithmicBytesPerSecond: 4.19e8, busBytesPerSecond: 6.29e8,
            wireBytesPerRank: nil, failure: nil)

        let text = BenchTable.render(row, options: options)
        XCTAssertTrue(text.contains("1 MiB"))
        XCTAssertTrue(text.contains("ring"))
        XCTAssertTrue(text.contains("int8/256"))
        XCTAssertTrue(text.contains("2.500 ms"))
        XCTAssertEqual(text.count, BenchTable.header(options).count,
                       "rows must line up with the header")

        options.csv = true
        let csv = BenchTable.render(row, options: options)
        XCTAssertEqual(csv.split(separator: ",", omittingEmptySubsequences: false).count,
                       BenchTable.header(options).split(separator: ",").count)
        XCTAssertTrue(csv.hasPrefix("1048576,ring,int8/256,12,"))
    }

    func testFailedPointsAreReportedRatherThanDropped() {
        var options = BenchOptions()
        let row = BenchRow(
            messageBytes: 1024, algorithm: .tree, codec: "topk/0.01", iterations: 0,
            seconds: 0, algorithmicBytesPerSecond: 0, busBytesPerSecond: 0,
            wireBytesPerRank: nil, failure: "mccl: unsupported wire compression: nope")
        XCTAssertTrue(BenchTable.render(row, options: options).contains("unsupported"))

        options.csv = true
        let csv = BenchTable.render(row, options: options)
        XCTAssertTrue(csv.contains("\"mccl: unsupported wire compression: nope\""),
                      "a comma-bearing message must be quoted: \(csv)")
    }

    func testByteAndSecondFormatting() {
        XCTAssertEqual(BenchTable.bytes(512), "512 B")
        XCTAssertEqual(BenchTable.bytes(1024), "1 KiB")
        XCTAssertEqual(BenchTable.bytes(1536), "1.5 KiB")
        XCTAssertEqual(BenchTable.bytes(64 << 20), "64 MiB")
        XCTAssertEqual(BenchTable.seconds(0), "—")
        XCTAssertEqual(BenchTable.seconds(1.5e-6), "1.5 µs")
        XCTAssertEqual(BenchTable.seconds(2.5e-3), "2.500 ms")
        XCTAssertEqual(BenchTable.seconds(1.25), "1.250 s")
    }
    // MARK: - Inapplicable algorithms

    /// A world with no hierarchy produces no hierarchical rows, and that has to
    /// be *said*. An n=3 sweep asked for `hierarchical` and silently got a table
    /// without it, which reads as a broken harness rather than as a property of
    /// a uniform three-rank fabric.
    func testHierarchicalReportsWhyItIsInapplicable() {
        for worldSize in [1, 2, 3] {
            XCTAssertNil(BenchAlgorithm.hierarchical.plan(worldSize: worldSize))
            let reason = BenchAlgorithm.hierarchical.inapplicabilityReason(worldSize: worldSize)
            XCTAssertNotNil(reason, "world of \(worldSize)")
            XCTAssertTrue(reason?.contains("islands") ?? false, reason ?? "nil")
            XCTAssertTrue(reason?.contains("\(worldSize)") ?? false, reason ?? "nil")
        }
    }

    func testApplicableAlgorithmsGiveNoReason() {
        for worldSize in [2, 3, 4, 8] {
            XCTAssertNil(BenchAlgorithm.ring.inapplicabilityReason(worldSize: worldSize))
            XCTAssertNil(BenchAlgorithm.tree.inapplicabilityReason(worldSize: worldSize))
        }
        XCTAssertNil(BenchAlgorithm.hierarchical.inapplicabilityReason(worldSize: 4))
    }

}
