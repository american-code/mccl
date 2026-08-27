import MCCL
import MLX
import XCTest

@testable import MCCLMLX

/// The adapter's collectives, run over a real two-rank loopback world inside
/// one process — two `Communicator`s joined by `LoopbackTransport`, each driven
/// from its own thread.
///
/// The values are deliberately asymmetric between the ranks. A test where both
/// ranks contribute the same array cannot distinguish a working all-reduce from
/// a no-op that returns its input.
final class MLXCollectiveTests: XCTestCase {

    /// rank r contributes [r+1, 2(r+1), 3(r+1), ...], so a 2-rank sum is 3x the
    /// base vector and no rank's own contribution is the answer.
    ///
    /// The base vector wraps at `ceiling` so the *sum* stays inside the dtype.
    /// int8 tops out at 127 and mccl's integer reductions wrap rather than
    /// saturate, so a test that let the sum overflow would be measuring
    /// two's-complement arithmetic and calling it a collective failure.
    private func base(count: Int, dtype: DType) -> [Float] {
        let ceiling = dtype == .int8 ? 20 : count
        return (0..<count).map { Float($0 % ceiling + 1) }
    }

    private func contribution(rank: Int, count: Int, dtype: DType) -> MLXArray {
        MLXArray(base(count: count, dtype: dtype).map { $0 * Float(rank + 1) }, [count])
            .asType(dtype)
    }

    /// What a 2-rank sum of `contribution` must produce.
    private func worldSum(count: Int, dtype: DType) -> [Float] {
        base(count: count, dtype: dtype).map { $0 * 3 }
    }

    // MARK: - All-reduce

    func testAllReduceSumOverEverySharedDataType() throws {
        try MLXRuntime.requireMetallib()
        for dtype in MLXDataType.supported {
            let count = 64
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: count, dtype: dtype)
                let reduced = try comm.allReduce(mine, op: .sum)
                XCTAssertEqual(reduced.dtype, dtype)
                XCTAssertEqual(reduced.shape, [count])
                results.set(comm.rank, reduced.floats)
                // The input is not modified — the whole reason for the copy.
                XCTAssertEqual(mine.floats, self.contribution(rank: comm.rank, count: count,
                                                              dtype: dtype).floats)
            }
            let expected = worldSum(count: count, dtype: dtype)
            for rank in 0..<2 {
                XCTAssertEqual(results.get(rank) ?? [], expected,
                               "\(dtype) rank \(rank) did not get the world sum")
            }
        }
    }

    func testAllReduceAverage() throws {
        try MLXRuntime.requireMetallib()
        let count = 32
        let results = RankResults<[Float]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = self.contribution(rank: comm.rank, count: count, dtype: .float32)
            results.set(comm.rank, try comm.allReduce(mine, op: .avg).floats)
        }
        let expected = (0..<count).map { Float($0 + 1) * 1.5 }
        for rank in 0..<2 { XCTAssertEqual(results.get(rank) ?? [], expected) }
    }

    func testAllReduceMinAndMax() throws {
        try MLXRuntime.requireMetallib()
        for (op, factor) in [(ReduceOp.min, Float(1)), (.max, Float(2))] {
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: 16, dtype: .float32)
                results.set(comm.rank, try comm.allReduce(mine, op: op).floats)
            }
            let expected = (0..<16).map { Float($0 + 1) * factor }
            for rank in 0..<2 { XCTAssertEqual(results.get(rank) ?? [], expected, "\(op)") }
        }
    }

    /// Shape is preserved through the flattening the collective does internally.
    func testAllReducePreservesMultiDimensionalShape() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<([Int], [Float])>()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = MLXArray(
                (0..<24).map { Float(($0 + 1) * (comm.rank + 1)) }, [2, 3, 4])
            let reduced = try comm.allReduce(mine, op: .sum)
            results.set(comm.rank, (reduced.shape, reduced.floats))
        }
        for rank in 0..<2 {
            let (shape, values) = results.get(rank) ?? ([], [])
            XCTAssertEqual(shape, [2, 3, 4])
            XCTAssertEqual(values, (0..<24).map { Float(($0 + 1) * 3) })
        }
    }

    /// A strided input goes through the extra materialisation and still lands
    /// on the right answer.
    func testAllReduceAcceptsAStridedInput() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<[Float]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let base = MLXArray(
                (0..<32).map { Float(($0 + 1) * (comm.rank + 1)) }, [16, 2])
            let column = base[.ellipsis, 0]
            results.set(comm.rank, try comm.allReduce(column, op: .sum).floats)
        }
        // Column 0 is elements 1, 3, 5, ... of the base vector.
        let expected = (0..<16).map { Float(($0 * 2 + 1) * 3) }
        for rank in 0..<2 { XCTAssertEqual(results.get(rank) ?? [], expected) }
    }

    /// Element counts either side of `MLXWorkingBuffer.adoptionFloor`, where the
    /// adapter switches between reducing in the result array's own storage and
    /// reducing in a scratch buffer.
    ///
    /// Both paths are correct, so this is not testing a behaviour difference —
    /// it is testing that there is *no* behaviour difference, which is the claim
    /// that lets the threshold be an implementation detail. A one-element
    /// all-reduce is not academic: it is how the demo reports its loss.
    func testAllReduceAcrossTheAdoptionThreshold() throws {
        try MLXRuntime.requireMetallib()
        // 4 bytes .. 132 bytes in fp32, stepping across the 64-byte floor.
        for count in [1, 2, 3, 4, 8, 15, 16, 17, 32, 33] {
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: count, dtype: .float32)
                results.set(comm.rank, try comm.allReduce(mine, op: .sum).floats)
            }
            let expected = worldSum(count: count, dtype: .float32)
            for rank in 0..<2 {
                XCTAssertEqual(results.get(rank) ?? [], expected,
                               "count \(count) (\(count * 4) bytes) rank \(rank)")
            }
        }
    }

    /// A degenerate one-rank world is a defined no-op, not a crash — the demo
    /// runs exactly this configuration for its single-process control.
    func testAllReduceOnASingleRankWorldIsIdentity() throws {
        try MLXRuntime.requireMetallib()
        let comm = Communicator(rank: 0, worldSize: 1, topology: .uniform(nodeCount: 1))
        defer { comm.shutdown() }
        let array = MLXArray([1, 2, 3] as [Float], [3])
        XCTAssertEqual(try comm.allReduce(array, op: .sum).floats, [1, 2, 3])
        XCTAssertEqual(try comm.allReduce(array, op: .avg).floats, [1, 2, 3])
    }

    // MARK: - Compression variants

    /// Compression is in-flight only: it changes what crosses the wire, never
    /// the dtype or shape the caller sees, and for the lossless-enough codecs
    /// it lands close enough to the exact answer to be usable for gradients.
    func testAllReduceUnderEveryCompressionScheme() throws {
        try MLXRuntime.requireMetallib()
        let count = 512
        let cases: [(WireCompression, Float, String)] = [
            (.none, 0, "none"),
            // fp32 -> fp16 on the wire: ~3 decimal digits on values up to 1536.
            (.downcast, 1.0, "downcast"),
            // per-block absmax int8: one part in 127 of the block's largest.
            (.int8Blockwise(blockSize: 256), 16.0, "int8/256"),
        ]
        for (compression, tolerance, name) in cases {
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: count, dtype: .float32)
                let reduced = try comm.allReduce(mine, op: .sum, compression: compression)
                XCTAssertEqual(reduced.dtype, .float32, name)
                XCTAssertEqual(reduced.shape, [count], name)
                results.set(comm.rank, reduced.floats)
            }
            let expected = (0..<count).map { Float(($0 + 1) * 3) }
            for rank in 0..<2 {
                let got = results.get(rank) ?? []
                XCTAssertEqual(got.count, expected.count, name)
                for i in 0..<Swift.min(got.count, expected.count) {
                    XCTAssertEqual(got[i], expected[i], accuracy: tolerance,
                                   "\(name) rank \(rank) element \(i)")
                }
            }
        }
    }

    /// Top-k is sparsified with error feedback. Any single call is visibly
    /// lossy and is *supposed* to be: only the `k` largest magnitudes go on the
    /// wire. What makes it usable is that the rest is carried forward rather
    /// than dropped, so it is the **running mean** over a sequence of calls that
    /// converges on the uncompressed answer — the same property
    /// `MCCLTests.TopKTests` pins for the raw collective, checked here through
    /// the MLX surface.
    func testTopKAllReduceRunningMeanConvergesOnTheExactAnswer() throws {
        try MLXRuntime.requireMetallib()
        let count = 256
        let iterations = 60
        let results = RankResults<(early: Float, late: Float)>()
        try MLXRanks.run(worldSize: 2) { comm in
            let expected = self.worldSum(count: count, dtype: .float32)
            var running = [Float](repeating: 0, count: count)
            var early: Float = 0
            for step in 1...iterations {
                let mine = self.contribution(rank: comm.rank, count: count, dtype: .float32)
                let result = try comm.allReduce(
                    mine, op: .sum, compression: .topK(fraction: 0.25)).floats
                for i in 0..<count { running[i] += result[i] }
                if step == 4 {
                    early = self.meanError(running, expected, steps: step)
                }
            }
            results.set(comm.rank,
                        (early, self.meanError(running, expected, steps: iterations)))
        }
        for rank in 0..<2 {
            let (early, late) = results.get(rank) ?? (0, 0)
            XCTAssertGreaterThan(early, 0, "rank \(rank): top-k should start out lossy")
            XCTAssertLessThan(late, early / 4,
                              "rank \(rank): the running mean is not converging "
                              + "(4 steps: \(early), \(iterations) steps: \(late))")
        }
    }

    /// Mean absolute error of the running mean against the exact answer.
    private func meanError(_ running: [Float], _ expected: [Float], steps: Int) -> Float {
        var total: Float = 0
        for i in 0..<expected.count {
            total += abs(running[i] / Float(steps) - expected[i])
        }
        return total / Float(expected.count)
    }

    /// Integer dtypes are accumulators, not gradients — magnitude sparsification
    /// would corrupt them, and mccl says so rather than doing it.
    func testTopKIsRejectedForIntegerDataTypes() throws {
        try MLXRuntime.requireMetallib()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = self.contribution(rank: comm.rank, count: 32, dtype: .int32)
            XCTAssertThrowsError(
                try comm.allReduce(mine, op: .sum, compression: .topK(fraction: 0.5))
            ) { error in
                guard case MCCLError.unsupportedCompression = error else {
                    return XCTFail("expected unsupportedCompression, got \(error)")
                }
            }
        }
    }

    // MARK: - All-gather

    /// The zero-copy collective: shapes stack on a new leading axis and every
    /// rank sees every contribution in rank order.
    func testAllGatherStacksEveryRankInOrder() throws {
        try MLXRuntime.requireMetallib()
        for dtype in MLXDataType.supported {
            let results = RankResults<([Int], [Float])>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: 8, dtype: dtype)
                let gathered = try comm.allGather(mine)
                results.set(comm.rank, (gathered.shape, gathered.floats))
            }
            // Spelled out in steps. As a single `+` of two mapped ranges this
            // hit the type-checker's expression budget on a CI runner and
            // failed to compile at all, which is a silly way to lose a suite.
            let first: [Float] = (0..<8).map { Float($0 + 1) }
            let second: [Float] = (0..<8).map { Float(($0 + 1) * 2) }
            let expected: [Float] = first + second
            for rank in 0..<2 {
                let (shape, values) = results.get(rank) ?? ([], [])
                XCTAssertEqual(shape, [2, 8], "\(dtype)")
                XCTAssertEqual(values, expected, "\(dtype) rank \(rank)")
            }
        }
    }

    func testAllGatherPreservesTrailingShape() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<[Int]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = MLXArray((0..<12).map { Float($0) }, [3, 4])
            results.set(comm.rank, try comm.allGather(mine).shape)
        }
        for rank in 0..<2 { XCTAssertEqual(results.get(rank) ?? [], [2, 3, 4]) }
    }

    // MARK: - Broadcast and reduce

    func testBroadcastOverwritesNonRootRanks() throws {
        try MLXRuntime.requireMetallib()
        for root in 0..<2 {
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: 16, dtype: .float32)
                results.set(comm.rank, try comm.broadcast(mine, root: root).floats)
            }
            let expected = (0..<16).map { Float(($0 + 1) * (root + 1)) }
            for rank in 0..<2 {
                XCTAssertEqual(results.get(rank) ?? [], expected, "root \(root), rank \(rank)")
            }
        }
    }

    func testReduceLandsTheAnswerOnTheRoot() throws {
        try MLXRuntime.requireMetallib()
        for root in 0..<2 {
            let results = RankResults<[Float]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let mine = self.contribution(rank: comm.rank, count: 16, dtype: .float32)
                results.set(comm.rank, try comm.reduce(mine, op: .sum, root: root).floats)
            }
            XCTAssertEqual(results.get(root) ?? [], (0..<16).map { Float(($0 + 1) * 3) },
                           "root \(root) did not receive the sum")
        }
    }

    // MARK: - Reduce-scatter

    func testReduceScatterGivesEachRankItsSlice() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<([Int], [Float])>()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = MLXArray(
                (0..<16).map { Float(($0 + 1) * (comm.rank + 1)) }, [4, 4])
            let scattered = try comm.reduceScatter(mine, op: .sum)
            results.set(comm.rank, (scattered.shape, scattered.floats))
        }
        for rank in 0..<2 {
            let (shape, values) = results.get(rank) ?? ([], [])
            XCTAssertEqual(shape, [2, 4], "rank \(rank)")
            let expected = (0..<8).map { Float(($0 + 1 + rank * 8) * 3) }
            XCTAssertEqual(values, expected, "rank \(rank)")
        }
    }

    func testReduceScatterRejectsAnIndivisibleLeadingDimension() throws {
        try MLXRuntime.requireMetallib()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = MLXArray((0..<15).map { Float($0) }, [5, 3])
            XCTAssertThrowsError(try comm.reduceScatter(mine, op: .sum)) { error in
                guard case MCCLError.invalidArgument(let message) = error else {
                    return XCTFail("expected invalidArgument, got \(error)")
                }
                XCTAssertTrue(message.contains("world size"), message)
            }
        }
    }

    // MARK: - Errors

    /// `int64` rather than `float64`: MLX refuses float64 on the GPU with a
    /// fatal error of its own, before any adapter code runs. int64 is a dtype
    /// MLX is perfectly happy with and mccl simply does not carry, which is the
    /// case this test is about.
    func testUnsupportedDataTypeIsRejectedByTheCollective() throws {
        try MLXRuntime.requireMetallib()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = MLXArray([1, 2, 3] as [Float], [3]).asType(.int64)
            XCTAssertThrowsError(try comm.allReduce(mine, op: .sum)) { error in
                guard case MCCLError.invalidArgument = error else {
                    return XCTFail("expected invalidArgument, got \(error)")
                }
            }
        }
    }
}
