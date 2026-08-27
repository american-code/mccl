import MCCL
import MLX
import MLXNN
import XCTest

@testable import MCCLMLX

/// `averageGradients` on a real `MLXNN` model.
///
/// The property that matters is not "the numbers changed" but the exact one the
/// data-parallel demo rests on: with equal shards, averaging each rank's
/// gradient reproduces the gradient of the loss over the combined batch. If that
/// holds, an N-rank run and a single-process run on N times the data take the
/// same optimisation path.
final class GradientAveragingTests: XCTestCase {

    /// A small MLP with a fixed initialisation, so both ranks start identical.
    /// Deterministic weights rather than a seeded RNG: a test that depends on
    /// MLX's RNG stream is a test that breaks when MLX changes it.
    private func makeModel(inputs: Int = 6, hidden: Int = 8, outputs: Int = 3) -> Sequential {
        let model = Sequential {
            Linear(inputs, hidden)
            ReLU()
            Linear(hidden, outputs)
        }
        var replacements: [(String, MLXArray)] = []
        let existing = model.parameters().flattened().sorted { $0.0 < $1.0 }
        for (index, entry) in existing.enumerated() {
            let (key, value) = entry
            // A cheap deterministic spread in [-0.5, 0.5), keyed on position so
            // the two layers do not get identical weights.
            var flat: [Float] = []
            flat.reserveCapacity(value.size)
            for i in 0..<value.size {
                let scaled: Int = (i * 37 + index * 11) % 101
                flat.append(Float(scaled) / 101.0 - 0.5)
            }
            replacements.append((key, MLXArray(flat, value.shape)))
        }
        model.update(parameters: ModuleParameters.unflattened(replacements))
        eval(model)
        return model
    }

    private func synthetic(count: Int, offset: Int, modulus: Int) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        for i in 0..<count {
            let scaled: Int = ((i + offset) * 13) % modulus
            values.append(Float(scaled) / Float(modulus) - 0.5)
        }
        return values
    }

    private func data(rows: Int, inputs: Int, outputs: Int, offset: Int) -> (MLXArray, MLXArray) {
        let x = MLXArray(synthetic(count: rows * inputs, offset: offset, modulus: 71), [rows, inputs])
        let y = MLXArray(synthetic(count: rows * outputs, offset: offset + 7, modulus: 53),
                         [rows, outputs])
        return (x, y)
    }

    private func lossFunction(_ model: Sequential, _ x: MLXArray, _ y: MLXArray) -> MLXArray {
        mseLoss(predictions: model(x), targets: y, reduction: .mean)
    }

    // MARK: - The equivalence the demo depends on

    /// Averaged shard gradients == full-batch gradient, to fp32 tolerance.
    ///
    /// This is the adapter's correctness proof in miniature: the demo checks the
    /// same identity end to end over a loss trajectory, and this checks it on
    /// one step where the expected value can be computed directly.
    func testAveragedShardGradientsEqualTheFullBatchGradient() throws {
        try MLXRuntime.requireMetallib()
        let inputs = 6, outputs = 3, perRank = 16

        // The reference: one process, both shards concatenated.
        let (x0, y0) = data(rows: perRank, inputs: inputs, outputs: outputs, offset: 0)
        let (x1, y1) = data(rows: perRank, inputs: inputs, outputs: outputs, offset: 500)
        let reference = makeModel(inputs: inputs, hidden: 8, outputs: outputs)
        let referenceGrads = valueAndGrad(model: reference, lossFunction)(
            reference,
            MLX.concatenated([x0, x1], axis: 0),
            MLX.concatenated([y0, y1], axis: 0)
        ).1
        eval(referenceGrads)
        let expected = Dictionary(
            uniqueKeysWithValues: referenceGrads.flattened().map { ($0.0, $0.1.floats) })

        let results = RankResults<[String: [Float]]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let model = self.makeModel(inputs: inputs, hidden: 8, outputs: outputs)
            let (x, y) = comm.rank == 0 ? (x0, y0) : (x1, y1)
            let (_, grads) = valueAndGrad(model: model, self.lossFunction)(model, x, y)
            eval(grads)
            let averaged = try averageGradients(grads, comm: comm)
            eval(averaged)
            results.set(
                comm.rank,
                Dictionary(uniqueKeysWithValues: averaged.flattened().map { ($0.0, $0.1.floats) }))
        }

        for rank in 0..<2 {
            let got = try XCTUnwrap(results.get(rank), "rank \(rank) produced nothing")
            XCTAssertEqual(Set(got.keys), Set(expected.keys), "rank \(rank) key set")
            for (key, want) in expected {
                let have = try XCTUnwrap(got[key], "rank \(rank) missing \(key)")
                XCTAssertEqual(have.count, want.count, "rank \(rank) \(key) length")
                for i in 0..<Swift.min(have.count, want.count) {
                    XCTAssertEqual(have[i], want[i], accuracy: 1e-5,
                                   "rank \(rank) \(key)[\(i)]")
                }
            }
        }
    }

    /// Both ranks must come out of the average holding bit-identical gradients,
    /// or they will step apart and the models will diverge over a run.
    func testBothRanksReceiveTheSameAveragedGradients() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<[String: [Float]]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let model = self.makeModel()
            let (x, y) = self.data(rows: 12, inputs: 6, outputs: 3, offset: comm.rank * 400)
            let (_, grads) = valueAndGrad(model: model, self.lossFunction)(model, x, y)
            let averaged = try averageGradients(grads, comm: comm)
            eval(averaged)
            results.set(
                comm.rank,
                Dictionary(uniqueKeysWithValues: averaged.flattened().map { ($0.0, $0.1.floats) }))
        }
        let a = try XCTUnwrap(results.get(0))
        let b = try XCTUnwrap(results.get(1))
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        for key in a.keys {
            XCTAssertEqual(a[key], b[key], "\(key) differs between the ranks")
        }
    }

    // MARK: - Structure

    /// The tree comes back with the same keys and shapes it went in with —
    /// otherwise `model.update(parameters:)` on the far side would fail or,
    /// worse, silently reshape something.
    func testShapesAndKeysSurviveTheRoundTrip() throws {
        try MLXRuntime.requireMetallib()
        let results = RankResults<[(String, [Int])]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let model = self.makeModel()
            let (x, y) = self.data(rows: 8, inputs: 6, outputs: 3, offset: comm.rank * 100)
            let (_, grads) = valueAndGrad(model: model, self.lossFunction)(model, x, y)
            let before = grads.flattened().sorted { $0.0 < $1.0 }.map { ($0.0, $0.1.shape) }
            let averaged = try averageGradients(grads, comm: comm)
            let after = averaged.flattened().sorted { $0.0 < $1.0 }.map { ($0.0, $0.1.shape) }
            XCTAssertEqual(before.map(\.0), after.map(\.0), "keys changed")
            XCTAssertEqual(before.map(\.1), after.map(\.1), "shapes changed")
            results.set(comm.rank, after)
        }
        XCTAssertEqual(results.get(0)?.map(\.0), results.get(1)?.map(\.0))
    }

    /// A one-rank world returns the gradients untouched. The demo's
    /// single-process control runs through exactly this path, so it must not
    /// perturb anything.
    func testSingleRankWorldReturnsGradientsUnchanged() throws {
        try MLXRuntime.requireMetallib()
        let comm = Communicator(rank: 0, worldSize: 1, topology: .uniform(nodeCount: 1))
        defer { comm.shutdown() }
        let model = makeModel()
        let (x, y) = data(rows: 8, inputs: 6, outputs: 3, offset: 0)
        let (_, grads) = valueAndGrad(model: model, lossFunction)(model, x, y)
        eval(grads)
        let before = grads.flattened().sorted { $0.0 < $1.0 }.map { $0.1.floats }
        let averaged = try averageGradients(grads, comm: comm)
        let after = averaged.flattened().sorted { $0.0 < $1.0 }.map { $0.1.floats }
        XCTAssertEqual(before, after)
    }

    // MARK: - Compression

    /// Compression applies to the fused gradient buffer, and the result is
    /// still close enough to the exact average to train with.
    func testGradientAveragingUnderCompression() throws {
        try MLXRuntime.requireMetallib()
        let exact = RankResults<[String: [Float]]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let model = self.makeModel()
            let (x, y) = self.data(rows: 16, inputs: 6, outputs: 3, offset: comm.rank * 300)
            let (_, grads) = valueAndGrad(model: model, self.lossFunction)(model, x, y)
            let averaged = try averageGradients(grads, comm: comm)
            eval(averaged)
            exact.set(
                comm.rank,
                Dictionary(uniqueKeysWithValues: averaged.flattened().map { ($0.0, $0.1.floats) }))
        }
        let reference = try XCTUnwrap(exact.get(0))

        for (compression, tolerance, name) in [
            (WireCompression.downcast, Float(1e-3), "downcast"),
            (.int8Blockwise(blockSize: 128), 5e-2, "int8/128"),
        ] {
            let results = RankResults<[String: [Float]]>()
            try MLXRanks.run(worldSize: 2) { comm in
                let model = self.makeModel()
                let (x, y) = self.data(rows: 16, inputs: 6, outputs: 3, offset: comm.rank * 300)
                let (_, grads) = valueAndGrad(model: model, self.lossFunction)(model, x, y)
                let averaged = try averageGradients(grads, comm: comm, compression: compression)
                eval(averaged)
                results.set(
                    comm.rank,
                    Dictionary(uniqueKeysWithValues: averaged.flattened().map { ($0.0, $0.1.floats) }))
            }
            for rank in 0..<2 {
                let got = try XCTUnwrap(results.get(rank), "\(name) rank \(rank)")
                for (key, want) in reference {
                    let have = try XCTUnwrap(got[key], "\(name) rank \(rank) \(key)")
                    for i in 0..<Swift.min(have.count, want.count) {
                        XCTAssertEqual(have[i], want[i], accuracy: tolerance,
                                       "\(name) rank \(rank) \(key)[\(i)]")
                    }
                }
            }
        }
    }

    /// Top-k over a whole model uses one residual for the fused buffer.
    ///
    /// As in `MLXCollectiveTests`, the property is about the **running mean**
    /// over a sequence of calls, not about any single call: a single sparsified
    /// all-reduce is meant to be lossy, and error feedback is what makes the
    /// average of many of them land on the uncompressed answer.
    func testGradientAveragingUnderTopKConvergesOverSteps() throws {
        try MLXRuntime.requireMetallib()
        let iterations = 40
        let results = RankResults<(early: Float, late: Float)>()
        try MLXRanks.run(worldSize: 2) { comm in
            let model = self.makeModel()
            let (x, y) = self.data(rows: 16, inputs: 6, outputs: 3, offset: comm.rank * 300)
            let lossAndGrad = valueAndGrad(model: model, self.lossFunction)

            // The exact average, for reference. Taken on its own communicator
            // stream so it does not disturb the residual measured below.
            let (_, grads) = lossAndGrad(model, x, y)
            let exact = try averageGradients(grads, comm: comm)
            eval(exact)
            let target = exact.flattened().sorted { $0.0 < $1.0 }.flatMap { $0.1.floats }

            var running = [Float](repeating: 0, count: target.count)
            var early: Float = 0
            for step in 1...iterations {
                let (_, g) = lossAndGrad(model, x, y)
                let sparse = try averageGradients(
                    g, comm: comm, compression: .topK(fraction: 0.2))
                eval(sparse)
                let got = sparse.flattened().sorted { $0.0 < $1.0 }.flatMap { $0.1.floats }
                for i in 0..<running.count { running[i] += got[i] }
                if step == 4 { early = Self.meanError(running, target, steps: step) }
            }
            results.set(comm.rank, (early, Self.meanError(running, target, steps: iterations)))
        }
        for rank in 0..<2 {
            let (early, late) = try XCTUnwrap(results.get(rank))
            XCTAssertGreaterThan(early, 0, "rank \(rank): top-k should start out lossy")
            XCTAssertLessThan(late, early / 2,
                              "rank \(rank): the running mean is not converging "
                              + "(4 steps: \(early), \(iterations) steps: \(late))")
        }
    }

    private static func meanError(_ running: [Float], _ target: [Float], steps: Int) -> Float {
        var total: Float = 0
        for i in 0..<target.count { total += abs(running[i] / Float(steps) - target[i]) }
        return total / Float(target.count)
    }

    // MARK: - Scalars

    /// A per-rank loss is a 0-d array, which is the one shape where "how many
    /// elements is this" has a non-obvious answer. Both spellings are checked:
    /// the rank-0 scalar `valueAndGrad` actually returns, and the explicit
    /// one-element vector.
    func testAverageScalarMatchesTheMeanOfTheRanks() throws {
        try MLXRuntime.requireMetallib()
        let scalars = RankResults<Float>()
        let vectors = RankResults<Float>()
        let sizes = RankResults<[Int]>()
        try MLXRanks.run(worldSize: 2) { comm in
            let mine = Float(comm.rank) * 2 + 1
            let scalar = MLXArray(mine)
            let vector = MLXArray([mine], [1])
            sizes.set(comm.rank, [scalar.size, scalar.ndim, vector.size, vector.ndim])
            vectors.set(comm.rank, try averageScalar(vector, comm: comm))
            scalars.set(comm.rank, try averageScalar(scalar, comm: comm))
        }
        // (1 + 3) / 2
        for rank in 0..<2 {
            XCTAssertEqual(vectors.get(rank) ?? 0, 2.0, accuracy: 1e-6,
                           "rank \(rank) [1]-shaped, sizes \(sizes.get(rank) ?? [])")
            XCTAssertEqual(scalars.get(rank) ?? 0, 2.0, accuracy: 1e-6,
                           "rank \(rank) 0-d, sizes \(sizes.get(rank) ?? [])")
        }
    }
}
