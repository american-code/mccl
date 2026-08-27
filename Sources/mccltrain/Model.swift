import Foundation
import MLX
import MLXNN

// The model, the data and the optimiser for the data-parallel demo.
//
// All three are deliberately the simplest thing that still exercises the whole
// path, because the demo's job is to prove the *collective* correct, not to
// train anything interesting. Every source of nondeterminism is removed:
//
//  * Weights are seeded from a pure-Swift PRNG rather than MLX's, so two ranks
//    (and two runs, and two machines) start from bit-identical parameters
//    without having to agree on MLX's RNG stream.
//  * The dataset is a pure function of the seed, generated identically on every
//    rank; a rank takes its shard by index rather than being sent one.
//  * The optimiser is plain SGD, written out. Not because Adam would be wrong —
//    averaging gradients is correct under any optimiser that is a function of
//    the gradient — but because `p -= lr * g` makes the equivalence argument
//    checkable by eye: SGD is linear in the gradient, so "average the shard
//    gradients, then step" and "step on the full-batch gradient" are the same
//    arithmetic, not merely the same answer to five decimal places.
//
// The MLP shape follows Neural/MLP.swift in SwiftSci (a `Sequential` of
// `Linear` with ReLU between the hidden layers); its `train()` is also where the
// compiled-step pattern noted in `Trainer.swift` comes from.

/// Deterministic, portable PRNG. `SystemRandomNumberGenerator` is not
/// reproducible and `Float.random` is not portable across platforms; the demo's
/// entire correctness argument rests on both ranks seeing the same numbers.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    mutating func uniform() -> Float {
        Float(next() >> 40) / Float(1 << 23) - 1.0
    }

    /// Approximately standard normal — the sum of twelve uniforms, which is
    /// plenty for initialising a demo and costs no transcendental functions.
    mutating func normal() -> Float {
        var total: Float = 0
        for _ in 0..<12 { total += (uniform() + 1) / 2 }
        return total - 6
    }
}

/// An MLP with a deterministic initialisation.
enum DemoModel {

    /// `dimensions` is `[inputs, hidden..., outputs]`.
    static func make(dimensions: [Int], seed: UInt64) -> Sequential {
        precondition(dimensions.count >= 2, "need at least [inputs, outputs]")
        let layers = zip(dimensions, dimensions.dropFirst()).map { Linear($0, $1) }
        let model = Sequential(layers: layers.map { $0 as any UnaryLayer })
        initialise(model, seed: seed)
        return model
    }

    /// Overwrites every parameter from the seeded PRNG.
    ///
    /// Keyed on the *sorted parameter path*, so the value a given weight gets
    /// depends only on the model's structure — not on iteration order, not on
    /// how many parameters were visited first, and not on MLX's internals.
    static func initialise(_ model: Sequential, seed: UInt64) {
        var replacements: [(String, MLXArray)] = []
        for (index, entry) in model.parameters().flattened()
            .sorted(by: { $0.0 < $1.0 }).enumerated() {
            let (key, value) = entry
            var rng = SplitMix64(seed: seed &+ UInt64(index) &* 0x9E37_79B9)
            // He-style scale on weights (fan-in is the last axis of a Linear's
            // weight), zeros on biases — the usual choice, and stable.
            let isBias = value.shape.count == 1
            let fanIn = Swift.max(1, value.shape.last ?? 1)
            let scale = isBias ? 0 : (2.0 / Float(fanIn)).squareRoot()
            let flat = (0..<value.size).map { _ in rng.normal() * scale }
            replacements.append((key, MLXArray(flat, value.shape)))
        }
        model.update(parameters: ModuleParameters.unflattened(replacements))
        eval(model)
    }
}

/// A synthetic regression problem: a fixed random teacher network plus noise.
///
/// A teacher rather than a closed-form function so the loss actually falls in a
/// few hundred steps — a linear target would be solved immediately and the loss
/// trajectory, which is the thing being compared, would be a flat line and prove
/// nothing.
struct DemoData {
    let x: MLXArray
    let y: MLXArray
    let rows: Int

    init(rows: Int, inputs: Int, outputs: Int, seed: UInt64) {
        var rng = SplitMix64(seed: seed)
        let features = (0..<(rows * inputs)).map { _ in rng.uniform() }

        // The teacher: one hidden layer, generated from the same stream so the
        // whole dataset is a function of `seed` alone.
        let hidden = 16
        let w1 = (0..<(inputs * hidden)).map { _ in rng.normal() * 0.6 }
        let w2 = (0..<(hidden * outputs)).map { _ in rng.normal() * 0.6 }

        var targets = [Float](repeating: 0, count: rows * outputs)
        for row in 0..<rows {
            var h = [Float](repeating: 0, count: hidden)
            for j in 0..<hidden {
                var total: Float = 0
                for i in 0..<inputs { total += features[row * inputs + i] * w1[i * hidden + j] }
                h[j] = Swift.max(0, total)          // ReLU
            }
            for k in 0..<outputs {
                var total: Float = 0
                for j in 0..<hidden { total += h[j] * w2[j * outputs + k] }
                targets[row * outputs + k] = total + rng.normal() * 0.02
            }
        }

        self.rows = rows
        self.x = MLXArray(features, [rows, inputs])
        self.y = MLXArray(targets, [rows, outputs])
        eval(self.x, self.y)
    }

    /// The slice of the global batch for `step` that belongs to `rank`.
    ///
    /// The global batch is a contiguous window that wraps, and it is split into
    /// `worldSize` equal contiguous shards. Equal shards are load-bearing: the
    /// averaged gradient equals the full-batch gradient only when every rank
    /// contributes the same number of rows.
    func shard(step: Int, batch: Int, rank: Int, worldSize: Int) -> (MLXArray, MLXArray) {
        precondition(batch % worldSize == 0, "batch must divide evenly across ranks")
        let perRank = batch / worldSize
        let start = (step * batch + rank * perRank) % rows
        let indices = (0..<perRank).map { Int32((start + $0) % rows) }
        let picker = MLXArray(indices)
        return (x[picker], y[picker])
    }
}

/// Plain SGD, applied to a gradient tree.
///
/// Written out rather than taken from `MLXOptimizers` so that the update is
/// visible at the call site: this is the step whose linearity in the gradient
/// makes the 2-rank run and the single-process run provably identical.
enum SGD {
    static func step(model: Sequential, gradients: ModuleParameters, learningRate: Float) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        let grads = Dictionary(uniqueKeysWithValues: gradients.flattened())
        var updated: [(String, MLXArray)] = []
        updated.reserveCapacity(parameters.count)
        for (key, value) in parameters {
            guard let g = grads[key] else {
                updated.append((key, value))
                continue
            }
            updated.append((key, value - learningRate * g))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }
}

func demoLoss(_ model: Sequential, _ x: MLXArray, _ y: MLXArray) -> MLXArray {
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}
