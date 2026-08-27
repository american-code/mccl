import Foundation
import MCCL
import MCCLMLX
import MLX
import MLXNN

// The data-parallel training step, and the two ways of running it.
//
// ## The step
//
//     (loss, grads) = valueAndGrad(model, shard)     // local, on the GPU
//     eval(loss, grads)                              // force the graph
//     averaged = averageGradients(grads, comm)       // one fused all-reduce
//     SGD.step(model, averaged)                      // identical on every rank
//
// The `eval` in the middle is not incidental — it is the shape of the whole
// problem. MLX is lazy: `grads` is a graph, not numbers, and mccl needs bytes.
// So the backward pass has to be forced before the collective can start, which
// also means the collective cannot be fused into the compiled graph.
//
// That is why this loop is *not* written the way SwiftSci's `Neural/Trainer.swift`
// writes its `train()` — `compile(inputs: state, outputs: state)` around
// forward + backward + optimiser update, which fuses the whole step into one
// Metal graph. Here the all-reduce sits between the backward pass and the
// update, on the host, so the step is necessarily cut in two. Compiling only the
// forward+backward half is possible; it is left out because at this model scale
// the communication dominates by more than an order of magnitude (see the
// numbers in docs/USAGE.md) and a faster local half would not move the total.
// The gotcha SwiftSci records — an optimiser whose state is allocated lazily
// inside a compiled trace silently stops updating — is avoided here for free,
// since SGD has no state.
//
// ## Why the loss is all-reduced too
//
// Each rank's loss is the mean over its own shard. With equal shards the world
// mean of those is exactly the loss over the combined batch, which is the
// quantity the single-process control computes — so comparing them is a like-for-like
// comparison. Reporting rank 0's local loss instead would compare a half-batch
// loss against a full-batch loss and the trajectories would differ for reasons
// that have nothing to do with the collective.

struct TrainConfig {
    var inputs = 32
    var outputs = 4
    var hidden = [128, 128]
    var rows = 4096
    var batch = 256
    var steps = 200
    var warmup = 5
    var learningRate: Float = 0.05
    var seed: UInt64 = 20260826
    var compression: WireCompression = .none
    var compressionName = "none"
}

struct TrainResult {
    /// World-mean loss after every step, in order. The trajectory that has to
    /// match between a 1-rank and an N-rank run.
    var losses: [Float]
    /// Wall time of the timed region (everything after the warm-up steps).
    var seconds: Double
    var timedSteps: Int
    /// Seconds spent inside `averageGradients`, summed over the timed region.
    var communicationSeconds: Double
    var parameterCount: Int
    /// Bytes one all-reduce carries, uncompressed.
    var gradientBytes: Int

    var stepsPerSecond: Double { seconds > 0 ? Double(timedSteps) / seconds : 0 }
    var communicationFraction: Double { seconds > 0 ? communicationSeconds / seconds : 0 }
}

enum Trainer {

    /// Runs `config.steps` data-parallel steps on this rank.
    ///
    /// A one-rank `comm` makes this the single-process control: `averageGradients`
    /// returns its input untouched, the rank's shard is the whole batch, and the
    /// loss is the full-batch loss. Same code path, which is the point — a
    /// control that ran different code would not be a control.
    static func run(comm: Communicator, config: TrainConfig) throws -> TrainResult {
        let dimensions = [config.inputs] + config.hidden + [config.outputs]
        let model = DemoModel.make(dimensions: dimensions, seed: config.seed)
        let data = DemoData(rows: config.rows, inputs: config.inputs,
                            outputs: config.outputs, seed: config.seed &+ 1)
        let lossAndGrad = valueAndGrad(model: model, demoLoss)

        let parameterCount = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
        var losses: [Float] = []
        losses.reserveCapacity(config.steps)

        var timedSeconds: Double = 0
        var communicationSeconds: Double = 0
        var timedSteps = 0

        for step in 0..<config.steps {
            let timed = step >= config.warmup
            let stepStart = DispatchTime.now().uptimeNanoseconds

            let (x, y) = data.shard(step: step, batch: config.batch,
                                    rank: comm.rank, worldSize: comm.worldSize)
            let (loss, grads) = lossAndGrad(model, x, y)
            // Force the backward pass. mccl reduces bytes, and until this line
            // `grads` is a graph.
            eval(loss, grads)

            let communicationStart = DispatchTime.now().uptimeNanoseconds
            let averaged = try averageGradients(grads, comm: comm, compression: config.compression)
            eval(averaged)
            let communicationEnd = DispatchTime.now().uptimeNanoseconds

            SGD.step(model: model, gradients: averaged, learningRate: config.learningRate)
            eval(model)

            // The reported loss is the world's, not this rank's.
            let reported = try averageScalar(loss, comm: comm)
            losses.append(reported)

            if timed {
                let end = DispatchTime.now().uptimeNanoseconds
                timedSeconds += Double(end - stepStart) / 1e9
                communicationSeconds += Double(communicationEnd - communicationStart) / 1e9
                timedSteps += 1
            }
        }

        return TrainResult(
            losses: losses, seconds: timedSeconds, timedSteps: timedSteps,
            communicationSeconds: communicationSeconds,
            parameterCount: parameterCount,
            gradientBytes: parameterCount * 4)
    }

    /// A degenerate one-rank communicator: no fabric, every collective a no-op.
    static func soloCommunicator() -> Communicator {
        Communicator(rank: 0, worldSize: 1, topology: .uniform(nodeCount: 1))
    }
}

// MARK: - Local verification

/// The correctness proof, run entirely on this machine.
///
/// Trains the same model twice — once in a single process on the whole batch,
/// once across two ranks that each see half of it and all-reduce their gradients
/// — and compares the loss trajectories step by step. If the adapter is correct
/// the two are the same computation, so they must agree to floating-point noise.
/// If it is wrong in any way that matters (a dropped write, a mis-ordered
/// flatten, a wrong scale factor) the trajectories diverge, usually within a few
/// steps.
///
/// The two ranks run in one process over **real TCP loopback sockets**, not the
/// in-process transport: the wire format, the framing and the ring are all in
/// the path, so this checks the same code the cluster runs.
enum Verifier {

    struct Report {
        let single: TrainResult
        let distributed: TrainResult
        let maxAbsoluteDifference: Float
        let maxRelativeDifference: Float
        let worstStep: Int
    }

    static func run(config: TrainConfig, tolerance: Float) throws -> Report {
        let solo = Trainer.soloCommunicator()
        defer { solo.shutdown() }
        let single = try Trainer.run(comm: solo, config: config)

        let comms = try Communicator.tcpGroup(worldSize: 2)
        defer { comms.forEach { $0.shutdown() } }

        let results = ResultBox()
        let group = DispatchGroup()
        for comm in comms {
            group.enter()
            let thread = Thread {
                defer { group.leave() }
                do { results.set(comm.rank, .success(try Trainer.run(comm: comm, config: config))) }
                catch { results.set(comm.rank, .failure(error)) }
            }
            thread.stackSize = 8 << 20
            thread.start()
        }
        guard group.wait(timeout: .now() + 600) == .success else {
            throw MCCLError.timedOut("two-rank verification did not finish within 600s")
        }
        guard let zero = results.get(0), let one = results.get(1) else {
            throw MCCLError.protocolViolation("a rank produced no result")
        }
        let distributed = try zero.get()
        let other = try one.get()

        // Both ranks stepped on the same averaged gradient, so they must agree
        // with each other before either is compared with the control.
        guard distributed.losses == other.losses else {
            let step = zip(distributed.losses, other.losses).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset ?? -1
            throw MCCLError.protocolViolation(
                "the two ranks disagree about the loss from step \(step) — they are not "
                + "training the same model")
        }

        var maxAbsolute: Float = 0
        var maxRelative: Float = 0
        var worst = 0
        for (index, pair) in zip(single.losses, distributed.losses).enumerated() {
            let absolute = abs(pair.0 - pair.1)
            let relative = pair.0 != 0 ? absolute / abs(pair.0) : absolute
            if absolute > maxAbsolute { maxAbsolute = absolute; worst = index }
            maxRelative = Swift.max(maxRelative, relative)
        }
        _ = tolerance
        return Report(single: single, distributed: distributed,
                      maxAbsoluteDifference: maxAbsolute,
                      maxRelativeDifference: maxRelative, worstStep: worst)
    }
}

final class ResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int: Result<TrainResult, Error>] = [:]

    func set(_ rank: Int, _ value: Result<TrainResult, Error>) {
        lock.lock(); values[rank] = value; lock.unlock()
    }
    func get(_ rank: Int) -> Result<TrainResult, Error>? {
        lock.lock(); defer { lock.unlock() }; return values[rank]
    }
}
