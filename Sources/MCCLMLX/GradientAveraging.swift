import Foundation
import MCCL
import MLX
import MLXNN

// Data-parallel gradient averaging.
//
// This is the thing an MLX training loop actually wants from a collectives
// library, and it is not simply "call all-reduce once per tensor".
//
// **Why one fused collective and not one per parameter.** A small MLP has eight
// or ten gradient tensors. Each all-reduce over the Thunderbolt cable costs at
// least one round trip — 167.8 µs measured, before a single byte of payload
// moves — so ten separate all-reduces spend 1.7 ms on latency alone, which for
// an MLP is more than the whole backward pass. Flattening the gradients into one
// contiguous buffer turns that into a single round trip, and lets the ring
// amortise its fixed cost over a message big enough to reach its bandwidth
// plateau. This is the same reason NCCL has `ncclGroupStart`.
//
// **Why the key order is sorted and the dtype order is fixed.** Every rank has
// to lay its buffer out identically or the ranks will average unrelated numbers
// and no error will be raised — the byte counts would still match. Sorting the
// flattened parameter paths and grouping dtypes in a fixed canonical order makes
// the layout a function of the model's structure alone, which is the one thing
// every rank is guaranteed to agree on.

/// Averages a gradient tree across the world in place of each rank's local
/// gradients, and hands back the averaged tree.
///
/// This is the whole data-parallel step, minus the optimiser: each rank computes
/// gradients on its own shard of the batch, this averages them, and every rank
/// then applies the identical update. With equal shards the averaged gradient is
/// exactly the gradient of the loss over the combined batch, which is what makes
/// an N-rank run reproduce a single-process run on N times the data.
///
/// Takes the gradient tree rather than the model because a model does not hold
/// gradients — `MLXNN.valueAndGrad` returns them separately, and this is meant to
/// sit directly on that call's result:
///
/// ```swift
/// let (loss, grads) = lossAndGrad(model, x, y)
/// let averaged = try averageGradients(grads, comm: comm)
/// // ... apply `averaged` with any optimiser
/// ```
///
/// - Parameters:
///   - gradients: the tree returned by `valueAndGrad`.
///   - comm: a communicator whose world is the set of data-parallel ranks.
///   - compression: applied to the fused buffer on the wire only. `.topK` keeps
///     one error-feedback residual per dtype group, so a model with a single
///     dtype — which is the normal case — gets exactly one residual covering the
///     whole flattened gradient, and calling this repeatedly across a training
///     run accumulates it correctly.
/// - Returns: a tree with the same keys and shapes, holding the world average.
public func averageGradients(
    _ gradients: ModuleParameters,
    comm: Communicator,
    compression: WireCompression = .none
) throws -> ModuleParameters {
    // A one-rank world has nothing to average and no fabric to do it over.
    guard comm.worldSize > 1 else { return gradients }

    let entries = gradients.flattened().sorted { $0.0 < $1.0 }
    guard !entries.isEmpty else { return gradients }

    var averaged: [String: MLXArray] = [:]
    averaged.reserveCapacity(entries.count)

    // Canonical dtype order, not first-appearance order: two ranks must group
    // identically even if their trees were built in different orders.
    for (group, dtype) in MLXDataType.supported.enumerated() {
        let members = entries.filter { $0.1.dtype == dtype }
        guard !members.isEmpty else { continue }
        let reduced = try averageGroup(
            members, dtype: dtype, comm: comm, compression: compression,
            stream: StreamID(UInt32(group)))
        for (key, value) in reduced { averaged[key] = value }
    }

    // Anything left over is a dtype mccl does not carry. Fail loudly rather
    // than returning a tree in which some entries were silently not reduced.
    if averaged.count != entries.count {
        let missing = entries.filter { averaged[$0.0] == nil }
        let described = missing.map { "\($0.0) (\($0.1.dtype))" }.joined(separator: ", ")
        throw MCCLError.invalidArgument(
            "gradient tree contains dtypes mccl cannot reduce: \(described)")
    }

    return ModuleParameters.unflattened(entries.map { ($0.0, averaged[$0.0]!) })
}

/// One fused all-reduce over every gradient sharing a dtype.
private func averageGroup(
    _ members: [(String, MLXArray)],
    dtype: DType,
    comm: Communicator,
    compression: WireCompression,
    stream: StreamID
) throws -> [(String, MLXArray)] {
    let dataType = try MLXDataType.mccl(dtype)

    // `concatenated` produces a fresh contiguous array that nothing else holds,
    // which is the precondition `unsafelyAdopting` needs. The flatten to 1-D is
    // what makes the concatenation well-defined across differently-shaped
    // tensors.
    let flat = MLX.concatenated(members.map { $0.1.reshaped([-1]) }, axis: 0)
    flat.eval()
    let work = try MLXWorkingBuffer.copying(flat)

    let borrowed = Borrowed(work.buffer)
    let count = work.count
    try runBlocking {
        try await comm.allReduce(
            borrowed.value, count: count, dataType: dataType,
            op: .avg, compression: compression, stream: stream)
    }
    let reduced = work.finish()

    // Slice the reduced buffer back into the original shapes. These are MLX
    // views onto the fused array, so nothing is copied out; the fused buffer
    // stays alive as long as any gradient derived from it does.
    var results: [(String, MLXArray)] = []
    results.reserveCapacity(members.count)
    var offset = 0
    for (key, original) in members {
        let n = original.size
        let slice = n == 0
            ? MLX.zeros(original.shape, dtype: dtype)
            : reduced[offset ..< (offset + n)].reshaped(original.shape)
        results.append((key, slice))
        offset += n
    }
    return results
}

/// The scalar companion to `averageGradients`: the world's mean of a per-rank
/// loss, so a data-parallel run can report the same number a single-process run
/// on the combined batch would.
///
/// Separate from the gradient path on purpose. Fusing the loss into the gradient
/// buffer would make the reported loss depend on the gradient compression
/// setting, and a metric that moves when you change the wire codec is not a
/// metric.
public func averageScalar(_ value: MLXArray, comm: Communicator) throws -> Float {
    guard comm.worldSize > 1 else { return value.item(Float.self) }
    let reduced = try comm.allReduce(value.asType(.float32), op: .avg)
    return reduced.item(Float.self)
}
