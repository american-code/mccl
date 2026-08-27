import Foundation
import MCCL
import MLX

// mccl's collectives, spelled in MLXArrays.
//
// Every entry point is synchronous and returns a *new* array, matching what an
// MLX caller expects from `mlx.core.distributed` — see the note on `runBlocking`
// and the copy accounting in `MLXBridge.swift`.
//
// Nothing here makes a decision. Dtype support, buffer sizing, compression
// legality, plan selection and the reduction itself are all stated once, in
// `MCCL`, and surfaced here as an error. The C shim is the model.

extension Communicator {

    /// All-reduce an `MLXArray` across the world.
    ///
    /// The input is not modified. The result has the same shape and dtype.
    ///
    /// `stream` only matters for `.topK`, which carries an error-feedback
    /// residual per stream: two different tensors reduced on one communicator
    /// must not share one, or each will be corrected with the other's leftovers.
    /// `averageGradients` handles this for a whole model; a caller reducing
    /// tensors one at a time has to pass distinct `StreamID`s itself.
    public func allReduce(
        _ array: MLXArray,
        op: ReduceOp = .sum,
        compression: WireCompression = .none,
        stream: StreamID = .default
    ) throws -> MLXArray {
        let work = try MLXWorkingBuffer.copying(array)
        let borrowed = Borrowed(work.buffer)
        let count = work.count
        let dataType = work.dataType
        try runBlocking {
            try await self.allReduce(
                borrowed.value, count: count, dataType: dataType,
                op: op, compression: compression, stream: stream)
        }
        return work.finish()
    }

    /// Reduce onto `root`. Every rank must call it; only `root`'s result is
    /// meaningful, exactly as in `MCCL` and in NCCL.
    public func reduce(
        _ array: MLXArray,
        op: ReduceOp = .sum,
        root: Int,
        compression: WireCompression = .none
    ) throws -> MLXArray {
        let work = try MLXWorkingBuffer.copying(array)
        let borrowed = Borrowed(work.buffer)
        let count = work.count
        let dataType = work.dataType
        try runBlocking {
            try await self.reduce(
                borrowed.value, count: count, dataType: dataType,
                op: op, root: root, compression: compression)
        }
        return work.finish()
    }

    /// Broadcast `root`'s copy of the array to every rank.
    ///
    /// Non-root ranks still have to pass an array: it supplies the shape and
    /// dtype the collective is defined over, and its contents are overwritten.
    public func broadcast(
        _ array: MLXArray,
        root: Int,
        compression: WireCompression = .none
    ) throws -> MLXArray {
        let work = try MLXWorkingBuffer.copying(array)
        let borrowed = Borrowed(work.buffer)
        let count = work.count
        let dataType = work.dataType
        try runBlocking {
            try await self.broadcast(
                borrowed.value, count: count, dataType: dataType,
                root: root, compression: compression)
        }
        return work.finish()
    }

    /// All-gather: every rank contributes `array`, every rank receives all of
    /// them stacked along a new leading axis of length `worldSize`.
    ///
    /// This is the collective that copies nothing, at least above
    /// `MLXWorkingBuffer.adoptionFloor`: mccl's all-gather takes separate send
    /// and receive buffers, so the send side reads the input's backing directly
    /// and the receive side writes the output array's backing directly.
    public func allGather(
        _ array: MLXArray,
        compression: WireCompression = .none
    ) throws -> MLXArray {
        let dataType = try MLXDataType.mccl(array.dtype)
        let out = try MLXWorkingBuffer.zeros(
            shape: [worldSize] + array.shape, dtype: array.dtype)
        let count = array.size
        try withContiguousBytes(of: array) { send in
            let sent = Borrowed(send)
            let received = Borrowed(out.buffer)
            try runBlocking {
                try await self.allGather(
                    sent.value, into: received.value, count: count,
                    dataType: dataType, compression: compression)
            }
        }
        return out.finish()
    }

    /// Reduce-scatter: the world reduces the whole of `array` and each rank
    /// keeps the slice its rank index selects.
    ///
    /// `array`'s leading dimension must be divisible by `worldSize`; the result
    /// is that leading dimension divided by `worldSize`, with the trailing
    /// shape unchanged.
    public func reduceScatter(
        _ array: MLXArray,
        op: ReduceOp = .sum,
        compression: WireCompression = .none
    ) throws -> MLXArray {
        let dataType = try MLXDataType.mccl(array.dtype)
        guard let leading = array.shape.first, leading % worldSize == 0 else {
            throw MCCLError.invalidArgument(
                "reduceScatter needs a leading dimension divisible by the world size; "
                + "got shape \(array.shape) for world size \(worldSize)")
        }
        var outShape = array.shape
        outShape[0] = leading / worldSize
        let out = try MLXWorkingBuffer.zeros(shape: outShape, dtype: array.dtype)
        let recvCount = out.count
        try withContiguousBytes(of: array) { send in
            let sent = Borrowed(send)
            let received = Borrowed(out.buffer)
            try runBlocking {
                try await self.reduceScatter(
                    sent.value, into: received.value, recvCount: recvCount,
                    dataType: dataType, op: op, compression: compression)
            }
        }
        return out.finish()
    }
}
