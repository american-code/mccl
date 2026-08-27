import Foundation
import MCCL
import MLX

// The MLX seam.
//
// mccl's collectives take `UnsafeMutableRawBufferPointer` + `DataType` + element
// count. An evaluated, contiguous `MLXArray` has exactly those three things, and
// because MLX allocates in unified memory the pointer it hands over is the same
// address the GPU reads — there is no host/device staging step anywhere in this
// file. That is the property that makes an adapter rather than a copy layer
// possible.
//
// Two things still have to be arranged before mccl may look at the memory, and
// they are the whole reason this file exists.
//
// **Laziness.** An `MLXArray` is a node in a graph until something forces it.
// `mlx_array_data_uint8` on an unevaluated array is not a buffer of results.
// Every entry point here calls `eval()` first — see `MLXWorkingBuffer`.
//
// **Layout.** A strided or broadcast array has no single contiguous run of
// bytes to send. `asData(access: .noCopyIfContiguous)` returns the backing
// directly when the layout allows and materialises a contiguous copy when it
// does not, which is the correct behaviour in both cases and is why it is used
// instead of reaching for the pointer directly.
//
// ## What actually gets copied
//
// The interesting zero-copy property is free and always holds: because MLX
// allocates in unified memory, the bytes handed to mccl *are* the bytes the GPU
// reads. Nothing is staged across a bus. A CUDA equivalent of this adapter would
// have to `cudaMemcpy` every payload to host memory and back; this one does not.
//
// What is *not* free is writing a result into an `MLXArray`'s backing, and the
// reason is worth recording because it is not in any documentation:
//
//  * mlx-swift's only public route to an array's bytes is `asData(access:)`,
//    which hands back a Foundation `Data`. `Cmlx` — where `mlx_array_data_uint8`
//    lives — is not a public product of the package, so there is no supported
//    way to ask an `MLXArray` for a mutable pointer.
//  * `Data(bytesNoCopy:count:deallocator:)`, which is what `.noCopy` builds, is
//    **not** no-copy for small buffers. Foundation stores payloads of 14 bytes
//    or fewer in inline storage, copying them; `withUnsafeBytes` then yields the
//    address of that temporary. Measured at the boundary in
//    `MLXBridgeTests.testFoundationInlinesSmallNoCopyData`: 14 bytes copies,
//    15 bytes aliases.
//
// So writing through the wrapper is correct above the threshold and silently
// loses every write below it — a bias vector of three floats is 12 bytes. That
// is not a bug worth risking to save a memcpy, so `MLXWorkingBuffer` has two
// backings and picks by size:
//
// | payload | backing | copies |
// |---|---|---|
// | ≥ `adoptionFloor` (64 B) | the result array's own storage, written in place | 1 (in) |
// | < `adoptionFloor` | a scratch buffer this adapter owns | 2 (in and out) |
//
// Both are correct; the threshold only decides which. It is set well clear of
// Foundation's 14-byte boundary rather than at it, and
// `MLXCollectiveTests.testAllReduceAcrossTheAdoptionThreshold` sweeps element
// counts across it so a Foundation change is caught by a failing test rather
// than by a wrong answer.
//
// One copy is unavoidable in any case, and it is not a bridging artefact: it is
// the price of *not* mutating the caller's array. mccl's all-reduce is in-place,
// so the output buffer has to start life holding this rank's contribution. MLX
// arrays share backings freely (`reshaped`, `broadcast`, a slice of a larger
// tensor), so reducing over the input's storage would silently corrupt every
// array aliasing it.
//
// The size of the price, measured: a memcpy runs at host memory bandwidth
// (~100 GB/s — the rate `Kernels.scale` achieves on this machine) against a
// Thunderbolt fabric that delivers 1.06 GB/s. One copy of an N-byte payload
// costs about **1%** of the time that payload spends on the cable, two costs
// about 2%. `allGather` pays neither: mccl's all-gather takes separate send and
// receive buffers, so the send side reads the input's backing and the receive
// side is written directly.

// MARK: - Data types

/// Translation between MLX's `DType` and mccl's `DataType`.
///
/// Five of MLX's dtypes have mccl equivalents. The rest are an error rather than
/// a silent cast: reducing a `float64` tensor by quietly narrowing it to
/// `float32` would produce a plausible, wrong answer.
public enum MLXDataType {

    /// The dtypes an `MLXArray` can be handed to mccl in.
    public static let supported: [DType] = [.float32, .float16, .bfloat16, .int32, .int8]

    public static func mccl(_ dtype: DType) throws -> DataType {
        switch dtype {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int32: return .int32
        case .int8: return .int8
        default:
            throw MCCLError.invalidArgument(
                "MLX dtype \(dtype) has no mccl equivalent; mccl carries "
                + "\(supported.map(String.init(describing:)).joined(separator: ", "))"
                + ". Cast the array yourself if a narrowing conversion is what you want.")
        }
    }

    public static func mlx(_ dataType: DataType) -> DType {
        switch dataType {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int32: return .int32
        case .int8: return .int8
        }
    }
}

// MARK: - Buffers

/// A contiguous, mutable buffer that mccl may reduce in place, and which can be
/// turned back into an `MLXArray` when the collective is done.
///
/// See the table at the top of this file for which of the two backings is used
/// and why. `finish()` must be called exactly once: it is what releases a
/// scratch allocation and what produces the result array.
final class MLXWorkingBuffer {

    /// Payloads at or above this many bytes are reduced directly in the result
    /// array's own storage; smaller ones go through a scratch buffer.
    ///
    /// The real boundary is Foundation's inline-`Data` limit — 14 bytes, above
    /// which `Data(bytesNoCopy:)` genuinely aliases. 64 leaves four times that
    /// margin, costs one extra memcpy on payloads too small to measure, and
    /// means a change in Foundation moves a *correct* path's threshold rather
    /// than breaking anything.
    static let adoptionFloor = 64

    private enum Backing {
        /// mccl writes `array`'s own storage. `array` also keeps it alive.
        case adopted(MLXArray)
        /// mccl writes memory this object owns; `finish()` copies it into a new
        /// array and frees it.
        case scratch(UnsafeMutableRawBufferPointer, shape: [Int], dtype: DType)
    }

    let base: UnsafeMutableRawPointer
    let byteCount: Int
    let count: Int
    let dataType: DataType
    private let backing: Backing
    private var finished = false

    private init(base: UnsafeMutableRawPointer, byteCount: Int, count: Int,
                 dataType: DataType, backing: Backing) {
        self.base = base
        self.byteCount = byteCount
        self.count = count
        self.dataType = dataType
        self.backing = backing
    }

    var buffer: UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: base, count: byteCount)
    }

    /// True when mccl is writing the result array's own storage rather than a
    /// scratch allocation. Exposed for the tests that pin the threshold.
    var isAdopted: Bool {
        if case .adopted = backing { return true }
        return false
    }

    /// The result. Call once, after the collective has returned.
    func finish() -> MLXArray {
        precondition(!finished, "MLXWorkingBuffer.finish() called twice")
        finished = true
        switch backing {
        case .adopted(let array):
            return array
        case .scratch(let scratch, let shape, let dtype):
            defer { scratch.deallocate() }
            // `mlx_array_new_data` copies, which is the second copy the small
            // path pays and the reason the threshold exists.
            let data = Data(bytesNoCopy: scratch.baseAddress!, count: scratch.count,
                            deallocator: .none)
            return MLXArray(data, shape, dtype: dtype)
        }
    }

    deinit {
        if case .scratch(let scratch, _, _) = backing, !finished { scratch.deallocate() }
    }

    // MARK: - Construction

    /// A working buffer holding a copy of `source`'s contents.
    static func copying(_ source: MLXArray) throws -> MLXWorkingBuffer {
        let dataType = try MLXDataType.mccl(source.dtype)
        source.eval()
        // Zero-copy when the layout allows; a contiguous materialisation when
        // it does not. In practice MLX's own `eval()` has already produced a
        // contiguous array by this point, so the fallback rarely fires.
        let contiguous = source.asData(access: .noCopyIfContiguous)

        if source.nbytes >= adoptionFloor {
            // `mlx_array_new_data` copies the bytes into fresh storage nobody
            // else holds — the precondition `adopt` needs.
            let out = MLXArray(contiguous.data, source.shape, dtype: source.dtype)
            return try adopt(out, dataType: dataType)
        }

        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: Swift.max(1, source.nbytes), alignment: 64)
        scratch.initializeMemory(as: UInt8.self, repeating: 0)
        contiguous.data.withUnsafeBytes { raw in
            if let from = raw.baseAddress {
                scratch.baseAddress!.copyMemory(from: from, byteCount: source.nbytes)
            }
        }
        return MLXWorkingBuffer(
            base: scratch.baseAddress!, byteCount: source.nbytes, count: source.size,
            dataType: dataType,
            backing: .scratch(scratch, shape: source.shape, dtype: source.dtype))
    }

    /// A zero-filled working buffer of the given shape — the receive side of an
    /// all-gather or a reduce-scatter, which never needs the caller's bytes.
    static func zeros(shape: [Int], dtype: DType) throws -> MLXWorkingBuffer {
        let dataType = try MLXDataType.mccl(dtype)
        let count = shape.reduce(1, *)
        let byteCount = count * dtype.size

        if byteCount >= adoptionFloor {
            return try adopt(MLX.zeros(shape, dtype: dtype), dataType: dataType)
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: Swift.max(1, byteCount), alignment: 64)
        scratch.initializeMemory(as: UInt8.self, repeating: 0)
        return MLXWorkingBuffer(
            base: scratch.baseAddress!, byteCount: byteCount, count: count,
            dataType: dataType, backing: .scratch(scratch, shape: shape, dtype: dtype))
    }

    /// Wraps an array this adapter created and nobody else holds, exposing its
    /// storage for in-place mutation.
    ///
    /// Only ever called on arrays constructed a line or two above — a fresh
    /// `mlx_array_new_data`, a `zeros`, or a `concatenated` — and only when the
    /// payload is large enough that `Data(bytesNoCopy:)` really does alias.
    /// Passing an arbitrary caller-supplied array here would be exactly the
    /// aliasing bug the copy exists to avoid.
    static func adopt(_ array: MLXArray, dataType: DataType) throws -> MLXWorkingBuffer {
        precondition(array.nbytes >= adoptionFloor,
                     "adopting a \(array.nbytes)-byte array: Foundation may have inlined it")
        array.eval()
        let view = array.asData(access: .noCopy)
        guard let base = view.data.withUnsafeBytes({ $0.baseAddress }) else {
            throw MCCLError.invalidArgument("MLX array has no backing store")
        }
        return MLXWorkingBuffer(
            base: UnsafeMutableRawPointer(mutating: base),
            byteCount: array.nbytes, count: array.size,
            dataType: dataType, backing: .adopted(array))
    }
}

/// Runs `body` over `array`'s bytes without copying them when the array is
/// already contiguous. Read-only: mccl is never given a chance to write here.
func withContiguousBytes<T>(
    of array: MLXArray, _ body: (UnsafeRawBufferPointer) throws -> T
) rethrows -> T {
    array.eval()
    let contiguous = array.asData(access: .noCopyIfContiguous)
    return try contiguous.data.withUnsafeBytes(body)
}

// MARK: - Async bridge

/// Carries a raw buffer across the blocking bridge.
///
/// The same argument as `Borrowed` in `Sources/MCCLShim/CABI.swift`: the call
/// blocks until the collective has finished, so exactly one thread touches the
/// memory at a time and nothing can free it underneath the operation.
struct Borrowed<Pointer>: @unchecked Sendable {
    let value: Pointer
    init(_ value: Pointer) { self.value = value }
}

private final class ThrowBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Error?
    var error: Error? {
        get { lock.lock(); defer { lock.unlock() }; return storage }
        set { lock.lock(); storage = newValue; lock.unlock() }
    }
}

/// Drives an `async` mccl collective to completion from a synchronous caller.
///
/// The MLX-facing surface is synchronous on purpose. `mlx.core.distributed`'s
/// `all_sum` returns a value, not an awaitable, and an MLX training loop is
/// ordinary straight-line code; an `async` adapter would force every call site
/// into a `Task` for no benefit. This is the same bridge the C shim uses, for
/// the same reason.
func runBlocking(_ body: @escaping @Sendable () async throws -> Void) throws {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ThrowBox()
    Task.detached {
        do { try await body() } catch { box.error = error }
        semaphore.signal()
    }
    semaphore.wait()
    if let error = box.error { throw error }
}
