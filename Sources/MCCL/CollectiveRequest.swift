import Foundation

/// A collective that has been issued but not yet completed.
///
/// ## Why a request handle and not a stream
///
/// NCCL's asynchrony is the CUDA stream's: `ncclAllReduce` enqueues onto a
/// stream and returns, and the caller synchronises the *stream*. That works
/// because a CUDA stream is already the program's unit of ordering and already
/// has a synchronise call; NCCL borrows a mechanism that exists.
///
/// mccl has no device queue to borrow. Its `mcclStream_t` is deliberately not an
/// execution context — it names an independent *sequence* of collectives, which
/// is what lets top-k keep one error-feedback residual per sequence. Making that
/// same handle mean "the queue you wait on" would give one type two jobs, and
/// would silently turn every existing blocking call into a non-blocking one:
/// callers that read their buffer on return would start reading it mid-flight.
/// For a library whose stated priority is a stable C ABI, that is the wrong
/// trade.
///
/// So the asynchrony is additive and explicit, in MPI's idiom rather than
/// CUDA's: issue returns a handle, and the handle is waited or tested.
/// `mcclAllReduce` still blocks and still means exactly what it meant.
///
/// ## What the overlap actually buys, and what it does not
///
/// Every collective on one communicator runs on that communicator's serial
/// queue, so two outstanding requests execute one after the other, in the order
/// they were issued. That ordering is not a limitation to work around — it is
/// required. Ranks must run collectives in the same order or they deadlock
/// against each other, and issue order is the only order every rank agrees on.
///
/// What is genuinely overlapped is the caller's own work with the collective's
/// I/O. On a Mac cluster a collective is socket traffic, so a thread that issues
/// an all-reduce and goes on computing is the thing worth having. Two
/// collectives do not run at once; a collective and your program do.
public final class CollectiveRequest: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var storedError: Error?
    private var isComplete = false

    init() {}

    /// Called once, on the communicator's work queue, when the collective ends.
    func complete(_ error: Error?) {
        lock.lock()
        guard !isComplete else { lock.unlock(); return }
        isComplete = true
        storedError = error
        lock.unlock()
        semaphore.signal()
    }

    /// Whether the collective has finished. Does not block and does not consume
    /// the completion: a request that tests complete must still be waited.
    public var isFinished: Bool {
        lock.lock(); defer { lock.unlock() }
        return isComplete
    }

    /// Blocks until the collective finishes, then rethrows whatever it threw.
    ///
    /// Safe to call more than once — the second call sees the recorded state
    /// rather than waiting again — but the buffer is only safe to touch after
    /// the first one returns.
    public func wait() throws {
        lock.lock()
        let already = isComplete
        lock.unlock()
        if !already { semaphore.wait() }
        lock.lock(); defer { lock.unlock() }
        if let storedError { throw storedError }
    }
}

/// Carries a caller-owned raw buffer onto the work queue.
///
/// `UnsafeMutableRawBufferPointer` is not `Sendable`, and rightly so. Here the
/// crossing is safe by the same contract as the blocking path: exactly one
/// thread touches the memory at a time, and the caller has undertaken not to be
/// that thread until the request is waited.
struct UncheckedBuffer: @unchecked Sendable {
    let value: UnsafeMutableRawBufferPointer
    init(_ value: UnsafeMutableRawBufferPointer) { self.value = value }
}

extension Communicator {
    /// Issues an all-reduce and returns without waiting for it.
    ///
    /// The buffer must not be read or written until the returned request has
    /// been waited: the collective owns it in the meantime. Issue order is
    /// preserved across requests on one communicator, which is what keeps a
    /// program that issues several of them from deadlocking against its peers.
    public func allReduceAsync(
        _ buffer: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        op: ReduceOp = .sum,
        compression: WireCompression = .none,
        stream: StreamID = .default
    ) -> CollectiveRequest {
        // The buffer crosses into the work queue by the same contract the
        // blocking calls use: exactly one thread touches it at a time, and the
        // caller has undertaken not to be that thread until `wait` returns.
        let borrowed = UncheckedBuffer(buffer)
        return enqueue {
            try self.allReduceSync(borrowed.value, count: count, dataType: dataType,
                                   op: op, compression: compression, stream: stream)
        }
    }

    /// Puts `body` on the communicator's work queue and hands back its handle.
    ///
    /// The submission itself is synchronous, so the queue receives operations in
    /// the order the caller issued them. Dispatching through a detached task
    /// instead would let two issues race into the queue in either order, and two
    /// ranks that disagreed about that order would hang.
    func enqueue(_ body: @escaping @Sendable () throws -> Void) -> CollectiveRequest {
        let request = CollectiveRequest()
        workQueue.async {
            do {
                try body()
                request.complete(nil)
            } catch {
                request.complete(error)
            }
        }
        return request
    }
}
