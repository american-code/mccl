import Foundation
import MCCL

// The C ABI declared in include/mccl.h. Every exported symbol is thin: validate
// arguments, map the C enums onto Swift types, run the async collective to
// completion, and turn any thrown error into a flat result code.
//
// Nothing here decides anything. If a rule needs stating — which compression
// applies to which collective, which reduction top-k admits — it is stated once
// in MCCL and surfaces here as an error code.

// MARK: - Result codes

/// Mirrors `mcclResult_t`. `MCCLError` was written flat for exactly this.
enum ResultCode: Int32 {
    case success = 0
    case notImplemented = 1
    case invalidArgument = 2
    case rankOutOfRange = 3
    case noFabric = 4
    case bufferTooSmall = 5
    case connectionClosed = 6
    case protocolViolation = 7
    case socketFailure = 8
    case timedOut = 9
    case unsupportedCompression = 10
    case topologyInvalid = 11
    case internalError = 12

    init(_ error: MCCLError) {
        switch error {
        case .notImplemented: self = .notImplemented
        case .invalidArgument: self = .invalidArgument
        case .rankOutOfRange: self = .rankOutOfRange
        case .noFabric: self = .noFabric
        case .bufferTooSmall: self = .bufferTooSmall
        case .connectionClosed: self = .connectionClosed
        case .protocolViolation: self = .protocolViolation
        case .socketFailure: self = .socketFailure
        case .timedOut: self = .timedOut
        case .unsupportedCompression: self = .unsupportedCompression
        case .topologyInvalid: self = .topologyInvalid
        }
    }

    var name: String {
        switch self {
        case .success: return "mcclSuccess"
        case .notImplemented: return "mcclNotImplemented"
        case .invalidArgument: return "mcclInvalidArgument"
        case .rankOutOfRange: return "mcclRankOutOfRange"
        case .noFabric: return "mcclNoFabric"
        case .bufferTooSmall: return "mcclBufferTooSmall"
        case .connectionClosed: return "mcclConnectionClosed"
        case .protocolViolation: return "mcclProtocolViolation"
        case .socketFailure: return "mcclSocketFailure"
        case .timedOut: return "mcclTimedOut"
        case .unsupportedCompression: return "mcclUnsupportedCompression"
        case .topologyInvalid: return "mcclTopologyInvalid"
        case .internalError: return "mcclInternalError"
        }
    }
}

// MARK: - Error reporting

/// Holds the last error text for one communicator (or, at `global`, for calls
/// made before a communicator existed) in memory the caller can keep reading
/// until the next failure.
final class ErrorSlot: @unchecked Sendable {
    static let global = ErrorSlot()

    private let lock = NSLock()
    private var storage: UnsafeMutablePointer<CChar>?

    func record(_ message: String) {
        let copy = strdup(message)
        lock.lock()
        let previous = storage
        storage = copy
        lock.unlock()
        // Freed only after the swap: a concurrent reader holding the old pointer
        // has already returned it, and the contract is "valid until the next
        // failing call".
        free(previous)
    }

    var pointer: UnsafePointer<CChar> {
        lock.lock(); defer { lock.unlock() }
        if let storage { return UnsafePointer(storage) }
        return ErrorSlot.empty
    }

    private static let empty: UnsafePointer<CChar> = {
        let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: 1)
        buffer.pointee = 0
        return UnsafePointer(buffer)
    }()

    deinit { free(storage) }
}

/// Runs `body`, converting a throw into a result code and stashing the text.
func guarded(_ slot: ErrorSlot = .global, _ body: () throws -> Void) -> Int32 {
    do {
        try body()
        return ResultCode.success.rawValue
    } catch let error as MCCLError {
        slot.record(error.description)
        return ResultCode(error).rawValue
    } catch {
        slot.record("mccl: \(error)")
        return ResultCode.internalError.rawValue
    }
}

// MARK: - Blocking bridge

/// Runs an async collective to completion on the calling thread.
///
/// mccl's Swift surface is async because a collective is I/O; the C surface is
/// synchronous because that is what NCCL callers expect and what a runtime
/// without a device queue can honestly promise. The collective itself already
/// runs on the communicator's own dispatch queue, so this only parks the
/// caller — it never blocks a Swift concurrency worker that has work to do.
///
/// Do not call from inside a Swift `async` function; use the Swift API there.
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

/// Carries a caller-owned raw buffer across the blocking bridge.
///
/// `UnsafeRawBufferPointer` is not `Sendable`, and rightly so. Here the crossing
/// is safe by construction: the C entry point blocks until the collective has
/// finished, so exactly one thread touches the memory at a time and the C caller
/// cannot free it underneath us. That is the whole reason the shim is
/// synchronous.
struct Borrowed<Pointer>: @unchecked Sendable {
    let value: Pointer
    init(_ value: Pointer) { self.value = value }
}

final class ThrowBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Error?
    var error: Error? {
        get { lock.lock(); defer { lock.unlock() }; return storage }
        set { lock.lock(); storage = newValue; lock.unlock() }
    }
}

// MARK: - Handles

/// What `mcclComm_t` actually points at.
final class CommHandle {
    let communicator: Communicator
    let errors = ErrorSlot()

    init(_ communicator: Communicator) { self.communicator = communicator }

    static func from(_ pointer: UnsafeMutableRawPointer?) throws -> CommHandle {
        guard let pointer else { throw MCCLError.invalidArgument("comm is NULL") }
        return Unmanaged<CommHandle>.fromOpaque(pointer).takeUnretainedValue()
    }
}

// MARK: - Enum translation

func dataType(_ raw: Int32) throws -> DataType {
    switch raw {
    case 0: return .float32
    case 1: return .float16
    case 2: return .bfloat16
    case 3: return .int32
    case 4: return .int8
    default: throw MCCLError.invalidArgument("unknown mcclDataType_t \(raw)")
    }
}

func reduceOp(_ raw: Int32) throws -> ReduceOp {
    switch raw {
    case 0: return .sum
    case 1: return .prod
    case 2: return .min
    case 3: return .max
    case 4: return .avg
    default: throw MCCLError.invalidArgument("unknown mcclRedOp_t \(raw)")
    }
}

func compression(_ raw: Int32, blockSize: Int32, fraction: Double) throws -> WireCompression {
    switch raw {
    case 0: return .none
    case 1: return .downcast
    case 2: return .int8Blockwise(blockSize: blockSize > 0 ? Int(blockSize) : 256)
    case 3: return .topK(fraction: fraction)
    default: throw MCCLError.invalidArgument("unknown mcclCompression_t \(raw)")
    }
}

func streamID(_ raw: UnsafeMutableRawPointer?) -> StreamID {
    StreamID(UInt32(truncatingIfNeeded: UInt(bitPattern: raw)))
}

/// Validates a buffer pair and reports the in-place destination.
///
/// NCCL's convention: identical pointers mean in place. Different pointers mean
/// the caller's input is read-only, so copy it across before reducing.
func destination(
    send: UnsafeRawPointer?, recv: UnsafeMutableRawPointer?, byteCount: Int, label: String
) throws -> UnsafeMutableRawBufferPointer {
    guard byteCount >= 0 else { throw MCCLError.invalidArgument("\(label): negative element count") }
    guard let recv else { throw MCCLError.invalidArgument("\(label): recvbuff is NULL") }
    guard byteCount > 0 else { return UnsafeMutableRawBufferPointer(start: recv, count: 0) }
    guard let send else { throw MCCLError.invalidArgument("\(label): sendbuff is NULL") }
    if send != UnsafeRawPointer(recv) { recv.copyMemory(from: send, byteCount: byteCount) }
    return UnsafeMutableRawBufferPointer(start: recv, count: byteCount)
}

func copyCString(_ value: String, into buffer: UnsafeMutablePointer<CChar>?, size: Int) throws {
    guard let buffer else { throw MCCLError.invalidArgument("output buffer is NULL") }
    let bytes = Array(value.utf8)
    guard size > bytes.count else {
        throw MCCLError.bufferTooSmall(need: bytes.count + 1, have: size)
    }
    for (index, byte) in bytes.enumerated() { buffer[index] = CChar(bitPattern: byte) }
    buffer[bytes.count] = 0
}

// MARK: - Exports: diagnostics

@_cdecl("mcclGetVersion")
public func mcclGetVersion(_ version: UnsafeMutablePointer<Int32>?) -> Int32 {
    guarded {
        guard let version else { throw MCCLError.invalidArgument("version is NULL") }
        version.pointee = Int32(MCCLShimVersion.code)
    }
}

@_cdecl("mcclGetErrorString")
public func mcclGetErrorString(_ result: Int32) -> UnsafePointer<CChar> {
    let code = ResultCode(rawValue: result) ?? .internalError
    return ResultStrings.pointer(for: code, requested: result)
}

@_cdecl("mcclGetLastError")
public func mcclGetLastError(_ comm: UnsafeMutableRawPointer?) -> UnsafePointer<CChar> {
    guard let comm else { return ErrorSlot.global.pointer }
    return Unmanaged<CommHandle>.fromOpaque(comm).takeUnretainedValue().errors.pointer
}

enum MCCLShimVersion {
    static let major = 0, minor = 5, patch = 0
    static var code: Int { major * 10000 + minor * 100 + patch }
}

/// Interned C strings for the result names. `mcclGetErrorString` must return a
/// pointer that outlives the call, and Swift strings do not.
enum ResultStrings {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var interned: [Int32: UnsafePointer<CChar>] = [:]

    static func pointer(for code: ResultCode, requested: Int32) -> UnsafePointer<CChar> {
        let text = ResultCode(rawValue: requested) == nil
            ? "mcclUnknownResult(\(requested))" : code.name
        lock.lock(); defer { lock.unlock() }
        if let existing = interned[requested] { return existing }
        let pointer = UnsafePointer(strdup(text)!)
        interned[requested] = pointer
        return pointer
    }
}

// MARK: - Exports: unique ids

@_cdecl("mcclGetUniqueId")
public func mcclGetUniqueId(_ out: UnsafeMutablePointer<CChar>?) -> Int32 {
    guarded {
        let id = try Rendezvous.createUniqueID()
        try copyCString(id.text, into: out, size: UniqueIDBytes)
    }
}

@_cdecl("mcclUniqueIdDiscard")
public func mcclUniqueIdDiscard(_ raw: UnsafePointer<CChar>?) -> Int32 {
    guarded { Rendezvous.discard(try parseUniqueID(raw)) }
}

@_cdecl("mcclUniqueIdToString")
public func mcclUniqueIdToString(
    _ raw: UnsafePointer<CChar>?, _ buffer: UnsafeMutablePointer<CChar>?, _ size: Int
) -> Int32 {
    guarded { try copyCString(try parseUniqueID(raw).text, into: buffer, size: size) }
}

@_cdecl("mcclUniqueIdFromString")
public func mcclUniqueIdFromString(
    _ text: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<CChar>?
) -> Int32 {
    guarded {
        guard let text else { throw MCCLError.invalidArgument("text is NULL") }
        let value = String(cString: text)
        guard let id = UniqueID(text: value) else {
            throw MCCLError.invalidArgument("'\(value)' is not an mccl unique id")
        }
        try copyCString(id.text, into: out, size: UniqueIDBytes)
    }
}

/// Matches MCCL_UNIQUE_ID_BYTES in mccl.h.
let UniqueIDBytes = 128

func parseUniqueID(_ raw: UnsafePointer<CChar>?) throws -> UniqueID {
    guard let raw else { throw MCCLError.invalidArgument("uniqueId is NULL") }
    // The payload is NUL-terminated text inside a fixed 128-byte struct.
    let bytes = UnsafeBufferPointer(start: raw, count: UniqueIDBytes)
    let length = bytes.firstIndex(of: 0) ?? UniqueIDBytes
    let text = String(decoding: bytes[0..<length].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    guard let id = UniqueID(text: text) else {
        throw MCCLError.invalidArgument("'\(text)' is not an mccl unique id")
    }
    return id
}

// MARK: - Exports: communicators

@_cdecl("mcclCommInitRankFromId")
public func mcclCommInitRankFromId(
    _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ nranks: Int32,
    _ raw: UnsafePointer<CChar>?,
    _ rank: Int32
) -> Int32 {
    guarded {
        guard let out else { throw MCCLError.invalidArgument("comm out-parameter is NULL") }
        guard nranks > 0 else { throw MCCLError.invalidArgument("nranks must be > 0, got \(nranks)") }
        guard rank >= 0, rank < nranks else {
            throw MCCLError.rankOutOfRange(rank: Int(rank), worldSize: Int(nranks))
        }
        let id = try parseUniqueID(raw)
        let communicator = try Communicator.join(
            uniqueID: id, rank: Int(rank), worldSize: Int(nranks))
        out.pointee = Unmanaged.passRetained(CommHandle(communicator)).toOpaque()
    }
}

@_cdecl("mcclCommDestroy")
public func mcclCommDestroy(_ comm: UnsafeMutableRawPointer?) -> Int32 {
    guarded {
        guard let comm else { return }
        let unmanaged = Unmanaged<CommHandle>.fromOpaque(comm)
        unmanaged.takeUnretainedValue().communicator.shutdown()
        unmanaged.release()
    }
}

@_cdecl("mcclCommCount")
public func mcclCommCount(_ comm: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Int32>?) -> Int32 {
    guarded {
        let handle = try CommHandle.from(comm)
        guard let out else { throw MCCLError.invalidArgument("count out-parameter is NULL") }
        out.pointee = Int32(handle.communicator.worldSize)
    }
}

@_cdecl("mcclCommUserRank")
public func mcclCommUserRank(_ comm: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Int32>?) -> Int32 {
    guarded {
        let handle = try CommHandle.from(comm)
        guard let out else { throw MCCLError.invalidArgument("rank out-parameter is NULL") }
        out.pointee = Int32(handle.communicator.rank)
    }
}

@_cdecl("mcclCommPlanDescription")
public func mcclCommPlanDescription(
    _ comm: UnsafeMutableRawPointer?, _ bytes: Int,
    _ buffer: UnsafeMutablePointer<CChar>?, _ size: Int
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        try copyCString(handle.communicator.plan(messageBytes: bytes).description,
                        into: buffer, size: size)
    }
}

// MARK: - Exports: collectives

@_cdecl("mcclAllReduce")
public func mcclAllReduce(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ op: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    mcclAllReduceCompressed(send, recv, count, datatype, op, 0, 0, 0, comm, stream)
}

@_cdecl("mcclAllReduceCompressed")
public func mcclAllReduceCompressed(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ op: Int32,
    _ codec: Int32, _ blockSize: Int32, _ fraction: Double,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        let buffer = try destination(send: send, recv: recv,
                                     byteCount: count * type.byteWidth, label: "mcclAllReduce")
        let reduction = try reduceOp(op)
        let scheme = try compression(codec, blockSize: blockSize, fraction: fraction)
        let id = streamID(stream)
        let borrowed = Borrowed(buffer)
        try runBlocking {
            try await handle.communicator.allReduce(
                borrowed.value, count: count, dataType: type, op: reduction,
                compression: scheme, stream: id)
        }
    }
}

// MARK: - Exports: non-blocking collectives

/// What `mcclRequest_t` points at: the in-flight collective plus the
/// communicator whose error slot its failure belongs in.
final class RequestHandle {
    let request: CollectiveRequest
    let errors: ErrorSlot

    init(_ request: CollectiveRequest, errors: ErrorSlot) {
        self.request = request
        self.errors = errors
    }

    static func from(_ pointer: UnsafeMutableRawPointer?) throws -> RequestHandle {
        guard let pointer else { throw MCCLError.invalidArgument("request is NULL") }
        return Unmanaged<RequestHandle>.fromOpaque(pointer).takeUnretainedValue()
    }
}

@_cdecl("mcclAllReduceAsync")
public func mcclAllReduceAsync(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ op: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?,
    _ request: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Int32 {
    mcclAllReduceCompressedAsync(send, recv, count, datatype, op, 0, 0, 0, comm, stream, request)
}

@_cdecl("mcclAllReduceCompressedAsync")
public func mcclAllReduceCompressedAsync(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ op: Int32,
    _ codec: Int32, _ blockSize: Int32, _ fraction: Double,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?,
    _ request: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        guard let request else {
            throw MCCLError.invalidArgument("mcclAllReduceAsync: request out-parameter is NULL")
        }
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        // The in-place copy happens here, synchronously, so `sendbuff` is
        // released the moment this call returns even though the collective is
        // still running against `recvbuff`.
        let buffer = try destination(send: send, recv: recv,
                                     byteCount: count * type.byteWidth, label: "mcclAllReduceAsync")
        let reduction = try reduceOp(op)
        let scheme = try compression(codec, blockSize: blockSize, fraction: fraction)
        let pending = handle.communicator.allReduceAsync(
            buffer, count: count, dataType: type, op: reduction,
            compression: scheme, stream: streamID(stream))
        request.pointee = Unmanaged.passRetained(
            RequestHandle(pending, errors: handle.errors)).toOpaque()
    }
}

@_cdecl("mcclRequestWait")
public func mcclRequestWait(_ raw: UnsafeMutableRawPointer?) -> Int32 {
    let handle = try? RequestHandle.from(raw)
    let code = guarded(handle?.errors ?? .global) {
        let handle = try RequestHandle.from(raw)
        try handle.request.wait()
    }
    // Freed whichever way it went: the contract is that the handle is invalid
    // on return, so there is nothing left to release it later.
    if let raw { Unmanaged<RequestHandle>.fromOpaque(raw).release() }
    return code
}

@_cdecl("mcclRequestTest")
public func mcclRequestTest(
    _ raw: UnsafeMutableRawPointer?, _ done: UnsafeMutablePointer<Int32>?
) -> Int32 {
    let handle = try? RequestHandle.from(raw)
    return guarded(handle?.errors ?? .global) {
        let handle = try RequestHandle.from(raw)
        guard let done else {
            throw MCCLError.invalidArgument("mcclRequestTest: done out-parameter is NULL")
        }
        done.pointee = handle.request.isFinished ? 1 : 0
    }
}

@_cdecl("mcclAllGather")
public func mcclAllGather(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ sendcount: Int,
    _ datatype: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        guard sendcount >= 0 else { throw MCCLError.invalidArgument("mcclAllGather: negative sendcount") }
        guard let recv else { throw MCCLError.invalidArgument("mcclAllGather: recvbuff is NULL") }
        let chunkBytes = sendcount * type.byteWidth
        let worldSize = handle.communicator.worldSize
        // NCCL allows in-place all-gather by pointing sendbuff at this rank's
        // own slot of recvbuff; honour that rather than rejecting it.
        let source: UnsafeRawPointer
        if let send { source = send }
        else if chunkBytes == 0 { source = UnsafeRawPointer(recv) }
        else { throw MCCLError.invalidArgument("mcclAllGather: sendbuff is NULL") }

        let destination = Borrowed(UnsafeMutableRawBufferPointer(start: recv, count: chunkBytes * worldSize))
        let input = Borrowed(UnsafeRawBufferPointer(start: source, count: chunkBytes))
        try runBlocking {
            try await handle.communicator.allGather(
                input.value, into: destination.value, count: sendcount, dataType: type)
        }
    }
}

@_cdecl("mcclBroadcast")
public func mcclBroadcast(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ root: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        let byteCount = count * type.byteWidth
        // Only the root has meaningful input; everyone else's recvbuff is filled
        // by the collective, so do not demand a sendbuff from them.
        let buffer: UnsafeMutableRawBufferPointer
        if handle.communicator.rank == Int(root) {
            buffer = try destination(send: send, recv: recv, byteCount: byteCount, label: "mcclBroadcast")
        } else {
            guard let recv else { throw MCCLError.invalidArgument("mcclBroadcast: recvbuff is NULL") }
            buffer = UnsafeMutableRawBufferPointer(start: recv, count: byteCount)
        }
        let borrowed = Borrowed(buffer)
        try runBlocking {
            try await handle.communicator.broadcast(
                borrowed.value, count: count, dataType: type, root: Int(root))
        }
    }
}

@_cdecl("mcclReduceScatter")
public func mcclReduceScatter(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ recvcount: Int,
    _ datatype: Int32, _ op: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        guard recvcount >= 0 else { throw MCCLError.invalidArgument("mcclReduceScatter: negative recvcount") }
        guard let send else { throw MCCLError.invalidArgument("mcclReduceScatter: sendbuff is NULL") }
        guard let recv else { throw MCCLError.invalidArgument("mcclReduceScatter: recvbuff is NULL") }
        let segmentBytes = recvcount * type.byteWidth
        let input = Borrowed(UnsafeRawBufferPointer(
            start: send, count: segmentBytes * handle.communicator.worldSize))
        let output = Borrowed(UnsafeMutableRawBufferPointer(start: recv, count: segmentBytes))
        let reduction = try reduceOp(op)
        try runBlocking {
            try await handle.communicator.reduceScatter(
                input.value, into: output.value, recvCount: recvcount,
                dataType: type, op: reduction)
        }
    }
}

@_cdecl("mcclReduce")
public func mcclReduce(
    _ send: UnsafeRawPointer?, _ recv: UnsafeMutableRawPointer?, _ count: Int,
    _ datatype: Int32, _ op: Int32, _ root: Int32,
    _ comm: UnsafeMutableRawPointer?, _ stream: UnsafeMutableRawPointer?
) -> Int32 {
    let handle = try? CommHandle.from(comm)
    return guarded(handle?.errors ?? .global) {
        let handle = try CommHandle.from(comm)
        let type = try dataType(datatype)
        let buffer = try destination(send: send, recv: recv,
                                     byteCount: count * type.byteWidth, label: "mcclReduce")
        let reduction = try reduceOp(op)
        let borrowed = Borrowed(buffer)
        try runBlocking {
            try await handle.communicator.reduce(
                borrowed.value, count: count, dataType: type, op: reduction, root: Int(root))
        }
    }
}
