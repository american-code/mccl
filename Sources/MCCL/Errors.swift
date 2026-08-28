import Foundation

/// Every error mccl can raise. Deliberately a flat, C-shim-friendly enum: the
/// planned `mcclResult_t` maps one-to-one onto these cases.
public enum MCCLError: Error, CustomStringConvertible, Equatable {
    /// A feature that is declared in the API but not yet implemented.
    /// Never a `fatalError` — callers linking mccl must be able to recover.
    case notImplemented(String)
    case invalidArgument(String)
    case rankOutOfRange(rank: Int, worldSize: Int)
    /// The communicator has no transport fabric, so it cannot talk to peers.
    case noFabric
    case bufferTooSmall(need: Int, have: Int)
    case connectionClosed
    case protocolViolation(String)
    case socketFailure(String, errno: Int32)
    case timedOut(String)
    /// The requested `WireCompression` scheme cannot be applied here.
    case unsupportedCompression(String)
    case topologyInvalid(String)
    /// A verbs call failed. `code` is the shim's own code (see `crdma.h`) or a
    /// negated errno from the library.
    case rdmaFailure(String, code: Int32)
    /// A work completion came back with a non-success status.
    case rdmaCompletionFailed(status: UInt32, description: String)
    /// A slot arrived out of order on an unreliable-connection queue pair.
    /// Loss is surfaced rather than papered over — see `RDMASequenceChecker`.
    case rdmaSequenceGap(expected: UInt32, received: UInt32)
    /// RDMA over Thunderbolt cannot be used on this machine, with the reason.
    case rdmaUnavailable(String)

    public var description: String {
        switch self {
        case .notImplemented(let what):
            return "mccl: not implemented: \(what)"
        case .invalidArgument(let what):
            return "mccl: invalid argument: \(what)"
        case .rankOutOfRange(let rank, let worldSize):
            return "mccl: rank \(rank) out of range for world size \(worldSize)"
        case .noFabric:
            return "mccl: communicator has no transport fabric (build it with Communicator.bootstrap / .loopbackGroup / .tcpGroup)"
        case .bufferTooSmall(let need, let have):
            return "mccl: buffer too small: need \(need) bytes, have \(have)"
        case .connectionClosed:
            return "mccl: peer closed the connection"
        case .protocolViolation(let what):
            return "mccl: wire protocol violation: \(what)"
        case .socketFailure(let what, let e):
            return "mccl: socket failure: \(what): \(String(cString: strerror(e))) (errno \(e))"
        case .timedOut(let what):
            return "mccl: timed out: \(what)"
        case .unsupportedCompression(let what):
            return "mccl: unsupported wire compression: \(what)"
        case .topologyInvalid(let what):
            return "mccl: invalid topology: \(what)"
        case .rdmaFailure(let what, let code):
            return "mccl: RDMA failure: \(what) (code \(code))"
        case .rdmaCompletionFailed(let status, let text):
            return "mccl: RDMA work completion failed: \(text) (status \(status))"
        case .rdmaSequenceGap(let expected, let received):
            return """
                mccl: RDMA slot sequence gap: expected \(expected), received \(received). \
                A Thunderbolt queue pair is an unreliable connection — the hardware does not \
                acknowledge or retransmit — so a missing slot is data loss that mccl refuses to \
                paper over. Re-run over the TCP transport if this recurs.
                """
        case .rdmaUnavailable(let why):
            return "mccl: RDMA over Thunderbolt is unavailable: \(why)"
        }
    }
}
