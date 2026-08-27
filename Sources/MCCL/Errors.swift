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
        }
    }
}
