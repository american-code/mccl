import Foundation

/// Where a peer can be reached. Deliberately transport-agnostic: a Thunderbolt
/// transport can reuse it (TB peer-to-peer presents as an IP link), and a
/// future non-IP transport can reinterpret `host` as its own identifier.
public struct PeerAddress: Sendable, Hashable, Codable, CustomStringConvertible {
    public var host: String
    public var port: Int

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }

    /// Parses `host:port`, `[v6::addr]:port`, or a bare port.
    public init?(_ string: String) {
        if string.hasPrefix("[") , let close = string.firstIndex(of: "]") {
            let host = String(string[string.index(after: string.startIndex)..<close])
            let rest = string[string.index(after: close)...]
            guard rest.hasPrefix(":"), let port = Int(rest.dropFirst()) else { return nil }
            self.init(host: host, port: port)
            return
        }
        let parts = string.split(separator: ":")
        if parts.count == 2, let port = Int(parts[1]) {
            self.init(host: String(parts[0]), port: port)
        } else if parts.count == 1, let port = Int(parts[0]) {
            self.init(host: "127.0.0.1", port: port)
        } else {
            return nil
        }
    }

    public var description: String { host.contains(":") ? "[\(host)]:\(port)" : "\(host):\(port)" }

    public var isLoopback: Bool {
        host == "127.0.0.1" || host == "::1" || host == "localhost" || host == "loopback"
    }
}

/// A bidirectional, reliable, ordered byte channel to exactly one peer.
///
/// The methods block. mccl runs every collective on its own thread (see
/// `Communicator.run`) and every concurrent send/receive pair on the channel's
/// own dispatch queues, so blocking here never occupies a Swift concurrency
/// cooperative thread.
public protocol Channel: AnyObject {
    /// Writes the whole buffer. Blocks until it is handed to the transport.
    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws
    /// Fills the whole buffer. Blocks until `buffer.count` bytes have arrived.
    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws
    func close()

    /// Queue on which this channel's blocking sends are executed. Must be
    /// distinct from `receiveQueue` so a rank can send and receive at once —
    /// every ring step depends on that full-duplex overlap.
    var sendQueue: DispatchQueue { get }
    var receiveQueue: DispatchQueue { get }

    var peerDescription: String { get }
}

/// Accepts inbound channels at a bound address.
public protocol Listener: AnyObject {
    /// The address actually bound. With port 0 requested, this reports the
    /// kernel-assigned ephemeral port.
    var address: PeerAddress { get }
    func accept(timeout: TimeInterval?) throws -> Channel
    func close()
}

/// A way of moving bytes between nodes. `TCPTransport` and `LoopbackTransport`
/// implement it today; a Thunderbolt-specific transport (bulk DMA over a TB
/// bridge rather than TCP) slots in here without touching the collectives.
public protocol Transport: AnyObject {
    var name: String { get }
    func listen(host: String, port: Int) throws -> Listener
    func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel
}

extension Transport {
    public func listen() throws -> Listener { try listen(host: "127.0.0.1", port: 0) }
    public func connect(to address: PeerAddress) throws -> Channel {
        try connect(to: address, timeout: 10)
    }
}

// MARK: - Full-duplex helper

/// Runs two blocking I/O closures at the same time on the given queues and
/// waits for both. Every ring step is one of these: "send my chunk forward
/// while the previous rank's chunk arrives".
final class IOJob: @unchecked Sendable {
    private let work: () throws -> Void
    private let lock = NSLock()
    private var _error: Error?

    init(_ work: @escaping () throws -> Void) { self.work = work }

    func run() {
        do { try work() } catch {
            lock.lock(); if _error == nil { _error = error }; lock.unlock()
        }
    }

    var error: Error? {
        lock.lock(); defer { lock.unlock() }
        return _error
    }
}

func runConcurrently(
    _ a: @escaping () throws -> Void, on queueA: DispatchQueue,
    _ b: @escaping () throws -> Void, on queueB: DispatchQueue
) throws {
    let jobA = IOJob(a)
    let jobB = IOJob(b)
    let group = DispatchGroup()
    queueA.async(group: group) { jobA.run() }
    queueB.async(group: group) { jobB.run() }
    group.wait()
    if let e = jobA.error { throw e }
    if let e = jobB.error { throw e }
}
