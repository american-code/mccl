import XCTest
@testable import MCCL

// Shared helpers: dtype-agnostic buffer filling/reading and an N-rank harness.

enum Buf {
    static func allocate(count: Int, dataType: DataType) -> UnsafeMutableRawBufferPointer {
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: max(16, count * dataType.byteWidth), alignment: 16)
        buffer.initializeMemory(as: UInt8.self, repeating: 0)
        return buffer
    }

    static func fill(
        _ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType,
        _ value: (Int) -> Float
    ) {
        for i in 0..<count {
            ElementIO.storeFloat(value(i), into: buffer.baseAddress!, byteOffset: i * dataType.byteWidth, dataType)
        }
    }

    static func read(_ buffer: UnsafeRawBufferPointer, count: Int, dataType: DataType) -> [Float] {
        (0..<count).map {
            ElementIO.loadFloat(buffer.baseAddress!, byteOffset: $0 * dataType.byteWidth, dataType)
        }
    }

    static func read(_ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType) -> [Float] {
        read(UnsafeRawBufferPointer(buffer), count: count, dataType: dataType)
    }
}

enum Ranks {
    /// Runs `body` on every rank of an N-rank world concurrently.
    static func run(
        worldSize: Int,
        topology: Topology? = nil,
        transport: @autoclosure () -> Transport = TCPTransport(),
        planOverride: CollectivePlan? = nil,
        body: @escaping @Sendable (Communicator) async throws -> Void
    ) async throws {
        let fabrics = try MeshFabric.makeGroup(worldSize: worldSize, transport: transport(), host: "127.0.0.1")
        let topo = topology ?? Topology.uniform(nodeCount: worldSize)
        let comms = fabrics.map {
            Communicator(rank: $0.rank, worldSize: worldSize, topology: topo, fabric: $0)
        }
        for comm in comms { comm.planOverride = planOverride }
        defer { comms.forEach { $0.shutdown() } }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for comm in comms {
                group.addTask { try await body(comm) }
            }
            try await group.waitForAll()
        }
    }

    /// Same, but over the in-process loopback transport (no sockets).
    static func runLoopback(
        worldSize: Int,
        topology: Topology? = nil,
        body: @escaping @Sendable (Communicator) async throws -> Void
    ) async throws {
        try await run(worldSize: worldSize, topology: topology,
                      transport: LoopbackTransport(), body: body)
    }
}

// MARK: - Byte counting

/// Counts every byte handed to the transport. Wraps a real transport rather
/// than faking one, so what it counts is exactly what the collectives sent,
/// framing headers included.
final class CountingTransport: Transport, @unchecked Sendable {
    let name = "counting"
    private let inner: Transport
    private let lock = NSLock()
    private var sent = 0
    private var received = 0

    init(wrapping inner: Transport = LoopbackTransport()) { self.inner = inner }

    var bytesSent: Int { lock.lock(); defer { lock.unlock() }; return sent }
    var bytesReceived: Int { lock.lock(); defer { lock.unlock() }; return received }

    func reset() { lock.lock(); sent = 0; received = 0; lock.unlock() }

    fileprivate func addSent(_ n: Int) { lock.lock(); sent += n; lock.unlock() }
    fileprivate func addReceived(_ n: Int) { lock.lock(); received += n; lock.unlock() }

    func listen(host: String, port: Int) throws -> Listener {
        CountingListener(inner: try inner.listen(host: host, port: port), transport: self)
    }

    func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel {
        CountingChannel(inner: try inner.connect(to: address, timeout: timeout), transport: self)
    }
}

private final class CountingListener: Listener {
    private let inner: Listener
    private let transport: CountingTransport

    init(inner: Listener, transport: CountingTransport) {
        self.inner = inner
        self.transport = transport
    }

    var address: PeerAddress { inner.address }
    func accept(timeout: TimeInterval?) throws -> Channel {
        CountingChannel(inner: try inner.accept(timeout: timeout), transport: transport)
    }
    func close() { inner.close() }
}

private final class CountingChannel: Channel {
    private let inner: Channel
    private let transport: CountingTransport

    init(inner: Channel, transport: CountingTransport) {
        self.inner = inner
        self.transport = transport
    }

    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws {
        transport.addSent(buffer.count)
        try inner.sendBytes(buffer)
    }
    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws {
        try inner.receiveBytes(into: buffer)
        transport.addReceived(buffer.count)
    }
    func close() { inner.close() }
    var sendQueue: DispatchQueue { inner.sendQueue }
    var receiveQueue: DispatchQueue { inner.receiveQueue }
    var peerDescription: String { inner.peerDescription }
}

/// Thread-safe collection point for per-rank values produced inside a
/// concurrent `Ranks.run` body.
final class Collector<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int: T] = [:]

    func set(_ rank: Int, _ value: T) { lock.lock(); values[rank] = value; lock.unlock() }
    func get(_ rank: Int) -> T? { lock.lock(); defer { lock.unlock() }; return values[rank] }
    var all: [Int: T] { lock.lock(); defer { lock.unlock() }; return values }
}

/// Synthetic topologies for planner tests. Bandwidths are realistic Apple
/// Silicon cluster numbers so the derived thresholds land in plausible places.
enum Fabrics {
    /// Every node on the same TB5 fabric.
    static func uniformFast(nodeCount: Int) -> Topology {
        Topology.uniform(nodeCount: nodeCount, kind: .thunderbolt5,
                         bandwidth: 5.5e9, latency: 20e-6, chip: "M4 Max")
    }

    /// Every node on the same 10GbE fabric.
    static func uniformSlow(nodeCount: Int) -> Topology {
        Topology.uniform(nodeCount: nodeCount, kind: .ethernet,
                         bandwidth: 1.15e9, latency: 200e-6, chip: "M4 Max")
    }

    /// Two TB5 pairs bridged by a single 10GbE hop: {0,1} and {2,3}.
    static func twoIslandsBridged() -> Topology {
        let nodes = (0..<4).map {
            Topology.Node(id: $0, hostname: "node\($0)", chip: "M4 Max", unifiedMemoryBytes: 128 << 30)
        }
        let links = [
            Topology.Link(from: 0, to: 1, kind: .thunderbolt5, measuredBandwidth: 5.5e9, measuredLatency: 20e-6),
            Topology.Link(from: 2, to: 3, kind: .thunderbolt5, measuredBandwidth: 5.4e9, measuredLatency: 21e-6),
            Topology.Link(from: 1, to: 2, kind: .ethernet, measuredBandwidth: 1.15e9, measuredLatency: 210e-6),
            Topology.Link(from: 0, to: 2, kind: .ethernet, measuredBandwidth: 1.10e9, measuredLatency: 230e-6),
            Topology.Link(from: 1, to: 3, kind: .ethernet, measuredBandwidth: 1.12e9, measuredLatency: 225e-6),
            Topology.Link(from: 0, to: 3, kind: .ethernet, measuredBandwidth: 1.08e9, measuredLatency: 235e-6),
        ]
        return Topology(nodes: nodes, links: links)
    }

    /// The lab's mixed fabric, to scale: two Mac Studios on a Thunderbolt cable
    /// plus a laptop that can only reach either of them over Wi-Fi. One island
    /// of two and one singleton — the smallest topology that is genuinely mixed,
    /// and the one a two-machine lab plus a laptop actually produces.
    ///
    /// The bandwidths are the measured ones from
    /// docs/ARCHITECTURE.md §Measured: mixed fabric.
    static func islandPlusBridge() -> Topology {
        let nodes = (0..<3).map {
            Topology.Node(id: $0, hostname: "node\($0)", chip: "M1 Max", unifiedMemoryBytes: 64 << 30)
        }
        let links = [
            Topology.Link(from: 0, to: 1, kind: .thunderbolt4, measuredBandwidth: 1.06e9, measuredLatency: 90e-6),
            Topology.Link(from: 0, to: 2, kind: .wifi, measuredBandwidth: 6.1e7, measuredLatency: 1.9e-3),
            Topology.Link(from: 1, to: 2, kind: .wifi, measuredBandwidth: 5.8e7, measuredLatency: 2.1e-3),
        ]
        return Topology(nodes: nodes, links: links)
    }
}
