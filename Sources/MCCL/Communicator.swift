import Foundation

/// A handle to a group of ranks participating in collective operations,
/// analogous to `ncclComm_t`.
///
/// Design intent: the communicator owns a measured `Topology` and picks the
/// collective algorithm (ring / tree / hierarchical) per-operation based on
/// message size and measured link bandwidth — not a hardcoded ring.
public final class Communicator: @unchecked Sendable {
    public let rank: Int
    public let worldSize: Int
    public let topology: Topology

    /// Rank-addressed transport. `nil` for a degenerate single-rank
    /// communicator, where every collective is a no-op.
    public let fabric: Fabric?

    /// Forces a specific algorithm instead of consulting the planner.
    /// Intended for benchmarking and for tests that need to pin a code path.
    public var planOverride: CollectivePlan?

    /// Every collective runs here, one at a time. Collectives block on socket
    /// I/O; keeping them off the Swift concurrency cooperative pool means a
    /// rank waiting on a peer never starves another rank's progress.
    private let workQueue: DispatchQueue

    /// Error-feedback state for `.topK`, one residual per `StreamID`. Nothing
    /// else in mccl is stateful across calls; this is what makes sparsified
    /// all-reduce unbiased over a training run instead of merely lossy.
    let residuals = ResidualStore()

    public init(rank: Int, worldSize: Int, topology: Topology) {
        self.rank = rank
        self.worldSize = worldSize
        self.topology = topology
        self.fabric = nil
        self.workQueue = DispatchQueue(label: "mccl.comm.\(rank)")
    }

    public init(rank: Int, worldSize: Int, topology: Topology, fabric: Fabric) {
        precondition(fabric.rank == rank, "fabric rank \(fabric.rank) != communicator rank \(rank)")
        precondition(fabric.worldSize == worldSize, "fabric world size mismatch")
        self.rank = rank
        self.worldSize = worldSize
        self.topology = topology
        self.fabric = fabric
        self.workQueue = DispatchQueue(label: "mccl.comm.\(rank)")
    }

    /// Releases the transport. Safe to call more than once.
    public func shutdown() { fabric?.shutdown() }

    // MARK: - Bring-up

    /// Joins a world whose peers are reachable at `addresses`. `listener` must
    /// already be bound to `addresses[rank]`.
    public static func bootstrap(
        rank: Int,
        worldSize: Int,
        addresses: [PeerAddress],
        listener: Listener,
        transport: Transport = TCPTransport(),
        topology: Topology? = nil,
        timeout: TimeInterval = 30
    ) throws -> Communicator {
        let fabric = try MeshFabric.bootstrap(
            rank: rank, worldSize: worldSize, addresses: addresses,
            listener: listener, transport: transport, timeout: timeout)
        return Communicator(
            rank: rank, worldSize: worldSize,
            topology: topology ?? Topology.uniform(nodeCount: worldSize),
            fabric: fabric)
    }

    /// Brings up an entire world inside this process over real loopback TCP —
    /// one ephemeral port and one socket pair per rank.
    public static func tcpGroup(
        worldSize: Int,
        host: String = "127.0.0.1",
        topology: Topology? = nil
    ) throws -> [Communicator] {
        let fabrics = try MeshFabric.makeGroup(worldSize: worldSize, transport: TCPTransport(), host: host)
        let topo = topology ?? Topology.uniform(nodeCount: worldSize, kind: .loopback)
        return fabrics.map { Communicator(rank: $0.rank, worldSize: worldSize, topology: topo, fabric: $0) }
    }

    /// Same, but over the in-process `LoopbackTransport` — no sockets at all.
    public static func loopbackGroup(
        worldSize: Int,
        topology: Topology? = nil
    ) throws -> [Communicator] {
        let fabrics = try MeshFabric.makeGroup(
            worldSize: worldSize, transport: LoopbackTransport(), host: "loopback")
        let topo = topology ?? Topology.uniform(nodeCount: worldSize, kind: .loopback)
        return fabrics.map { Communicator(rank: $0.rank, worldSize: worldSize, topology: topo, fabric: $0) }
    }

    // MARK: - Public collective API

    /// All-reduce over a raw buffer. `compression` applies in-flight only —
    /// buffers are compressed for the wire (TB5/Ethernet is the bottleneck,
    /// not compute) and decompressed before the reduction on each hop.
    ///
    /// `stream` only matters for `.topK`, which keeps an error-feedback residual
    /// per stream: reduce two different tensors on one communicator and they
    /// must not share a residual. It mirrors `mcclStream_t` in the C shim.
    public func allReduce(
        _ buffer: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        op: ReduceOp = .sum,
        compression: WireCompression = .none,
        stream: StreamID = .default
    ) async throws {
        try await run {
            try self.allReduceSync(buffer, count: count, dataType: dataType,
                                   op: op, compression: compression, stream: stream)
        }
    }

    public func allGather(
        _ send: UnsafeRawBufferPointer,
        into recv: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType
    ) async throws {
        try await run { try self.allGatherSync(send, into: recv, count: count, dataType: dataType, compression: .none) }
    }

    /// All-gather with in-flight compression.
    public func allGather(
        _ send: UnsafeRawBufferPointer,
        into recv: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        compression: WireCompression
    ) async throws {
        try await run { try self.allGatherSync(send, into: recv, count: count, dataType: dataType, compression: compression) }
    }

    public func broadcast(
        _ buffer: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        root: Int
    ) async throws {
        try await run { try self.broadcastSync(buffer, count: count, dataType: dataType, root: root, compression: .none) }
    }

    public func broadcast(
        _ buffer: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        root: Int,
        compression: WireCompression
    ) async throws {
        try await run { try self.broadcastSync(buffer, count: count, dataType: dataType, root: root, compression: compression) }
    }

    public func reduceScatter(
        _ send: UnsafeRawBufferPointer,
        into recv: UnsafeMutableRawBufferPointer,
        recvCount: Int,
        dataType: DataType,
        op: ReduceOp = .sum
    ) async throws {
        try await run {
            try self.reduceScatterSync(send, into: recv, recvCount: recvCount,
                                       dataType: dataType, op: op, compression: .none)
        }
    }

    public func reduceScatter(
        _ send: UnsafeRawBufferPointer,
        into recv: UnsafeMutableRawBufferPointer,
        recvCount: Int,
        dataType: DataType,
        op: ReduceOp,
        compression: WireCompression
    ) async throws {
        try await run {
            try self.reduceScatterSync(send, into: recv, recvCount: recvCount,
                                       dataType: dataType, op: op, compression: compression)
        }
    }

    /// Reduce onto a single root — the second half of the tree plan, exposed
    /// because MLX-style pipelines want it directly.
    public func reduce(
        _ buffer: UnsafeMutableRawBufferPointer,
        count: Int,
        dataType: DataType,
        op: ReduceOp = .sum,
        root: Int,
        compression: WireCompression = .none
    ) async throws {
        try await run {
            try self.reduceSync(buffer, count: count, dataType: dataType, op: op, root: root, compression: compression)
        }
    }

    // MARK: - Error-feedback residuals

    /// The un-sent remainder `.topK` is currently holding for `stream`, in
    /// element order, or nil if nothing has been sparsified on it yet.
    ///
    /// Exposed because it is the thing that makes compressed training
    /// reproducible: a checkpoint that saves the model without the residual has
    /// silently thrown away part of the gradient.
    public func topKResidual(for stream: StreamID = .default) -> [Float]? {
        residuals.snapshot(stream)
    }

    /// Drops one stream's residual, or every stream's when `stream` is nil.
    public func clearTopKResiduals(stream: StreamID? = nil) {
        residuals.reset(stream)
    }

    // MARK: - Planning

    /// The algorithm this communicator would use for a message of the given size.
    public func plan(messageBytes: Int) -> CollectivePlan {
        if let planOverride { return sanitize(planOverride) }
        return sanitize(TopologyPlanner.plan(for: topology, messageBytes: messageBytes))
    }

    /// A plan is only executable if it names exactly the ranks in this world.
    /// A topology loaded from disk may describe a different (or stale) cluster,
    /// so anything inconsistent degrades to the plain ring rather than failing.
    func sanitize(_ plan: CollectivePlan) -> CollectivePlan {
        let all = Set(0..<worldSize)
        switch plan {
        case .ring(let order):
            let filtered = order.filter { all.contains($0) }
            guard Set(filtered).count == worldSize else { return .ring(order: Array(0..<worldSize)) }
            return .ring(order: filtered)
        case .tree(let root, let children):
            guard all.contains(root) else { return .ring(order: Array(0..<worldSize)) }
            var seen: Set<Int> = [root]
            for (parent, kids) in children {
                guard all.contains(parent) else { return .ring(order: Array(0..<worldSize)) }
                for kid in kids {
                    guard all.contains(kid), !seen.contains(kid) else {
                        return .ring(order: Array(0..<worldSize))
                    }
                    seen.insert(kid)
                }
            }
            guard seen.count == worldSize else { return .ring(order: Array(0..<worldSize)) }
            return .tree(root: root, children: children)
        case .hierarchical(let islands, let interIslandRoot):
            let flat = islands.flatMap { $0 }
            guard Set(flat).count == worldSize, Set(flat) == all,
                  all.contains(interIslandRoot), islands.allSatisfy({ !$0.isEmpty }) else {
                return .ring(order: Array(0..<worldSize))
            }
            return .hierarchical(islands: islands, interIslandRoot: interIslandRoot)
        }
    }

    // MARK: - Async bridge

    private func run(_ body: @escaping () throws -> Void) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let job = IOJob(body)
            workQueue.async {
                job.run()
                if let error = job.error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            }
        }
    }
}

public enum DataType: Sendable {
    case float32, float16, bfloat16, int32, int8
}

public enum ReduceOp: Sendable {
    case sum, prod, min, max, avg
}

/// In-flight wire compression schemes. Applied per-hop on the interconnect;
/// never changes the dtype the caller sees.
public enum WireCompression: Sendable {
    case none
    /// Cast fp32 payloads to fp16/bf16 on the wire.
    case downcast
    /// Blockwise int8 quantization with per-block scales.
    case int8Blockwise(blockSize: Int = 256)
    /// Top-k gradient sparsification with error feedback (training-only).
    case topK(fraction: Double)
}
