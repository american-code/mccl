import Foundation
import MCCL

// The harness behind `mcclbench`. Lives in its own library target so the test
// suite can drive it directly instead of scraping a subprocess's stdout.

// MARK: - Dimensions

public enum BenchAlgorithm: String, CaseIterable, Sendable {
    case ring, tree, hierarchical

    /// The plan to pin on every rank. `nil` when the shape does not exist for
    /// this world — a hierarchical plan needs at least two islands of two.
    public func plan(worldSize: Int) -> CollectivePlan? {
        let order = Array(0..<worldSize)
        switch self {
        case .ring:
            return .ring(order: order)
        case .tree:
            return .tree(root: 0, children: TopologyPlanner.binomialTree(order: order, root: 0))
        case .hierarchical:
            guard worldSize >= 4 else { return nil }
            // Split down the middle: the shape of two TB-bridged pairs, which is
            // the fabric the hierarchical plan exists for.
            let half = worldSize / 2
            return .hierarchical(islands: [Array(0..<half), Array(half..<worldSize)],
                                 interIslandRoot: 0)
        }
    }

    /// Why this algorithm produces no rows for `worldSize`, or nil if it does.
    ///
    /// `plan(worldSize:)` returning nil used to make the sweep skip the
    /// algorithm silently, which reads as a bug in the harness rather than as a
    /// property of the world: an n=3 run asked for `hierarchical` and simply got
    /// a table with no hierarchical rows in it and no explanation. Dropping the
    /// rows is correct — three ranks on a uniform fabric have no islands — but
    /// saying so is part of being correct.
    public func inapplicabilityReason(worldSize: Int) -> String? {
        guard plan(worldSize: worldSize) == nil else { return nil }
        switch self {
        case .hierarchical:
            return "needs at least two islands of two ranks; a \(worldSize)-rank "
                + "uniform fabric has none"
        case .ring, .tree:
            return "no plan for a \(worldSize)-rank world"
        }
    }
}

public enum BenchCollective: String, CaseIterable, Sendable {
    case allReduce = "allreduce"
    case allGather = "allgather"
    case broadcast = "broadcast"
    case reduceScatter = "reducescatter"

    /// Bytes the caller's payload occupies, given a per-point message size.
    /// All-gather and reduce-scatter split the message across ranks so every
    /// row of the table moves the same number of bytes.
    func elementCount(messageBytes: Int, dataType: DataType, worldSize: Int) -> Int {
        let elements = messageBytes / dataType.byteWidth
        switch self {
        case .allReduce, .broadcast: return max(1, elements)
        case .allGather, .reduceScatter: return max(1, elements / worldSize)
        }
    }

    /// The `2(n-1)/n` style factor that turns payload throughput into the
    /// bandwidth actually crossing each link — NCCL's "bus bandwidth".
    func busFactor(worldSize n: Int) -> Double {
        guard n > 1 else { return 1 }
        switch self {
        case .allReduce: return 2 * Double(n - 1) / Double(n)
        case .allGather, .reduceScatter: return Double(n - 1) / Double(n)
        case .broadcast: return 1
        }
    }
}

public enum BenchCodec: Sendable, CustomStringConvertible {
    case none
    case downcast
    case int8Blockwise(blockSize: Int)
    case topK(fraction: Double)

    public var compression: WireCompression {
        switch self {
        case .none: return .none
        case .downcast: return .downcast
        case .int8Blockwise(let size): return .int8Blockwise(blockSize: size)
        case .topK(let fraction): return .topK(fraction: fraction)
        }
    }

    public var description: String {
        switch self {
        case .none: return "none"
        case .downcast: return "downcast"
        case .int8Blockwise(let size): return "int8/\(size)"
        case .topK(let fraction): return "topk/\(Self.trim(fraction))"
        }
    }

    private static func trim(_ value: Double) -> String {
        let text = String(format: "%.4f", value)
        return text.hasSuffix("0") ? String(text.reversed().drop(while: { $0 == "0" }).reversed()) : text
    }

    /// Codecs that apply to a collective. Top-k is an all-reduce-only scheme
    /// (see docs/ARCHITECTURE.md §Wire compression), so asking for it elsewhere
    /// would only produce a table full of errors.
    public func applies(to collective: BenchCollective, dataType: DataType) -> Bool {
        switch self {
        case .none: return true
        case .downcast: return dataType == .float32
        case .int8Blockwise: return dataType.isFloatingPoint
        case .topK: return collective == .allReduce && dataType.isFloatingPoint
        }
    }
}

public enum BenchTransport: String, CaseIterable, Sendable {
    /// Real loopback sockets — one ephemeral port per rank. The default,
    /// because it exercises the same code path a cluster would.
    case tcp
    /// In-process byte queues. Isolates codec and kernel cost from the socket.
    case loopback
}

// MARK: - Options

public struct BenchOptions: Sendable {
    public var worldSize = 4
    public var minBytes = 1 << 10
    public var maxBytes = 64 << 20
    /// Multiplier between successive message sizes.
    public var sizeStep = 4
    public var dataType: DataType = .float32
    public var op: ReduceOp = .sum
    public var collective: BenchCollective = .allReduce
    public var algorithms: [BenchAlgorithm] = BenchAlgorithm.allCases
    public var codecs: [BenchCodec] = [.none, .downcast, .int8Blockwise(blockSize: 256), .topK(fraction: 0.01)]
    public var transport: BenchTransport = .tcp
    public var warmup = 2
    public var minIterations = 3
    public var maxIterations = 50
    /// Bytes to move per measurement point. Keeps small sizes statistically
    /// meaningful without letting 64 MiB points dominate the wall clock.
    public var byteBudget = 128 << 20
    public var csv = false
    /// `--codec-bench`: measure the codec kernels on this machine instead of
    /// running a collective. The fabric half of the compression rule comes from
    /// the sweep; this is the machine half.
    public var codecBench = false
    /// Set by `--sizes`, and then the only sizes swept. Nil means the
    /// `minBytes`/`maxBytes`/`sizeStep` geometric sweep.
    public var explicitSizes: [Int]?
    /// The multi-process half, when `mcclbench` is one rank of a real world.
    /// Nil keeps the in-process behaviour, unchanged.
    public var distributed: DistributedOptions?

    public init() {}

    public var sizes: [Int] {
        if let explicitSizes, !explicitSizes.isEmpty { return explicitSizes }
        var result: [Int] = []
        var size = max(1, minBytes)
        while size <= maxBytes {
            result.append(size)
            size *= max(2, sizeStep)
        }
        if result.isEmpty { result = [max(1, minBytes)] }
        return result
    }

    func iterations(forMessageBytes bytes: Int) -> Int {
        guard bytes > 0 else { return maxIterations }
        return max(minIterations, min(maxIterations, byteBudget / bytes))
    }
}

// MARK: - Results

public struct BenchRow: Sendable {
    public let messageBytes: Int
    public let algorithm: BenchAlgorithm
    public let codec: String
    public let iterations: Int
    /// Mean wall time for one collective, seconds.
    public let seconds: Double
    /// Payload bytes / time — what the caller sees move.
    public let algorithmicBytesPerSecond: Double
    /// Bytes actually crossing a link / time — comparable to NCCL's busbw.
    public let busBytesPerSecond: Double
    /// Bytes this rank really handed to the transport per collective, when the
    /// run was instrumented. Nil otherwise.
    public let wireBytesPerRank: Int?
    /// Set instead of the timings when the point could not run.
    public let failure: String?
}

// MARK: - Runner

public enum BenchRunner {

    /// Sweeps every (size, algorithm, codec) point and returns one row each.
    ///
    /// One world is brought up for the whole sweep and the plan is pinned
    /// per point via `planOverride`, so the table compares algorithms rather
    /// than bootstrap costs.
    public static func run(
        _ options: BenchOptions,
        onRow: (@Sendable (BenchRow) -> Void)? = nil
    ) async throws -> [BenchRow] {
        guard options.worldSize > 1 else {
            throw MCCLError.invalidArgument("benchmark needs at least 2 ranks")
        }
        let transport: Transport = options.transport == .tcp ? TCPTransport() : LoopbackTransport()
        let fabrics = try MeshFabric.makeGroup(
            worldSize: options.worldSize, transport: transport,
            host: options.transport == .tcp ? "127.0.0.1" : "loopback")
        let topology = Topology.uniform(nodeCount: options.worldSize, kind: .loopback)
        let comms = fabrics.map {
            Communicator(rank: $0.rank, worldSize: options.worldSize, topology: topology, fabric: $0)
        }
        defer { comms.forEach { $0.shutdown() } }

        var rows: [BenchRow] = []
        for size in options.sizes {
            for algorithm in options.algorithms {
                guard let plan = algorithm.plan(worldSize: options.worldSize) else { continue }
                for comm in comms { comm.planOverride = plan }
                for codec in options.codecs {
                    guard codec.applies(to: options.collective, dataType: options.dataType) else { continue }
                    let row = await measure(size: size, algorithm: algorithm, codec: codec,
                                            comms: comms, options: options)
                    rows.append(row)
                    onRow?(row)
                }
            }
        }
        return rows
    }

    /// One point of the grid, over whatever ranks live in this process — every
    /// rank in the in-process runner, exactly one in the distributed one.
    ///
    /// `barrier` runs after the warm-up and before the clock starts. In-process
    /// there is nothing to wait for; across machines it keeps one rank's setup
    /// out of another rank's stopwatch.
    static func measure(
        size: Int, algorithm: BenchAlgorithm, codec: BenchCodec,
        comms: [Communicator], options: BenchOptions,
        barrier: (@Sendable () async throws -> Void)? = nil
    ) async -> BenchRow {
        let count = options.collective.elementCount(
            messageBytes: size, dataType: options.dataType, worldSize: options.worldSize)
        let iterations = options.iterations(forMessageBytes: size)
        let payloadBytes = count * options.dataType.byteWidth
            * (options.collective == .allGather || options.collective == .reduceScatter
               ? options.worldSize : 1)

        func row(seconds: Double, wire: Int?) -> BenchRow {
            let perCall = seconds / Double(iterations)
            let algorithmic = perCall > 0 ? Double(payloadBytes) / perCall : 0
            return BenchRow(
                messageBytes: payloadBytes, algorithm: algorithm, codec: codec.description,
                iterations: iterations, seconds: perCall,
                algorithmicBytesPerSecond: algorithmic,
                busBytesPerSecond: algorithmic * options.collective.busFactor(worldSize: options.worldSize),
                wireBytesPerRank: wire, failure: nil)
        }

        do {
            try await sweep(comms: comms, count: count, codec: codec,
                            options: options, iterations: options.warmup)
            try await barrier?()
            let start = DispatchTime.now().uptimeNanoseconds
            try await sweep(comms: comms, count: count, codec: codec,
                            options: options, iterations: iterations)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

            var wire: Int?
            if case .topK(let fraction) = codec, options.collective == .allReduce {
                wire = comms[0].topKWireBytes(count: count, dataType: options.dataType, fraction: fraction)
            }
            return row(seconds: seconds, wire: wire)
        } catch {
            return BenchRow(
                messageBytes: payloadBytes, algorithm: algorithm, codec: codec.description,
                iterations: 0, seconds: 0, algorithmicBytesPerSecond: 0, busBytesPerSecond: 0,
                wireBytesPerRank: nil, failure: "\(error)")
        }
    }

    /// Runs `iterations` collectives across every rank concurrently.
    private static func sweep(
        comms: [Communicator], count: Int, codec: BenchCodec,
        options: BenchOptions, iterations: Int
    ) async throws {
        guard iterations > 0 else { return }
        let dataType = options.dataType
        let op = options.op
        let collective = options.collective
        let compression = codec.compression
        let worldSize = options.worldSize

        try await withThrowingTaskGroup(of: Void.self) { group in
            for comm in comms {
                group.addTask {
                    let scratch = BenchBuffers(count: count, dataType: dataType,
                                               worldSize: worldSize, collective: collective)
                    defer { scratch.release() }
                    for _ in 0..<iterations {
                        scratch.refill(rank: comm.rank)
                        switch collective {
                        case .allReduce:
                            try await comm.allReduce(scratch.primary, count: count,
                                                     dataType: dataType, op: op,
                                                     compression: compression)
                        case .allGather:
                            try await comm.allGather(UnsafeRawBufferPointer(scratch.primary),
                                                     into: scratch.secondary, count: count,
                                                     dataType: dataType, compression: compression)
                        case .broadcast:
                            try await comm.broadcast(scratch.primary, count: count,
                                                     dataType: dataType, root: 0,
                                                     compression: compression)
                        case .reduceScatter:
                            try await comm.reduceScatter(UnsafeRawBufferPointer(scratch.secondary),
                                                         into: scratch.primary, recvCount: count,
                                                         dataType: dataType, op: op,
                                                         compression: compression)
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    }
}

/// Buffers for one rank's run. `primary` is the caller's tensor; `secondary` is
/// the `worldSize`-sized side of a gather or scatter.
private final class BenchBuffers: @unchecked Sendable {
    let primary: UnsafeMutableRawBufferPointer
    let secondary: UnsafeMutableRawBufferPointer
    private let count: Int
    private let dataType: DataType
    private let worldSize: Int

    init(count: Int, dataType: DataType, worldSize: Int, collective: BenchCollective) {
        self.count = count
        self.dataType = dataType
        self.worldSize = worldSize
        let bytes = max(16, count * dataType.byteWidth)
        primary = UnsafeMutableRawBufferPointer.allocate(byteCount: bytes, alignment: 64)
        secondary = UnsafeMutableRawBufferPointer.allocate(
            byteCount: max(16, bytes * worldSize), alignment: 64)
        primary.initializeMemory(as: UInt8.self, repeating: 0)
        secondary.initializeMemory(as: UInt8.self, repeating: 0)
    }

    /// Refill between iterations: an all-reduce is destructive, and reducing an
    /// already-reduced buffer would grow the values without bound (and, under
    /// top-k, feed the residual something that is not a gradient).
    func refill(rank: Int) {
        let width = dataType.byteWidth
        guard let base = primary.baseAddress else { return }
        // Cheap, deterministic, and varied enough that top-k has a real choice.
        let stride = max(1, count / 64)
        for i in Swift.stride(from: 0, to: count, by: stride) {
            let value = Float((rank + 1) * ((i % 97) + 1)) / 97.0
            switch dataType {
            case .float32: base.storeBytes(of: value, toByteOffset: i * width, as: Float.self)
            case .float16: base.storeBytes(of: Float16(value), toByteOffset: i * width, as: Float16.self)
            case .bfloat16:
                base.storeBytes(of: UInt16(truncatingIfNeeded: value.bitPattern >> 16),
                                toByteOffset: i * width, as: UInt16.self)
            case .int32: base.storeBytes(of: Int32(i % 97), toByteOffset: i * width, as: Int32.self)
            case .int8: base.storeBytes(of: Int8(i % 97), toByteOffset: i * width, as: Int8.self)
            }
        }
    }

    func release() {
        primary.deallocate()
        secondary.deallocate()
    }
}

// MARK: - Table rendering

public enum BenchTable {
    public static func header(_ options: BenchOptions) -> String {
        options.csv
            ? "size_bytes,algorithm,codec,iterations,wall_ms,alg_GB_s,bus_GB_s,wire_bytes_per_rank"
            : pad("size", 12, left: false) + "  " + pad("algorithm", 14) + pad("codec", 14)
              + pad("iters", 7, left: false) + "  " + pad("wall", 12, left: false) + "  "
              + pad("GB/s alg", 10, left: false) + "  " + pad("GB/s bus", 10, left: false)
    }

    public static func rule(_ options: BenchOptions) -> String? {
        options.csv ? nil : String(repeating: "-", count: 12 + 2 + 14 + 14 + 7 + 2 + 12 + 2 + 10 + 2 + 10)
    }

    public static func render(_ row: BenchRow, options: BenchOptions) -> String {
        if options.csv {
            guard row.failure == nil else {
                return "\(row.messageBytes),\(row.algorithm.rawValue),\(row.codec),0,,,,\(escape(row.failure!))"
            }
            return [
                String(row.messageBytes), row.algorithm.rawValue, row.codec, String(row.iterations),
                String(format: "%.4f", row.seconds * 1e3),
                String(format: "%.4f", row.algorithmicBytesPerSecond / 1e9),
                String(format: "%.4f", row.busBytesPerSecond / 1e9),
                row.wireBytesPerRank.map(String.init) ?? "",
            ].joined(separator: ",")
        }
        let prefix = pad(bytes(row.messageBytes), 12, left: false) + "  "
            + pad(row.algorithm.rawValue, 14) + pad(row.codec, 14)
        guard row.failure == nil else { return prefix + "—  " + row.failure! }
        return prefix
            + pad(String(row.iterations), 7, left: false) + "  "
            + pad(seconds(row.seconds), 12, left: false) + "  "
            + pad(String(format: "%.3f", row.algorithmicBytesPerSecond / 1e9), 10, left: false) + "  "
            + pad(String(format: "%.3f", row.busBytesPerSecond / 1e9), 10, left: false)
    }

    public static func bytes(_ value: Int) -> String {
        let units = ["B", "KiB", "MiB", "GiB"]
        var v = Double(value)
        var unit = 0
        while v >= 1024, unit < units.count - 1 { v /= 1024; unit += 1 }
        return unit == 0 ? "\(value) B"
            : (v == v.rounded() ? String(format: "%.0f %@", v, units[unit])
                                : String(format: "%.1f %@", v, units[unit]))
    }

    public static func seconds(_ value: Double) -> String {
        if value <= 0 { return "—" }
        if value < 1e-3 { return String(format: "%.1f µs", value * 1e6) }
        if value < 1 { return String(format: "%.3f ms", value * 1e3) }
        return String(format: "%.3f s", value)
    }

    /// `%@` ignores width flags on Darwin, so pad in Swift.
    public static func pad(_ value: String, _ width: Int, left: Bool = true) -> String {
        let padding = String(repeating: " ", count: max(0, width - value.count))
        return left ? value + padding : padding + value
    }

    private static func escape(_ value: String) -> String {
        "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
