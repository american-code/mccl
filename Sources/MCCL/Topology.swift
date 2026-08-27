import Foundation

/// A measured (not assumed) map of the cluster's interconnect.
///
/// Nodes advertise their links; `TopologyProbe` measures real point-to-point
/// bandwidth/latency per link, and the planner picks ring vs. tree vs.
/// hierarchical (intra-TB ring + inter-Ethernet tree) from the measurements.
public struct Topology: Sendable, Codable {
    public struct Node: Sendable, Identifiable, Codable, Equatable {
        public let id: Int
        public let hostname: String
        public let chip: String            // e.g. "M4 Max", from sysctl
        public let unifiedMemoryBytes: UInt64
        public init(id: Int, hostname: String, chip: String, unifiedMemoryBytes: UInt64) {
            self.id = id
            self.hostname = hostname
            self.chip = chip
            self.unifiedMemoryBytes = unifiedMemoryBytes
        }
    }

    public struct Link: Sendable, Codable, Equatable {
        public enum Kind: String, Sendable, Codable {
            case thunderbolt5, thunderbolt4, usb4, ethernet, wifi, loopback
        }
        public let from: Int
        public let to: Int
        public let kind: Kind
        /// Measured, in bytes/sec. Nil until probed.
        public let measuredBandwidth: Double?
        /// Measured round-trip, in seconds. Nil until probed.
        public let measuredLatency: Double?
        public init(from: Int, to: Int, kind: Kind, measuredBandwidth: Double? = nil, measuredLatency: Double? = nil) {
            self.from = from
            self.to = to
            self.kind = kind
            self.measuredBandwidth = measuredBandwidth
            self.measuredLatency = measuredLatency
        }
    }

    public var nodes: [Node]
    public var links: [Link]

    public init(nodes: [Node], links: [Link]) {
        self.nodes = nodes
        self.links = links
    }
}

// MARK: - Persistence

extension Topology {
    public func jsonData(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        return try encoder.encode(self)
    }

    public static func from(jsonData data: Data) throws -> Topology {
        try JSONDecoder().decode(Topology.self, from: data)
    }

    public func write(to url: URL) throws {
        try jsonData().write(to: url, options: .atomic)
    }

    public static func read(from url: URL) throws -> Topology {
        try from(jsonData: Data(contentsOf: url))
    }
}

// MARK: - Convenience constructors

extension Topology {
    /// A fully-connected topology with identical links — the "we have not
    /// probed anything yet, assume symmetry" default.
    public static func uniform(
        nodeCount: Int,
        kind: Link.Kind = .loopback,
        bandwidth: Double? = nil,
        latency: Double? = nil,
        chip: String = "unknown",
        memoryBytes: UInt64 = 0
    ) -> Topology {
        let nodes = (0..<nodeCount).map {
            Node(id: $0, hostname: "rank\($0)", chip: chip, unifiedMemoryBytes: memoryBytes)
        }
        var links: [Link] = []
        for i in 0..<nodeCount {
            for j in (i + 1)..<nodeCount {
                links.append(Link(from: i, to: j, kind: kind,
                                  measuredBandwidth: bandwidth, measuredLatency: latency))
            }
        }
        return Topology(nodes: nodes, links: links)
    }

    /// Best measured bandwidth between two ranks, in either direction.
    public func bandwidth(between a: Int, _ b: Int) -> Double? {
        links.compactMap { link -> Double? in
            guard (link.from == a && link.to == b) || (link.from == b && link.to == a) else { return nil }
            return link.measuredBandwidth
        }.max()
    }

    /// Best (lowest) measured round-trip between two ranks.
    public func latency(between a: Int, _ b: Int) -> Double? {
        links.compactMap { link -> Double? in
            guard (link.from == a && link.to == b) || (link.from == b && link.to == a) else { return nil }
            return link.measuredLatency
        }.min()
    }
}

/// The algorithm the planner selected for one collective invocation.
public enum CollectivePlan: Sendable, Equatable {
    case ring(order: [Int])
    case tree(root: Int, children: [Int: [Int]])
    /// Ring within each island of fast links, tree across islands.
    case hierarchical(islands: [[Int]], interIslandRoot: Int)
}

extension CollectivePlan: CustomStringConvertible {
    public var description: String {
        switch self {
        case .ring(let order):
            return "ring(\(order.map(String.init).joined(separator: "->")))"
        case .tree(let root, let children):
            let edges = children.keys.sorted().map { "\($0):\(children[$0]!.map(String.init).joined(separator: ","))" }
            return "tree(root: \(root), \(edges.joined(separator: " ")))"
        case .hierarchical(let islands, let root):
            let desc = islands.map { "[\($0.map(String.init).joined(separator: ","))]" }.joined(separator: " ")
            return "hierarchical(islands: \(desc), interIslandRoot: \(root))"
        }
    }
}

/// What the planner concluded about a topology, before message size is applied.
/// Exposed so `mcclprobe` can print the reasoning and tests can assert on it.
public struct PlannerAnalysis: Sendable, Equatable {
    public let rankCount: Int
    /// Slowest measured link — the ring's actual bottleneck, bytes/sec.
    public let bottleneckBandwidth: Double?
    /// Worst measured round-trip, seconds. Used as the per-hop latency term.
    public let worstLatency: Double?
    /// Message size, in bytes, at which ring overtakes tree for this fabric.
    /// Derived from the measurements — see `TopologyPlanner.crossoverBytes`.
    public let treeRingCrossoverBytes: Int?
    /// Fast-link islands, when the fabric is measurably mixed-speed.
    public let islands: [[Int]]?
    /// max(bandwidth) / min(bandwidth) across measured links.
    public let heterogeneityRatio: Double?

    public var isHeterogeneous: Bool { islands != nil }
    public var isProbed: Bool { bottleneckBandwidth != nil }
}

public enum TopologyPlanner {
    /// A fabric counts as mixed-speed once its fastest measured link is this
    /// many times its slowest. 3x separates "same fabric, some jitter" from
    /// "TB5 island bridged by Ethernet" (that ratio is ~4-5x in practice).
    public static var heterogeneityRatio: Double = 3.0

    /// Pick a plan for a collective of `messageBytes` over `topology`.
    ///
    /// - Mixed-speed fabric and a bandwidth-bound message: `.hierarchical`, so
    ///   the slow bridge carries one island-sum per island instead of the whole
    ///   ring's traffic.
    /// - Message below the measured tree/ring crossover: `.tree` — O(log n)
    ///   hops beats the ring's 2(n-1) latency-dominated steps.
    /// - Otherwise `.ring`: each link carries 2(n-1)/n of the data, which is
    ///   bandwidth-optimal.
    /// - Unprobed topology: `.ring`, the safe default.
    public static func plan(for topology: Topology, messageBytes: Int) -> CollectivePlan {
        let ranks = topology.nodes.map(\.id).sorted()
        guard ranks.count > 1 else { return .ring(order: ranks) }

        let analysis = analyze(topology)
        guard analysis.isProbed, let crossover = analysis.treeRingCrossoverBytes else {
            return .ring(order: ringOrder(for: topology, ranks: ranks))
        }

        let latencyBound = messageBytes < crossover

        if let islands = analysis.islands, !latencyBound {
            return .hierarchical(islands: islands, interIslandRoot: interIslandRoot(islands, topology))
        }
        if latencyBound {
            let root = bestConnectedRank(topology, ranks: ranks)
            return .tree(root: root, children: binomialTree(order: ringOrder(for: topology, ranks: ranks), root: root))
        }
        return .ring(order: ringOrder(for: topology, ranks: ranks))
    }

    /// The measured picture the planner reasons over.
    public static func analyze(_ topology: Topology) -> PlannerAnalysis {
        let ranks = topology.nodes.map(\.id).sorted()
        let n = ranks.count

        let bandwidths = topology.links.compactMap(\.measuredBandwidth).filter { $0 > 0 && $0.isFinite }
        let latencies = topology.links.compactMap(\.measuredLatency).filter { $0 >= 0 && $0.isFinite }

        guard let minBandwidth = bandwidths.min(), let maxBandwidth = bandwidths.max() else {
            return PlannerAnalysis(rankCount: n, bottleneckBandwidth: nil, worstLatency: nil,
                                   treeRingCrossoverBytes: nil, islands: nil, heterogeneityRatio: nil)
        }
        // Worst-case hop: a collective is only as fast as its slowest step.
        let alpha = latencies.max() ?? 0
        let crossover = crossoverBytes(rankCount: n, latency: alpha, bandwidth: minBandwidth)
        let ratio = maxBandwidth / minBandwidth
        let islands = detectIslands(topology, ranks: ranks, ratio: ratio)

        return PlannerAnalysis(
            rankCount: n,
            bottleneckBandwidth: minBandwidth,
            worstLatency: latencies.max(),
            treeRingCrossoverBytes: crossover,
            islands: islands,
            heterogeneityRatio: ratio
        )
    }

    /// Message size at which ring stops losing to tree.
    ///
    /// With per-hop latency α and bottleneck bandwidth B, for n ranks:
    ///
    ///     t_ring = 2(n-1)·α + 2·(n-1)/n·S/B
    ///     t_tree = 2·log2(n)·α + 2·log2(n)·S/B
    ///
    /// Equating and solving for S gives
    ///
    ///     S* = α·B·(log2 n − (n−1)) / ((n−1)/n − log2 n)
    ///
    /// Both factors are ≤ 0 for n ≥ 2, so S* ≥ 0. Note this is entirely a
    /// function of the *measured* α and B — there is no hardcoded byte count
    /// anywhere in mccl's algorithm selection.
    public static func crossoverBytes(rankCount n: Int, latency alpha: Double, bandwidth: Double) -> Int {
        guard n >= 2, alpha > 0, bandwidth > 0 else { return 0 }
        let logn = log2(Double(n))
        let numerator = logn - Double(n - 1)
        let denominator = Double(n - 1) / Double(n) - logn
        guard denominator != 0 else { return 0 }
        let bytes = alpha * bandwidth * (numerator / denominator)
        guard bytes.isFinite, bytes > 0 else { return 0 }
        return Int(bytes.rounded())
    }

    // MARK: - Island detection

    /// Splits the fabric into islands of fast links, or returns nil if the
    /// measured bandwidths are close enough to treat as one uniform fabric.
    ///
    /// The split point is the largest *ratio* gap between adjacent measured
    /// bandwidths, so it adapts to whatever fabrics are actually present rather
    /// than testing against named link speeds.
    static func detectIslands(_ topology: Topology, ranks: [Int], ratio: Double) -> [[Int]]? {
        guard ranks.count > 2, ratio >= heterogeneityRatio else { return nil }

        var best: [Set<Int>: Double] = [:]
        for link in topology.links {
            guard let bw = link.measuredBandwidth, bw > 0, bw.isFinite else { continue }
            let key: Set<Int> = [link.from, link.to]
            guard key.count == 2 else { continue }
            best[key] = max(best[key] ?? 0, bw)
        }
        let values = Array(Set(best.values)).sorted()
        guard values.count >= 2 else { return nil }

        var cut = values[1]
        var bestGap = 0.0
        for i in 0..<(values.count - 1) {
            let gap = values[i + 1] / values[i]
            if gap > bestGap {
                bestGap = gap
                cut = (values[i] * values[i + 1]).squareRoot()
            }
        }
        guard bestGap >= heterogeneityRatio else { return nil }

        var parent = Dictionary(uniqueKeysWithValues: ranks.map { ($0, $0) })
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r]! != r { r = parent[r]! }
            var c = x
            while parent[c]! != c { let next = parent[c]!; parent[c] = r; c = next }
            return r
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }
        for (pair, bw) in best where bw >= cut {
            let members = pair.sorted()
            guard members.count == 2, ranks.contains(members[0]), ranks.contains(members[1]) else { continue }
            union(members[0], members[1])
        }

        var groups: [Int: [Int]] = [:]
        for rank in ranks { groups[find(rank), default: []].append(rank) }
        let islands = groups.values.map { $0.sorted() }.sorted { $0[0] < $1[0] }

        // One island means uniform; n islands means we found no fast links at
        // all. Neither is worth a hierarchical plan.
        guard islands.count >= 2, islands.count < ranks.count,
              islands.contains(where: { $0.count >= 2 }) else { return nil }
        return islands
    }

    static func interIslandRoot(_ islands: [[Int]], _ topology: Topology) -> Int {
        var bestLeader = islands.first?.first ?? 0
        var bestScore = -Double.infinity
        for island in islands {
            guard let leader = island.first else { continue }
            let members = Set(island)
            var score = 0.0
            for link in topology.links {
                guard let bw = link.measuredBandwidth else { continue }
                let touchesIsland = members.contains(link.from) || members.contains(link.to)
                let crossesOut = !(members.contains(link.from) && members.contains(link.to))
                if touchesIsland && crossesOut { score += bw }
            }
            if score > bestScore || (score == bestScore && leader < bestLeader) {
                bestScore = score
                bestLeader = leader
            }
        }
        return bestLeader
    }

    // MARK: - Orders and trees

    /// Greedy maximum-bandwidth Hamiltonian path: from the lowest rank, always
    /// step to the unvisited neighbour with the fattest measured link. Falls
    /// back to rank order when nothing has been measured.
    public static func ringOrder(for topology: Topology, ranks: [Int]? = nil) -> [Int] {
        let ranks = ranks ?? topology.nodes.map(\.id).sorted()
        guard ranks.count > 2, topology.links.contains(where: { $0.measuredBandwidth != nil }) else {
            return ranks
        }
        var remaining = Set(ranks)
        var order: [Int] = []
        var current = ranks[0]
        remaining.remove(current)
        order.append(current)
        while !remaining.isEmpty {
            let next = remaining.max { a, b in
                let ba = topology.bandwidth(between: current, a) ?? 0
                let bb = topology.bandwidth(between: current, b) ?? 0
                if ba == bb { return a > b }   // deterministic: prefer lower id
                return ba < bb
            }!
            order.append(next)
            remaining.remove(next)
            current = next
        }
        return order
    }

    /// Best-connected rank by total measured bandwidth; ties go to the lowest id.
    public static func bestConnectedRank(_ topology: Topology, ranks: [Int]? = nil) -> Int {
        let ranks = ranks ?? topology.nodes.map(\.id).sorted()
        guard let first = ranks.first else { return 0 }
        var bestRank = first
        var bestScore = -Double.infinity
        for rank in ranks {
            var score = 0.0
            for link in topology.links {
                guard let bw = link.measuredBandwidth else { continue }
                if link.from == rank || link.to == rank { score += bw }
            }
            if score > bestScore { bestScore = score; bestRank = rank }
        }
        return bestRank
    }

    /// Binomial tree over `order`, rooted at `root`.
    ///
    /// Virtual index `v = (position - rootPosition) mod n`; the parent of `v` is
    /// `v` with its lowest set bit cleared. Depth is ⌈log2 n⌉, which is the
    /// property the latency-bound branch is buying.
    public static func binomialTree(order: [Int], root: Int) -> [Int: [Int]] {
        let n = order.count
        guard n > 1, let rootPosition = order.firstIndex(of: root) else { return [:] }
        var children: [Int: [Int]] = [:]
        for v in 0..<n {
            let actual = order[(rootPosition + v) % n]
            let lowestSetBit = v == 0 ? Int.max : (v & -v)
            var bit = 1
            var kids: [Int] = []
            while bit < n {
                if bit < lowestSetBit {
                    let child = v | bit
                    if child < n && child != v { kids.append(order[(rootPosition + child) % n]) }
                }
                bit <<= 1
            }
            if !kids.isEmpty { children[actual] = kids }
        }
        return children
    }

    /// Inverts a `children` map into child -> parent.
    public static func parents(of children: [Int: [Int]]) -> [Int: Int] {
        var result: [Int: Int] = [:]
        for (parent, kids) in children {
            for kid in kids { result[kid] = parent }
        }
        return result
    }
}
