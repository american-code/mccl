import XCTest
@testable import MCCL

final class PlannerTests: XCTestCase {

    func testPlannerReturnsRingForUnprobedTopology() {
        let nodes = (0..<4).map {
            Topology.Node(id: $0, hostname: "node\($0)", chip: "M4 Max", unifiedMemoryBytes: 128 << 30)
        }
        let topo = Topology(nodes: nodes, links: [])
        let plan = TopologyPlanner.plan(for: topo, messageBytes: 64 << 20)
        guard case .ring(let order) = plan else {
            return XCTFail("expected ring plan for unprobed topology")
        }
        XCTAssertEqual(order.sorted(), [0, 1, 2, 3])
    }

    func testUnprobedTopologyRingsRegardlessOfMessageSize() {
        let topo = Topology.uniform(nodeCount: 4)   // no measurements at all
        for bytes in [1, 1 << 10, 1 << 30] {
            guard case .ring = TopologyPlanner.plan(for: topo, messageBytes: bytes) else {
                return XCTFail("unprobed topology must fall back to ring at \(bytes) bytes")
            }
        }
    }

    // MARK: - uniform fabrics

    func testUniformFastFabricRingsForLargeMessagesAndTreesForSmall() {
        let topo = Fabrics.uniformFast(nodeCount: 4)
        let analysis = TopologyPlanner.analyze(topo)
        XCTAssertNil(analysis.islands, "a single TB5 fabric is not mixed-speed")
        let crossover = try! XCTUnwrap(analysis.treeRingCrossoverBytes)

        guard case .ring = TopologyPlanner.plan(for: topo, messageBytes: crossover * 4) else {
            return XCTFail("bandwidth-bound message should ring")
        }
        guard case .tree = TopologyPlanner.plan(for: topo, messageBytes: crossover / 4) else {
            return XCTFail("latency-bound message should tree")
        }
    }

    func testUniformSlowFabricRingsForLargeMessagesAndTreesForSmall() {
        let topo = Fabrics.uniformSlow(nodeCount: 4)
        let analysis = TopologyPlanner.analyze(topo)
        XCTAssertNil(analysis.islands)
        let crossover = try! XCTUnwrap(analysis.treeRingCrossoverBytes)

        guard case .tree = TopologyPlanner.plan(for: topo, messageBytes: 1 << 10) else {
            return XCTFail("1 KiB over 10GbE is latency-bound; expected tree")
        }
        guard case .ring = TopologyPlanner.plan(for: topo, messageBytes: 64 << 20) else {
            return XCTFail("64 MiB is bandwidth-bound; expected ring")
        }
        XCTAssertGreaterThan(crossover, 1 << 10)
        XCTAssertLessThan(crossover, 64 << 20)
    }

    /// The slow fabric has both higher latency and lower bandwidth. Latency
    /// dominates the crossover, so its switch point sits at a larger message.
    func testSlowerFabricSwitchesToRingAtALargerMessage() {
        let fast = try! XCTUnwrap(TopologyPlanner.analyze(Fabrics.uniformFast(nodeCount: 4)).treeRingCrossoverBytes)
        let slow = try! XCTUnwrap(TopologyPlanner.analyze(Fabrics.uniformSlow(nodeCount: 4)).treeRingCrossoverBytes)
        XCTAssertGreaterThan(slow, fast)
    }

    // MARK: - mixed fabrics

    func testTwoThunderboltIslandsBridgedByEthernetPlanHierarchically() {
        let topo = Fabrics.twoIslandsBridged()
        let analysis = TopologyPlanner.analyze(topo)
        XCTAssertEqual(analysis.islands ?? [], [[0, 1], [2, 3]])
        XCTAssertGreaterThan(analysis.heterogeneityRatio ?? 0, 3.0)

        guard case .hierarchical(let islands, let root) =
                TopologyPlanner.plan(for: topo, messageBytes: 64 << 20) else {
            return XCTFail("expected a hierarchical plan for a bridged fabric")
        }
        XCTAssertEqual(islands, [[0, 1], [2, 3]])
        XCTAssertTrue([0, 2].contains(root), "the inter-island root must be an island leader")
    }

    func testMixedFabricStillTreesForLatencyBoundMessages() {
        let topo = Fabrics.twoIslandsBridged()
        guard case .tree = TopologyPlanner.plan(for: topo, messageBytes: 256) else {
            return XCTFail("a 256-byte message is latency-bound even on a mixed fabric")
        }
    }

    // MARK: - one island and a bridge rank

    /// The textbook mixed case, and the smallest one a real lab produces: two
    /// machines on a fast cable plus a third that can only be reached slowly.
    ///
    /// It is worth being explicit that this *is* covered, because the obvious
    /// reading of "hierarchical needs two islands of two ranks" says it is not.
    /// What island detection actually requires is at least two groups, fewer
    /// groups than ranks, and at least one group with more than one member. A
    /// singleton island satisfies all three: it is an island whose intra-island
    /// reduction is a no-op, and its leader is itself.
    func testOneFastIslandPlusABridgeRankPlansHierarchically() {
        let topo = Fabrics.islandPlusBridge()
        let analysis = TopologyPlanner.analyze(topo)
        XCTAssertEqual(analysis.islands ?? [], [[0, 1], [2]])
        XCTAssertGreaterThan(analysis.heterogeneityRatio ?? 0, 3.0)

        guard case .hierarchical(let islands, let root) =
                TopologyPlanner.plan(for: topo, messageBytes: 16 << 20) else {
            return XCTFail("expected a hierarchical plan for a 2+1 fabric")
        }
        XCTAssertEqual(islands, [[0, 1], [2]])
        XCTAssertTrue([0, 2].contains(root), "the inter-island root must be an island leader")
    }

    /// The plan is only worth having if the collective can execute it. With a
    /// singleton island, step 1 (reduce inside the island) and step 3 (push the
    /// result back down) are both no-ops for rank 2, and step 2 carries it.
    func testAHierarchicalPlanWithASingletonIslandAllReducesCorrectly() async throws {
        let topo = Fabrics.islandPlusBridge()
        let comms = try Communicator.tcpGroup(worldSize: 3, topology: topo)
        defer { comms.forEach { $0.shutdown() } }
        for comm in comms {
            comm.planOverride = .hierarchical(islands: [[0, 1], [2]], interIslandRoot: 0)
        }

        let count = 4096
        try await withThrowingTaskGroup(of: Void.self) { group in
            for comm in comms {
                group.addTask {
                    let buffer = UnsafeMutableRawBufferPointer.allocate(
                        byteCount: count * 4, alignment: 64)
                    defer { buffer.deallocate() }
                    let floats = buffer.bindMemory(to: Float.self)
                    for i in 0..<count { floats[i] = Float((comm.rank + 1) * (i % 7 + 1)) }
                    try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)
                    for i in 0..<count {
                        XCTAssertEqual(floats[i], Float(6 * (i % 7 + 1)), accuracy: 1e-3, "element \(i)")
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    func testASingletonIslandBroadcastsFromEitherSideOfTheBridge() async throws {
        let topo = Fabrics.islandPlusBridge()
        for root in 0..<3 {
            let comms = try Communicator.tcpGroup(worldSize: 3, topology: topo)
            defer { comms.forEach { $0.shutdown() } }
            for comm in comms {
                comm.planOverride = .hierarchical(islands: [[0, 1], [2]], interIslandRoot: 0)
            }
            let count = 256
            try await withThrowingTaskGroup(of: Void.self) { group in
                for comm in comms {
                    group.addTask {
                        let buffer = UnsafeMutableRawBufferPointer.allocate(
                            byteCount: count * 4, alignment: 64)
                        defer { buffer.deallocate() }
                        let floats = buffer.bindMemory(to: Float.self)
                        for i in 0..<count { floats[i] = comm.rank == root ? Float(i) : -1 }
                        try await comm.broadcast(buffer, count: count, dataType: .float32, root: root)
                        for i in 0..<count {
                            XCTAssertEqual(floats[i], Float(i), "root \(root), rank \(comm.rank), element \(i)")
                        }
                    }
                }
                try await group.waitForAll()
            }
        }
    }

    /// Two ranks cannot be mixed however different the one link between them
    /// is: one link has no gap to find, and splitting them would make two
    /// singleton islands, which is a ring with extra words.
    func testTwoRanksAreNeverHierarchical() {
        let nodes = (0..<2).map {
            Topology.Node(id: $0, hostname: "n\($0)", chip: "M1 Max", unifiedMemoryBytes: 1)
        }
        let topo = Topology(nodes: nodes, links: [
            Topology.Link(from: 0, to: 1, kind: .wifi, measuredBandwidth: 6e7, measuredLatency: 2e-3),
        ])
        XCTAssertNil(TopologyPlanner.analyze(topo).islands)
    }

    func testThreeIslandsAreDetected() {
        let nodes = (0..<6).map {
            Topology.Node(id: $0, hostname: "n\($0)", chip: "M2 Ultra", unifiedMemoryBytes: 192 << 30)
        }
        var links: [Topology.Link] = []
        for pair in [(0, 1), (2, 3), (4, 5)] {
            links.append(Topology.Link(from: pair.0, to: pair.1, kind: .thunderbolt4,
                                       measuredBandwidth: 3.0e9, measuredLatency: 30e-6))
        }
        for a in [0, 2, 4] {
            for b in [0, 2, 4] where a < b {
                links.append(Topology.Link(from: a, to: b, kind: .ethernet,
                                           measuredBandwidth: 1.1e8, measuredLatency: 400e-6))
            }
        }
        let topo = Topology(nodes: nodes, links: links)
        guard case .hierarchical(let islands, _) =
                TopologyPlanner.plan(for: topo, messageBytes: 128 << 20) else {
            return XCTFail("expected hierarchical")
        }
        XCTAssertEqual(islands, [[0, 1], [2, 3], [4, 5]])
    }

    func testMildBandwidthVariationIsNotTreatedAsIslands() {
        let nodes = (0..<4).map {
            Topology.Node(id: $0, hostname: "n\($0)", chip: "M4", unifiedMemoryBytes: 32 << 30)
        }
        // Within 1.5x — jitter on one fabric, not two fabrics.
        let links = [
            Topology.Link(from: 0, to: 1, kind: .thunderbolt5, measuredBandwidth: 5.5e9, measuredLatency: 20e-6),
            Topology.Link(from: 1, to: 2, kind: .thunderbolt5, measuredBandwidth: 4.4e9, measuredLatency: 22e-6),
            Topology.Link(from: 2, to: 3, kind: .thunderbolt5, measuredBandwidth: 5.1e9, measuredLatency: 21e-6),
            Topology.Link(from: 0, to: 3, kind: .thunderbolt5, measuredBandwidth: 3.9e9, measuredLatency: 24e-6),
        ]
        let topo = Topology(nodes: nodes, links: links)
        XCTAssertNil(TopologyPlanner.analyze(topo).islands)
        guard case .ring = TopologyPlanner.plan(for: topo, messageBytes: 64 << 20) else {
            return XCTFail("a near-uniform fabric should stay on the ring")
        }
    }

    // MARK: - the threshold itself

    func testCrossoverIsDerivedFromMeasurementsNotConstants() {
        let base = TopologyPlanner.crossoverBytes(rankCount: 4, latency: 20e-6, bandwidth: 5.5e9)
        let doubleLatency = TopologyPlanner.crossoverBytes(rankCount: 4, latency: 40e-6, bandwidth: 5.5e9)
        let doubleBandwidth = TopologyPlanner.crossoverBytes(rankCount: 4, latency: 20e-6, bandwidth: 11.0e9)

        XCTAssertGreaterThan(base, 0)
        XCTAssertEqual(Double(doubleLatency), Double(base) * 2, accuracy: 2)
        XCTAssertEqual(Double(doubleBandwidth), Double(base) * 2, accuracy: 2)
    }

    func testCrossoverIsZeroForTwoRanks() {
        // With two ranks a tree *is* a ring; there is nothing to switch.
        XCTAssertEqual(TopologyPlanner.crossoverBytes(rankCount: 2, latency: 1e-3, bandwidth: 1e9), 0)
    }

    func testCrossoverGrowsWithRankCount() {
        let four = TopologyPlanner.crossoverBytes(rankCount: 4, latency: 50e-6, bandwidth: 1e9)
        let eight = TopologyPlanner.crossoverBytes(rankCount: 8, latency: 50e-6, bandwidth: 1e9)
        let sixteen = TopologyPlanner.crossoverBytes(rankCount: 16, latency: 50e-6, bandwidth: 1e9)
        XCTAssertLessThan(four, eight)
        XCTAssertLessThan(eight, sixteen)
    }

    func testCrossoverIsZeroWithoutMeasurements() {
        XCTAssertEqual(TopologyPlanner.crossoverBytes(rankCount: 4, latency: 0, bandwidth: 5e9), 0)
        XCTAssertEqual(TopologyPlanner.crossoverBytes(rankCount: 4, latency: 20e-6, bandwidth: 0), 0)
    }

    // MARK: - orders and trees

    func testRingOrderFollowsTheFastestLinks() {
        let nodes = (0..<4).map {
            Topology.Node(id: $0, hostname: "n\($0)", chip: "M4", unifiedMemoryBytes: 0)
        }
        // Fast path 0-2-1-3, everything else slow.
        let links = [
            Topology.Link(from: 0, to: 2, kind: .thunderbolt5, measuredBandwidth: 5e9),
            Topology.Link(from: 2, to: 1, kind: .thunderbolt5, measuredBandwidth: 5e9),
            Topology.Link(from: 1, to: 3, kind: .thunderbolt5, measuredBandwidth: 5e9),
            Topology.Link(from: 0, to: 1, kind: .ethernet, measuredBandwidth: 1e8),
            Topology.Link(from: 0, to: 3, kind: .ethernet, measuredBandwidth: 1e8),
            Topology.Link(from: 2, to: 3, kind: .ethernet, measuredBandwidth: 1e8),
        ]
        let topo = Topology(nodes: nodes, links: links)
        XCTAssertEqual(TopologyPlanner.ringOrder(for: topo), [0, 2, 1, 3])
    }

    func testBinomialTreeShape() {
        let children = TopologyPlanner.binomialTree(order: [0, 1, 2, 3], root: 0)
        XCTAssertEqual(children[0], [1, 2])
        XCTAssertEqual(children[2], [3])
        XCTAssertNil(children[1])
        XCTAssertNil(children[3])

        let parents = TopologyPlanner.parents(of: children)
        XCTAssertEqual(parents[1], 0)
        XCTAssertEqual(parents[2], 0)
        XCTAssertEqual(parents[3], 2)
        XCTAssertNil(parents[0])
    }

    func testBinomialTreeCoversEveryRankExactlyOnceForAnyRoot() {
        for size in 2...9 {
            let order = Array(0..<size)
            for root in order {
                let children = TopologyPlanner.binomialTree(order: order, root: root)
                let parents = TopologyPlanner.parents(of: children)
                XCTAssertNil(parents[root], "root \(root) must have no parent (n=\(size))")
                XCTAssertEqual(parents.count, size - 1,
                               "every non-root must have exactly one parent (n=\(size), root=\(root))")
                // Every rank must reach the root by walking up.
                for rank in order where rank != root {
                    var cursor = rank
                    var hops = 0
                    while cursor != root {
                        cursor = try! XCTUnwrap(parents[cursor])
                        hops += 1
                        XCTAssertLessThanOrEqual(hops, size, "cycle in tree (n=\(size), root=\(root))")
                    }
                    XCTAssertLessThanOrEqual(
                        hops, Int(ceil(log2(Double(size)))),
                        "tree depth must be O(log n) (n=\(size), root=\(root), rank=\(rank))")
                }
            }
        }
    }

    func testBinomialTreeRespectsARotatedOrder() {
        let children = TopologyPlanner.binomialTree(order: [3, 1, 0, 2], root: 1)
        let parents = TopologyPlanner.parents(of: children)
        XCTAssertNil(parents[1])
        XCTAssertEqual(Set(parents.keys), [0, 2, 3])
    }

    // MARK: - sanitisation

    func testCommunicatorRejectsPlansThatDoNotMatchTheWorld() {
        let comm = Communicator(rank: 0, worldSize: 4, topology: Topology.uniform(nodeCount: 4))
        // A stale topology naming ranks that do not exist here.
        XCTAssertEqual(comm.sanitize(.ring(order: [0, 1, 2, 7])), .ring(order: [0, 1, 2, 3]))
        XCTAssertEqual(comm.sanitize(.tree(root: 9, children: [:])), .ring(order: [0, 1, 2, 3]))
        XCTAssertEqual(comm.sanitize(.hierarchical(islands: [[0, 1]], interIslandRoot: 0)),
                       .ring(order: [0, 1, 2, 3]))
        XCTAssertEqual(comm.sanitize(.hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 0)),
                       .hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 0))
    }

    // MARK: - persistence

    func testTopologyJSONRoundTrip() throws {
        let original = Fabrics.twoIslandsBridged()
        let restored = try Topology.from(jsonData: original.jsonData())
        XCTAssertEqual(restored.nodes, original.nodes)
        XCTAssertEqual(restored.links, original.links)
        XCTAssertEqual(TopologyPlanner.plan(for: restored, messageBytes: 64 << 20),
                       TopologyPlanner.plan(for: original, messageBytes: 64 << 20))
    }

    func testTopologyFileRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mccl-topology-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let original = Fabrics.uniformFast(nodeCount: 3)
        try original.write(to: url)
        let restored = try Topology.read(from: url)
        XCTAssertEqual(restored.links.count, original.links.count)
        XCTAssertEqual(restored.links.first?.kind, .thunderbolt5)
        XCTAssertEqual(restored.links.first?.measuredBandwidth, 5.5e9)
    }
}
