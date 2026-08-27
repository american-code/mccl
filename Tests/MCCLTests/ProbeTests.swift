import XCTest
@testable import MCCL

final class ProbeTests: XCTestCase {

    /// Small enough to keep the suite fast, large enough that the timings mean
    /// something. The CLI defaults are far bigger.
    private func fastOptions() -> TopologyProbe.Options {
        var options = TopologyProbe.Options()
        options.streamBytes = 8 << 20
        options.chunkBytes = 1 << 20
        options.pingIterations = 25
        options.warmupIterations = 5
        options.timeout = 20
        return options
    }

    private func startServer() throws -> ProbeServer {
        let server = try ProbeServer(host: "127.0.0.1", port: 0)
        server.start()
        return server
    }

    func testMeasureAgainstALocallySpawnedServe() throws {
        let server = try startServer()
        defer { server.stop() }
        XCTAssertGreaterThan(server.address.port, 0)

        let measurement = try TopologyProbe.measure(peer: server.address, options: fastOptions())

        XCTAssertEqual(measurement.identity.hostname, ProcessInfo.processInfo.hostName)
        XCTAssertFalse(measurement.identity.chip.isEmpty)
        XCTAssertGreaterThan(measurement.identity.unifiedMemoryBytes, 0)
        XCTAssertGreaterThan(measurement.bandwidthBytesPerSecond, 0)
        XCTAssertGreaterThan(measurement.roundTripSeconds, 0)
        // Loopback on an M1 Pro is comfortably above 100 MB/s and well under a
        // millisecond round-trip; anything outside that means we mis-measured.
        XCTAssertGreaterThan(measurement.bandwidthBytesPerSecond, 1e8)
        XCTAssertLessThan(measurement.roundTripSeconds, 5e-3)
    }

    func testProbeOverLoopbackTransport() throws {
        let transport = LoopbackTransport()
        let server = try ProbeServer(transport: transport, host: "loopback", port: 0)
        server.start()
        defer { server.stop() }

        let measurement = try TopologyProbe.measure(
            peer: server.address, transport: transport, options: fastOptions())
        XCTAssertGreaterThan(measurement.bandwidthBytesPerSecond, 0)
        XCTAssertEqual(TopologyProbe.inferKind(address: server.address,
                                               bandwidth: measurement.bandwidthBytesPerSecond),
                       .loopback)
    }

    func testBuildTopologyProducesAMeasuredPairwiseMap() throws {
        let a = try startServer()
        let b = try startServer()
        defer { a.stop(); b.stop() }

        var log: [String] = []
        let topology = try TopologyProbe.buildTopology(
            peers: [a.address, b.address], options: fastOptions(), onProgress: { log.append($0) })

        XCTAssertEqual(topology.nodes.count, 3, "rank 0 is this machine plus the two peers")
        XCTAssertEqual(topology.nodes.map(\.id), [0, 1, 2])

        // Direct links from rank 0, plus the peer-to-peer link measured by
        // rank 1 on our behalf.
        XCTAssertNotNil(topology.bandwidth(between: 0, 1))
        XCTAssertNotNil(topology.bandwidth(between: 0, 2))
        XCTAssertNotNil(topology.bandwidth(between: 1, 2),
                        "pairwise probing should have measured 1<->2")

        for link in topology.links {
            XCTAssertGreaterThan(link.measuredBandwidth ?? 0, 0)
            XCTAssertGreaterThan(link.measuredLatency ?? 0, 0)
            XCTAssertEqual(link.kind, .loopback)
        }
        XCTAssertFalse(log.isEmpty)
    }

    func testMeasuredTopologyIsPersistableAndPlannable() throws {
        let server = try startServer()
        defer { server.stop() }

        var options = fastOptions()
        options.pairwise = false
        let topology = try TopologyProbe.buildTopology(peers: [server.address], options: options)

        let restored = try Topology.from(jsonData: topology.jsonData())
        XCTAssertEqual(restored.nodes.count, 2)
        XCTAssertEqual(restored.links.count, 1)
        XCTAssertEqual(restored.links[0].measuredBandwidth, topology.links[0].measuredBandwidth)

        let analysis = TopologyPlanner.analyze(restored)
        XCTAssertTrue(analysis.isProbed)
        XCTAssertEqual(analysis.rankCount, 2)
        // Two ranks: tree and ring are the same shape, so the planner rings.
        guard case .ring = TopologyPlanner.plan(for: restored, messageBytes: 1 << 20) else {
            return XCTFail("expected ring for a two-rank world")
        }
    }

    func testRemoteProbeReturnsTheRemoteMeasurement() throws {
        let a = try startServer()
        let b = try startServer()
        defer { a.stop(); b.stop() }

        let measurement = try TopologyProbe.measureRemote(
            from: a.address, to: b.address, options: fastOptions())
        XCTAssertEqual(measurement.address, b.address)
        XCTAssertGreaterThan(measurement.bandwidthBytesPerSecond, 0)
        XCTAssertGreaterThan(measurement.roundTripSeconds, 0)
    }

    func testProbingADeadPeerFailsCleanly() {
        // Port 1 on loopback is not going to answer.
        var options = fastOptions()
        options.timeout = 0.5
        XCTAssertThrowsError(
            try TopologyProbe.measure(peer: PeerAddress(host: "127.0.0.1", port: 1), options: options)
        ) { error in
            guard error is MCCLError else { return XCTFail("expected MCCLError, got \(error)") }
        }
    }

    func testLinkKindInference() {
        XCTAssertEqual(TopologyProbe.inferKind(address: PeerAddress(host: "127.0.0.1", port: 1),
                                               bandwidth: 9e9), .loopback)
        XCTAssertEqual(TopologyProbe.inferKind(address: PeerAddress(host: "10.0.0.2", port: 1),
                                               bandwidth: 6e9), .thunderbolt5)
        XCTAssertEqual(TopologyProbe.inferKind(address: PeerAddress(host: "10.0.0.2", port: 1),
                                               bandwidth: 2.5e9), .thunderbolt4)
        XCTAssertEqual(TopologyProbe.inferKind(address: PeerAddress(host: "10.0.0.2", port: 1),
                                               bandwidth: 1.2e9), .usb4)
        XCTAssertEqual(TopologyProbe.inferKind(address: PeerAddress(host: "10.0.0.2", port: 1),
                                               bandwidth: 1.1e8), .ethernet)
    }

    func testNodeIdentityReflectsThisMachine() {
        let identity = NodeIdentity.local()
        XCTAssertEqual(identity.hostname, ProcessInfo.processInfo.hostName)
        XCTAssertGreaterThan(identity.unifiedMemoryBytes, 1 << 30)
        let node = identity.node(id: 7)
        XCTAssertEqual(node.id, 7)
        XCTAssertEqual(node.chip, identity.chip)
    }
}
