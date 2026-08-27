import XCTest
@testable import MCCL

/// The strings and small accessors mccl exposes for diagnosis.
///
/// These are not decoration. `MCCLError.description` is what a C caller reads
/// back through `mcclGetLastError`, and `CollectivePlan.description` is the
/// whole answer `mcclCommPlanDescription` and `mcclprobe plan` give. If either
/// loses the offending value, an operator debugging a cluster is left with a
/// bare error code.
final class DiagnosticsTests: XCTestCase {

    // MARK: - Error text

    func testEveryErrorNamesItselfAndCarriesItsDetail() {
        let cases: [(MCCLError, [String])] = [
            (.notImplemented("thunderbolt transport"), ["not implemented", "thunderbolt transport"]),
            (.invalidArgument("blockSize must be > 0"), ["invalid argument", "blockSize must be > 0"]),
            (.rankOutOfRange(rank: 9, worldSize: 4), ["rank 9", "world size 4"]),
            (.noFabric, ["no transport fabric", "Communicator.bootstrap"]),
            (.bufferTooSmall(need: 128, have: 16), ["buffer too small", "need 128", "have 16"]),
            (.connectionClosed, ["peer closed the connection"]),
            (.protocolViolation("bad frame magic"), ["wire protocol violation", "bad frame magic"]),
            (.socketFailure("bind(0.0.0.0:7777)", errno: EADDRINUSE),
             ["socket failure", "bind(0.0.0.0:7777)", "errno \(EADDRINUSE)"]),
            (.timedOut("accept on 10.0.0.1:7000"), ["timed out", "accept on 10.0.0.1:7000"]),
            (.unsupportedCompression("topK on reduceScatter"), ["unsupported wire compression", "topK"]),
            (.topologyInvalid("link references rank 9"), ["invalid topology", "link references rank 9"]),
        ]
        for (error, fragments) in cases {
            let text = error.description
            XCTAssertTrue(text.hasPrefix("mccl: "), "every message identifies the library: '\(text)'")
            for fragment in fragments {
                XCTAssertTrue(text.contains(fragment),
                              "'\(text)' should mention '\(fragment)'")
            }
        }
        // socketFailure decodes errno rather than making the reader look it up.
        XCTAssertTrue(MCCLError.socketFailure("connect", errno: ECONNREFUSED).description
            .contains(String(cString: strerror(ECONNREFUSED))))
    }

    // MARK: - Plan text

    func testEveryPlanPrintsItsShape() {
        XCTAssertEqual(CollectivePlan.ring(order: [0, 2, 1, 3]).description, "ring(0->2->1->3)")

        // A tree prints parent:children, parents in ascending order, which is
        // what `mcclCommPlanDescription` hands a C caller.
        let tree = CollectivePlan.tree(root: 0, children: [0: [1, 2], 2: [3]])
        XCTAssertEqual(tree.description, "tree(root: 0, 0:1,2 2:3)")

        let hierarchical = CollectivePlan.hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 1)
        XCTAssertEqual(hierarchical.description, "hierarchical(islands: [0,1] [2,3], interIslandRoot: 1)")
    }

    func testPlansForEveryRegimePrintSomethingUseful() {
        // The three regimes the planner can land in, each rendered from a real
        // analysis rather than a hand-built plan.
        let uniform = Fabrics.uniformFast(nodeCount: 4)
        let small = TopologyPlanner.plan(for: uniform, messageBytes: 1 << 10)
        let large = TopologyPlanner.plan(for: uniform, messageBytes: 64 << 20)
        XCTAssertTrue(small.description.hasPrefix("tree("), small.description)
        XCTAssertTrue(large.description.hasPrefix("ring("), large.description)

        let mixed = TopologyPlanner.plan(for: Fabrics.twoIslandsBridged(), messageBytes: 64 << 20)
        XCTAssertTrue(mixed.description.hasPrefix("hierarchical("), mixed.description)
    }

    // MARK: - Compression text

    func testEveryCompressionSchemePrintsItsParameters() {
        XCTAssertEqual(WireCompression.none.description, "none")
        XCTAssertEqual(WireCompression.downcast.description, "downcast")
        XCTAssertEqual(WireCompression.int8Blockwise(blockSize: 128).description, "int8Blockwise(128)")
        XCTAssertEqual(WireCompression.topK(fraction: 0.01).description, "topK(0.01)")
        // The block size and fraction are the whole difference between two
        // otherwise identical schemes, so equality must see them.
        XCTAssertNotEqual(WireCompression.int8Blockwise(blockSize: 128),
                          WireCompression.int8Blockwise(blockSize: 256))
        XCTAssertNotEqual(WireCompression.topK(fraction: 0.01), WireCompression.topK(fraction: 0.02))
        XCTAssertNotEqual(WireCompression.none, WireCompression.downcast)
    }

    // MARK: - Analysis accessors

    func testAnalysisReportsWhetherItSawAMixedFabric() {
        let uniform = TopologyPlanner.analyze(Fabrics.uniformFast(nodeCount: 4))
        XCTAssertTrue(uniform.isProbed)
        XCTAssertFalse(uniform.isHeterogeneous)
        XCTAssertNil(uniform.islands)

        let mixed = TopologyPlanner.analyze(Fabrics.twoIslandsBridged())
        XCTAssertTrue(mixed.isProbed)
        XCTAssertTrue(mixed.isHeterogeneous, "two TB5 pairs on a 10GbE bridge is the mixed case")
        XCTAssertEqual(mixed.islands, [[0, 1], [2, 3]])

        let unprobed = TopologyPlanner.analyze(Topology.uniform(nodeCount: 4))
        XCTAssertFalse(unprobed.isProbed)
        XCTAssertFalse(unprobed.isHeterogeneous)
        XCTAssertNil(unprobed.treeRingCrossoverBytes)
    }

    func testTopologyReportsTheBestMeasurementInEitherDirection() {
        let nodes = (0..<3).map {
            Topology.Node(id: $0, hostname: "n\($0)", chip: "M4 Max", unifiedMemoryBytes: 0)
        }
        let topology = Topology(nodes: nodes, links: [
            // The same pair measured twice, once each way: keep the best of each.
            Topology.Link(from: 0, to: 1, kind: .thunderbolt5,
                          measuredBandwidth: 5.0e9, measuredLatency: 30e-6),
            Topology.Link(from: 1, to: 0, kind: .thunderbolt5,
                          measuredBandwidth: 5.4e9, measuredLatency: 21e-6),
            Topology.Link(from: 1, to: 2, kind: .ethernet, measuredBandwidth: 1.1e9),
        ])

        XCTAssertEqual(topology.bandwidth(between: 0, 1), 5.4e9, "bandwidth takes the maximum")
        XCTAssertEqual(topology.bandwidth(between: 1, 0), 5.4e9, "direction must not matter")
        XCTAssertEqual(topology.latency(between: 0, 1)!, 21e-6, accuracy: 1e-12,
                       "latency takes the minimum — scheduling noise only ever adds time")
        XCTAssertEqual(topology.latency(between: 1, 0)!, 21e-6, accuracy: 1e-12)
        XCTAssertNil(topology.latency(between: 1, 2), "an unmeasured RTT is nil, not zero")
        XCTAssertNil(topology.latency(between: 0, 2), "no link means no measurement")
        XCTAssertNil(topology.bandwidth(between: 0, 2))
    }

    // MARK: - Transport protocol defaults

    func testTransportProtocolSuppliesLoopbackDefaults() throws {
        // A caller that only wants "somewhere local, any port" should not have
        // to spell out 127.0.0.1 and a timeout.
        let transport = LoopbackTransport()
        let listener = try transport.listen()
        defer { listener.close() }
        XCTAssertEqual(listener.address.host, "127.0.0.1")
        XCTAssertGreaterThan(listener.address.port, 0)

        let accepted = expectation(description: "accepted")
        let box = Collector<Channel>()
        DispatchQueue.global().async {
            if let channel = try? listener.accept(timeout: 5) { box.set(0, channel) }
            accepted.fulfill()
        }
        let client = try transport.connect(to: listener.address)
        defer { client.close() }
        wait(for: [accepted], timeout: 10)
        box.get(0)?.close()
        XCTAssertNotNil(box.get(0))
    }

    // MARK: - Codec framing violations

    func testDecodeRejectsAPayloadThatDoesNotMatchItsHeader() throws {
        let elements = 8
        let destination = UnsafeMutableRawBufferPointer.allocate(byteCount: 64, alignment: 16)
        defer { destination.deallocate() }
        let payload = UnsafeMutableRawBufferPointer.allocate(byteCount: 256, alignment: 16)
        defer { payload.deallocate() }
        payload.initializeMemory(as: UInt8.self, repeating: 0)

        func decode(bytes: Int, codec: WireCodecID, blockSize: Int, dataType: DataType) throws {
            try WireCodec.decode(payload: payload.baseAddress!, payloadBytes: bytes,
                                 elementCount: elements, codec: codec, blockSize: blockSize,
                                 dataType: dataType, into: destination.baseAddress!)
        }

        // A truncated or over-long raw frame is a protocol violation, not a
        // partial read: a receiver never guesses what the sender meant.
        XCTAssertThrowsError(try decode(bytes: 31, codec: .raw, blockSize: 0, dataType: .float32)) {
            assertProtocolViolation($0, mentioning: "raw payload 31")
        }
        XCTAssertThrowsError(try decode(bytes: 15, codec: .fp16Downcast, blockSize: 0, dataType: .float32)) {
            assertProtocolViolation($0, mentioning: "fp16 payload 15")
        }
        // fp16 downcast only ever describes an fp32 buffer; anything else means
        // the two ranks disagree about the dtype in the frame.
        XCTAssertThrowsError(try decode(bytes: 16, codec: .fp16Downcast, blockSize: 0, dataType: .float16)) {
            assertProtocolViolation($0, mentioning: "decoded as")
        }
        XCTAssertThrowsError(try decode(bytes: 40, codec: .int8Blockwise, blockSize: 0, dataType: .float32)) {
            assertProtocolViolation($0, mentioning: "blockSize 0")
        }
        XCTAssertThrowsError(try decode(bytes: 5, codec: .int8Blockwise, blockSize: 4, dataType: .float32)) {
            assertProtocolViolation($0, mentioning: "int8 payload 5")
        }

        // The well-formed sizes still decode, so the guards are not merely
        // rejecting everything.
        XCTAssertNoThrow(try decode(bytes: 32, codec: .raw, blockSize: 0, dataType: .float32))
        XCTAssertNoThrow(try decode(bytes: 16, codec: .fp16Downcast, blockSize: 0, dataType: .float32))
        XCTAssertNoThrow(try decode(bytes: 2 * 4 + 8, codec: .int8Blockwise, blockSize: 4, dataType: .float32))
    }

    func testTopKIsRejectedAsAPerHopCodecWithAnActionableMessage() {
        XCTAssertThrowsError(try WireCodec(dataType: .float32, compression: .topK(fraction: 0.1))) { error in
            guard case MCCLError.unsupportedCompression(let what) = error else {
                return XCTFail("expected .unsupportedCompression, got \(error)")
            }
            XCTAssertTrue(what.contains("Communicator.allReduce"),
                          "the message must say what to call instead: \(what)")
        }
        XCTAssertThrowsError(try WireCodec(dataType: .float32, compression: .int8Blockwise(blockSize: 0))) { error in
            guard case MCCLError.invalidArgument = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
        }
    }

    private func assertProtocolViolation(_ error: Error, mentioning fragment: String) {
        guard case MCCLError.protocolViolation(let what) = error else {
            return XCTFail("expected .protocolViolation, got \(error)")
        }
        XCTAssertTrue(what.contains(fragment), "'\(what)' should mention '\(fragment)'")
    }
}
