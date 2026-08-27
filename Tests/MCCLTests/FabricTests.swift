import XCTest
@testable import MCCL

/// Bring-up failures, group constructors, and the plan sanitiser.
///
/// A mesh either comes up completely or not at all: a half-connected fabric
/// would deadlock the first collective rather than fail, which is the worst
/// possible outcome on a cluster. These tests drive each way bring-up can go
/// wrong and check it surfaces as an error instead.
final class FabricTests: XCTestCase {

    // MARK: - Argument validation

    func testBootstrapRejectsARankOutsideTheWorld() throws {
        let transport = LoopbackTransport()
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        XCTAssertThrowsError(try MeshFabric.bootstrap(
            rank: 4, worldSize: 4, addresses: [listener.address],
            listener: listener, transport: transport, timeout: 1)
        ) { error in
            guard case MCCLError.rankOutOfRange(let rank, let world) = error else {
                return XCTFail("expected .rankOutOfRange, got \(error)")
            }
            XCTAssertEqual(rank, 4)
            XCTAssertEqual(world, 4)
        }
    }

    func testBootstrapRejectsAnAddressTableOfTheWrongSize() throws {
        let transport = LoopbackTransport()
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        XCTAssertThrowsError(try MeshFabric.bootstrap(
            rank: 0, worldSize: 4, addresses: [listener.address],
            listener: listener, transport: transport, timeout: 1)
        ) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("expected 4 peer advertisements, got 1"), what)
        }
    }

    func testMakeGroupRejectsAnEmptyWorld() {
        XCTAssertThrowsError(try MeshFabric.makeGroup(worldSize: 0, transport: LoopbackTransport(),
                                                      host: "loopback")) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("worldSize"), what)
        }
    }

    func testAFabricWithAMissingChannelSaysSoRatherThanCrashing() throws {
        // A fabric is only ever built by `bootstrap`, but the accessor is the
        // last line of defence: the collectives index it by rank on every hop.
        let fabric = MeshFabric(rank: 0, worldSize: 3, channels: [:], listener: nil)
        defer { fabric.shutdown() }
        XCTAssertThrowsError(try fabric.channel(to: 1)) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("rank 0 has no channel to 1"), what)
        }
        XCTAssertThrowsError(try fabric.channel(to: 0), "no rank has a channel to itself")
        XCTAssertThrowsError(try fabric.channel(to: -1))
        XCTAssertThrowsError(try fabric.channel(to: 3))
        // Shutdown is idempotent; a communicator may be torn down twice.
        fabric.shutdown()
        fabric.shutdown()
    }

    // MARK: - Handshake violations

    func testBootstrapRejectsADialerThatDoesNotAnnounceItself() throws {
        let error = try bootstrapRejection { header in
            var forged = header
            forged.tag = 0x1234
            return forged
        }
        guard case MCCLError.protocolViolation(let what) = error else {
            return XCTFail("expected .protocolViolation, got \(error)")
        }
        XCTAssertTrue(what.contains("expected bootstrap handshake"), what)
    }

    func testBootstrapRejectsADialerClaimingAnImpossibleRank() throws {
        // The ordering rule is "higher rank dials lower", so rank 0 can only
        // ever be dialled by a rank above it. A 0 here means a confused peer.
        let error = try bootstrapRejection { header in
            var forged = header
            forged.tag = MeshFabric.handshakeTag
            forged.elementCount = 0
            return forged
        }
        guard case MCCLError.protocolViolation(let what) = error else {
            return XCTFail("expected .protocolViolation, got \(error)")
        }
        XCTAssertTrue(what.contains("impossible rank 0"), what)
    }

    /// Runs rank 0 of a two-rank bootstrap, dials it with one forged handshake
    /// frame, and returns what rank 0 threw.
    private func bootstrapRejection(_ makeHeader: (WireHeader) -> WireHeader) throws -> Error {
        let transport = LoopbackTransport()
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        let box = Collector<Error>()
        let finished = expectation(description: "rank 0 returned")
        DispatchQueue.global().async {
            do {
                _ = try MeshFabric.bootstrap(
                    rank: 0, worldSize: 2,
                    addresses: [listener.address, PeerAddress(host: "loopback", port: 999_999)],
                    listener: listener, transport: transport, timeout: 5)
            } catch {
                box.set(0, error)
            }
            finished.fulfill()
        }

        let channel = try transport.connect(to: listener.address, timeout: 5)
        defer { channel.close() }
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount)
        try channel.sendFrame(makeHeader(WireHeader()), payload: nil, payloadBytes: 0, using: scratch)

        wait(for: [finished], timeout: 10)
        guard let error = box.get(0) else {
            throw MCCLError.protocolViolation("rank 0 accepted a handshake it should have rejected")
        }
        return error
    }

    // MARK: - Partial bring-up

    func testAListenerThatCannotBindAbortsTheWholeGroup() {
        // Half a world is worse than none: the ranks that did come up would
        // block forever on the ones that did not.
        let transport = FlakyTransport(successfulListens: 2)
        XCTAssertThrowsError(try MeshFabric.makeGroup(
            worldSize: 3, transport: transport, host: "loopback", timeout: 1)
        ) { error in
            guard case MCCLError.socketFailure = error else {
                return XCTFail("expected the bind failure to propagate, got \(error)")
            }
        }
    }

    func testARankThatCannotDialAbortsTheWholeGroup() {
        let transport = FlakyTransport(refuseConnections: true)
        XCTAssertThrowsError(try MeshFabric.makeGroup(
            worldSize: 3, transport: transport, host: "loopback", timeout: 0.5)
        ) { error in
            // Rank 0 times out waiting for dials that never come; the ranks
            // above it fail on connect. Either is a legitimate first failure.
            switch error {
            case MCCLError.timedOut, MCCLError.connectionClosed:
                break
            default:
                XCTFail("expected .timedOut or .connectionClosed, got \(error)")
            }
        }
    }

    // MARK: - Group constructors

    func testTCPGroupBringsUpAWorkingWorldOverRealSockets() async throws {
        let worldSize = 4
        let count = 256
        let world = try Communicator.tcpGroup(worldSize: worldSize)
        defer { world.forEach { $0.shutdown() } }

        XCTAssertEqual(world.map(\.rank), [0, 1, 2, 3])
        XCTAssertEqual(Set(world.map(\.worldSize)), [worldSize])
        XCTAssertEqual(world[0].topology.nodes.count, worldSize)

        try await withThrowingTaskGroup(of: Void.self) { tasks in
            for comm in world {
                tasks.addTask {
                    let buffer = Buf.allocate(count: count, dataType: .float32)
                    defer { buffer.deallocate() }
                    Buf.fill(buffer, count: count, dataType: .float32) { Float(comm.rank + 1) * Float($0 % 3 + 1) }
                    try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)
                    for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                        XCTAssertEqual(value, Float(10 * (i % 3 + 1)), "rank \(comm.rank) index \(i)")
                    }
                }
            }
            try await tasks.waitForAll()
        }
    }

    func testLoopbackGroupBringsUpAWorkingWorldWithNoSocketsAtAll() async throws {
        let worldSize = 3
        let count = 64
        let world = try Communicator.loopbackGroup(worldSize: worldSize)
        defer { world.forEach { $0.shutdown() } }

        XCTAssertEqual(world.map(\.rank), [0, 1, 2])
        try await withThrowingTaskGroup(of: Void.self) { tasks in
            for comm in world {
                tasks.addTask {
                    let buffer = Buf.allocate(count: count, dataType: .float32)
                    defer { buffer.deallocate() }
                    Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank + 1) }
                    try await comm.allReduce(buffer, count: count, dataType: .float32, op: .max)
                    XCTAssertEqual(Buf.read(buffer, count: count, dataType: .float32),
                                   [Float](repeating: 3, count: count))
                }
            }
            try await tasks.waitForAll()
        }
    }

    // MARK: - Averaging integer payloads

    func testAveragingAnIntegerPayloadRoundsRatherThanTruncating() async throws {
        // `.avg` is a sum followed by one scaling pass. On an integer dtype
        // that pass has to land on an integer, and it rounds to nearest rather
        // than truncating — otherwise every averaged all-reduce would drift
        // systematically downwards over a training run.
        try await Ranks.run(worldSize: 4) { comm in
            let count = 32

            let wide = Buf.allocate(count: count, dataType: .int32)
            defer { wide.deallocate() }
            Buf.fill(wide, count: count, dataType: .int32) { Float((comm.rank + 1) * ($0 % 4 + 1) * 100) }
            try await comm.allReduce(wide, count: count, dataType: .int32, op: .avg)
            for (i, value) in Buf.read(wide, count: count, dataType: .int32).enumerated() {
                // (1+2+3+4)·100·(i%4+1) / 4 = 250·(i%4+1), exact.
                XCTAssertEqual(value, Float(250 * (i % 4 + 1)), "int32 avg at \(i)")
            }

            let narrow = Buf.allocate(count: count, dataType: .int8)
            defer { narrow.deallocate() }
            Buf.fill(narrow, count: count, dataType: .int8) { _ in Float(comm.rank + 1) }
            try await comm.allReduce(narrow, count: count, dataType: .int8, op: .avg)
            // 1+2+3+4 = 10 over four ranks is 2.5, which rounds to 3.
            XCTAssertEqual(Buf.read(narrow, count: count, dataType: .int8),
                           [Float](repeating: 3, count: count))
        }
    }

    // MARK: - Plan sanitising

    func testAPlanThatDoesNotNameThisWorldDegradesToTheRing() {
        let worldSize = 4
        let comm = Communicator(rank: 0, worldSize: worldSize, topology: Topology.uniform(nodeCount: worldSize))
        let fallback = CollectivePlan.ring(order: Array(0..<worldSize))

        // A topology map loaded from disk may describe a stale or larger
        // cluster. Every one of these has to become the plain ring, because a
        // plan naming a rank that is not here would hang the collective.
        let broken: [CollectivePlan] = [
            .ring(order: [0, 1, 2, 9]),                      // a rank that left
            .ring(order: [0, 1, 2]),                         // a rank that is missing
            .tree(root: 9, children: [:]),                   // root outside the world
            .tree(root: 0, children: [9: [1]]),              // parent outside the world
            .tree(root: 0, children: [0: [1, 2], 1: [2]]),   // rank 2 has two parents
            .tree(root: 0, children: [0: [1, 9]]),           // child outside the world
            .tree(root: 0, children: [0: [1]]),              // does not reach everyone
            .hierarchical(islands: [[0, 1], [2]], interIslandRoot: 0),      // rank 3 unplaced
            .hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 9),   // leader outside
            .hierarchical(islands: [[0, 1], [2, 3], []], interIslandRoot: 0), // empty island
        ]
        for plan in broken {
            comm.planOverride = plan
            XCTAssertEqual(comm.plan(messageBytes: 1 << 20), fallback,
                           "\(plan) should have degraded to the ring")
        }

        // The well-formed plans survive untouched.
        let intact: [CollectivePlan] = [
            .ring(order: [3, 1, 0, 2]),
            .tree(root: 0, children: [0: [1, 2], 2: [3]]),
            .hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 1),
        ]
        for plan in intact {
            comm.planOverride = plan
            XCTAssertEqual(comm.plan(messageBytes: 1 << 20), plan, "\(plan) is executable as written")
        }
    }
}

// MARK: - Test doubles

/// A transport that fails where a real cluster fails: a port it cannot bind, or
/// a peer it cannot reach. Wraps the loopback transport so everything that does
/// succeed behaves exactly as it normally would.
private final class FlakyTransport: Transport, @unchecked Sendable {
    let name = "flaky"
    private let inner = LoopbackTransport()
    private let lock = NSLock()
    private var listensLeft: Int
    private let refuseConnections: Bool

    init(successfulListens: Int = .max, refuseConnections: Bool = false) {
        self.listensLeft = successfulListens
        self.refuseConnections = refuseConnections
    }

    func listen(host: String, port: Int) throws -> Listener {
        lock.lock()
        let allowed = listensLeft > 0
        listensLeft -= 1
        lock.unlock()
        guard allowed else { throw MCCLError.socketFailure("bind(\(host):\(port))", errno: EADDRINUSE) }
        return try inner.listen(host: host, port: port)
    }

    func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel {
        guard !refuseConnections else { throw MCCLError.connectionClosed }
        return try inner.connect(to: address, timeout: timeout)
    }
}
