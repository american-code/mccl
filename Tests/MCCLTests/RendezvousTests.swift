import XCTest
@testable import MCCL

/// Bring-up from a single token, as `mcclCommInitRank` does it.
final class RendezvousTests: XCTestCase {

    // MARK: - Token encoding

    func testUniqueIDRoundTripsThroughText() {
        let id = UniqueID(nonce: 0xDEAD_BEEF_1234_5678,
                          address: PeerAddress(host: "192.168.1.10", port: 29500))
        XCTAssertEqual(id.text, "mccl1:deadbeef12345678:192.168.1.10:29500")
        XCTAssertEqual(UniqueID(text: id.text), id)
    }

    func testUniqueIDHandlesIPv6HostsWithColons() {
        let id = UniqueID(nonce: 1, address: PeerAddress(host: "fe80::1c6f:911d", port: 7000))
        let restored = UniqueID(text: id.text)
        XCTAssertEqual(restored?.address.host, "fe80::1c6f:911d",
                       "the host is everything between the nonce and the final colon")
        XCTAssertEqual(restored?.address.port, 7000)
        XCTAssertEqual(restored?.nonce, 1)
    }

    func testMalformedTokensAreRejected() {
        for text in ["", "mccl1", "mccl1:zz:host:1", "mccl2:01:host:1",
                     "mccl1:01:host:notaport", "mccl1:01::7000"] {
            XCTAssertNil(UniqueID(text: text), "'\(text)' should not parse")
        }
    }

    // MARK: - Address exchange

    func testFourRanksFindEachOtherThroughRankZero() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        let worldSize = 4

        var listeners: [Listener] = []
        for _ in 0..<worldSize { listeners.append(try transport.listen(host: "loopback", port: 0)) }
        defer { listeners.forEach { $0.close() } }

        let tables = ResultSlots<[PeerAddress]>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "rendezvous.test", attributes: .concurrent)
        for rank in 0..<worldSize {
            let listener = listeners[rank]
            queue.async(group: group) {
                do {
                    tables.set(rank, .success(try Rendezvous.join(
                        id: id, rank: rank, worldSize: worldSize,
                        dataListener: listener, transport: transport, timeout: 20)))
                } catch {
                    tables.set(rank, .failure(error))
                }
            }
        }
        group.wait()

        let collected = try tables.collect()
        let expected = listeners.map { PeerAddress(host: "loopback", port: $0.address.port) }
        for (rank, table) in collected.enumerated() {
            XCTAssertEqual(table, expected, "rank \(rank) got a different world")
        }
    }

    func testASingleRankNeedsNoExchange() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        let table = try Rendezvous.join(id: id, rank: 0, worldSize: 1,
                                        dataListener: listener, transport: transport)
        XCTAssertEqual(table.count, 1)
        XCTAssertEqual(table[0].port, listener.address.port)
    }

    func testRankZeroTimesOutRatherThanHangingForeverOnAMissingRank() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        // World size 3, nobody else shows up.
        XCTAssertThrowsError(try Rendezvous.join(
            id: id, rank: 0, worldSize: 3, dataListener: listener,
            transport: transport, timeout: 0.4)
        ) { error in
            guard case MCCLError.timedOut(let message) = error else {
                return XCTFail("expected .timedOut, got \(error)")
            }
            XCTAssertTrue(message.contains("of 3 ranks"), "the message should say who is missing: \(message)")
        }
    }

    func testDiscardReleasesTheListener() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        Rendezvous.discard(id)
        // The loopback registry entry is gone, so a dial can no longer find it.
        XCTAssertThrowsError(try transport.connect(to: id.address, timeout: 0.2))
    }

    func testAdvertisedHostSubstitutesForAWildcardBind() {
        XCTAssertEqual(
            Rendezvous.resolveAdvertisedHost(bound: "192.168.1.5", rendezvous: "192.168.1.9"),
            "192.168.1.5", "an explicit bind address is already what peers should dial")
        XCTAssertEqual(
            Rendezvous.resolveAdvertisedHost(bound: "0.0.0.0", rendezvous: "127.0.0.1"),
            "127.0.0.1", "a loopback job stays on loopback")
        // A wildcard bind against a remote rank 0 must resolve to something
        // routable; on a machine with no cable at all that is still loopback.
        let resolved = Rendezvous.resolveAdvertisedHost(bound: "0.0.0.0", rendezvous: "10.1.2.3")
        XCTAssertFalse(resolved.isEmpty)
        XCTAssertNotEqual(resolved, "0.0.0.0")
    }

    // MARK: - Communicator.join

    func testCommunicatorJoinBringsUpAWorkingWorldOverRealSockets() async throws {
        let id = try Rendezvous.createUniqueID(host: "127.0.0.1")
        let worldSize = 3
        let count = 512

        let comms = ResultSlots<Communicator>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "join.test", attributes: .concurrent)
        for rank in 0..<worldSize {
            queue.async(group: group) {
                do {
                    comms.set(rank, .success(try Communicator.join(
                        uniqueID: id, rank: rank, worldSize: worldSize, timeout: 30)))
                } catch {
                    comms.set(rank, .failure(error))
                }
            }
        }
        group.wait()

        let world = try comms.collect()
        defer { world.forEach { $0.shutdown() } }
        XCTAssertEqual(world.map(\.rank).sorted(), [0, 1, 2])
        XCTAssertEqual(Set(world.map(\.worldSize)), [worldSize])

        // A world that bootstrapped is only useful if it reduces.
        try await withThrowingTaskGroup(of: Void.self) { tasks in
            for comm in world {
                tasks.addTask {
                    let buffer = Buf.allocate(count: count, dataType: .float32)
                    defer { buffer.deallocate() }
                    Buf.fill(buffer, count: count, dataType: .float32) { Float((comm.rank + 1) * ($0 % 5 + 1)) }
                    try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)
                    for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                        XCTAssertEqual(value, Float(6 * (i % 5 + 1)), "rank \(comm.rank) index \(i)")
                    }
                }
            }
            try await tasks.waitForAll()
        }
    }

    func testJoinRejectsARankOutsideTheWorld() throws {
        let id = try Rendezvous.createUniqueID(transport: LoopbackTransport(), host: "loopback")
        defer { Rendezvous.discard(id) }
        XCTAssertThrowsError(try Communicator.join(uniqueID: id, rank: 4, worldSize: 4,
                                                   transport: LoopbackTransport())) { error in
            guard case MCCLError.rankOutOfRange = error else {
                return XCTFail("expected .rankOutOfRange, got \(error)")
            }
        }
    }

    func testTwoConcurrentJobsGetDistinctTokens() throws {
        let a = try Rendezvous.createUniqueID(transport: LoopbackTransport(), host: "loopback")
        let b = try Rendezvous.createUniqueID(transport: LoopbackTransport(), host: "loopback")
        defer { Rendezvous.discard(a); Rendezvous.discard(b) }
        XCTAssertNotEqual(a.nonce, b.nonce, "the nonce is what keeps two jobs apart")
        XCTAssertNotEqual(a.address.port, b.address.port)
    }
}
