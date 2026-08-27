import XCTest
@testable import MCCL

/// The rendezvous wire protocol under abuse.
///
/// `RendezvousTests` proves the happy path: n ranks announce, rank 0 hands the
/// table back. Everything here is a rank that lies, a launcher that gets the
/// world size wrong, or two jobs colliding on one host — the situations where a
/// silent success would corrupt a whole training run. Each one has to be a
/// typed error naming the offending rank, since a cluster operator's only
/// evidence is that string.
final class RendezvousProtocolTests: XCTestCase {

    /// Mirrors the private `Rendezvous.Hello` so a test can forge one.
    private struct Hello: Codable {
        var nonce: UInt64
        var rank: Int
        var worldSize: Int
        var address: PeerAddress
    }

    /// The two rendezvous frame tags, repeated here on purpose: if they ever
    /// change, this file should fail and force the decision to be deliberate.
    private static let helloTag: UInt32 = 0xBEE1
    private static let tableTag: UInt32 = 0xBEE2

    // MARK: - Argument validation

    func testJoinRejectsANonPositiveWorldSize() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        defer { Rendezvous.discard(id) }
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        XCTAssertThrowsError(try Rendezvous.join(id: id, rank: 0, worldSize: 0,
                                                 dataListener: listener, transport: transport)) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("worldSize"), what)
        }
    }

    func testJoinRejectsARankOutsideTheWorld() throws {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        defer { Rendezvous.discard(id) }
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        for rank in [-1, 4, 99] {
            XCTAssertThrowsError(try Rendezvous.join(id: id, rank: rank, worldSize: 4,
                                                     dataListener: listener, transport: transport)) { error in
                guard case MCCLError.rankOutOfRange(let got, let world) = error else {
                    return XCTFail("expected .rankOutOfRange for rank \(rank), got \(error)")
                }
                XCTAssertEqual(got, rank)
                XCTAssertEqual(world, 4)
            }
        }
    }

    // MARK: - Token surface

    func testUniqueIDPrintsAsItsToken() {
        let id = UniqueID(nonce: 0x0102_0304_0506_0708,
                          address: PeerAddress(host: "10.0.0.9", port: 29500))
        XCTAssertEqual(id.description, id.text)
        XCTAssertEqual("\(id)", "mccl1:0102030405060708:10.0.0.9:29500")
    }

    func testAWildcardBindAdvertisesSomethingAPeerCanActuallyDial() throws {
        // The default host is 0.0.0.0, which tells a peer nothing — the token
        // must carry a real local address instead.
        let id = try Rendezvous.createUniqueID()
        defer { Rendezvous.discard(id) }
        XCTAssertNotEqual(id.address.host, "0.0.0.0")
        XCTAssertNotEqual(id.address.host, "::")
        XCTAssertFalse(id.address.host.isEmpty)
        XCTAssertGreaterThan(id.address.port, 0)
        XCTAssertEqual(UniqueID(text: id.text), id, "the advertised token must round-trip")
    }

    func testMCCLHostOverridesWhateverTheInterfacesSay() {
        setenv("MCCL_HOST", "10.42.0.1", 1)
        defer { unsetenv("MCCL_HOST") }
        // The override wins even when the bind address was explicit, because an
        // operator setting it knows something about NAT that mccl does not.
        XCTAssertEqual(Rendezvous.resolveAdvertisedHost(bound: "0.0.0.0", rendezvous: "10.1.2.3"),
                       "10.42.0.1")
        XCTAssertEqual(Rendezvous.resolveAdvertisedHost(bound: "192.168.0.5", rendezvous: "127.0.0.1"),
                       "10.42.0.1")
    }

    // MARK: - Rank 0 rejects malformed check-ins

    func testRankZeroRejectsAFrameThatIsNotAHello() throws {
        let error = try rankZeroError(worldSize: 2) { _ in (0x1234, Data()) }
        guard case MCCLError.protocolViolation(let what) = error else {
            return XCTFail("expected .protocolViolation, got \(error)")
        }
        XCTAssertTrue(what.contains("expected hello"), what)
    }

    func testRankZeroRejectsARankBelongingToAnotherJob() throws {
        let error = try rankZeroError(worldSize: 2) { id in
            // Same host and port, different nonce: two jobs launched on one box.
            let hello = Hello(nonce: id.nonce &+ 1, rank: 1, worldSize: 2,
                              address: PeerAddress(host: "loopback", port: 1))
            return (Self.helloTag, try JSONEncoder().encode(hello))
        }
        guard case MCCLError.protocolViolation(let what) = error else {
            return XCTFail("expected .protocolViolation, got \(error)")
        }
        XCTAssertTrue(what.contains("different job"), what)
        XCTAssertTrue(what.contains("rank 1"), "the message must name the rank: \(what)")
    }

    func testRankZeroRejectsARankThatCannotExistInThisWorld() throws {
        // rank 0 is rank 0's own slot; nobody may announce into it.
        let zero = try rankZeroError(worldSize: 2) { id in
            let hello = Hello(nonce: id.nonce, rank: 0, worldSize: 2,
                              address: PeerAddress(host: "loopback", port: 1))
            return (Self.helloTag, try JSONEncoder().encode(hello))
        }
        guard case MCCLError.protocolViolation(let what) = zero else {
            return XCTFail("expected .protocolViolation, got \(zero)")
        }
        XCTAssertTrue(what.contains("impossible or duplicate rank 0"), what)

        // A rank past the end of the world is the same mistake in the other
        // direction — usually a launcher that disagrees about --nranks.
        let past = try rankZeroError(worldSize: 2) { id in
            let hello = Hello(nonce: id.nonce, rank: 7, worldSize: 2,
                              address: PeerAddress(host: "loopback", port: 1))
            return (Self.helloTag, try JSONEncoder().encode(hello))
        }
        guard case MCCLError.protocolViolation(let what) = past else {
            return XCTFail("expected .protocolViolation, got \(past)")
        }
        XCTAssertTrue(what.contains("rank 7"), what)
        XCTAssertTrue(what.contains("world size 2"), what)
    }

    func testRankZeroRejectsARankThatDisagreesAboutTheWorldSize() throws {
        let error = try rankZeroError(worldSize: 4) { id in
            let hello = Hello(nonce: id.nonce, rank: 1, worldSize: 8,
                              address: PeerAddress(host: "loopback", port: 1))
            return (Self.helloTag, try JSONEncoder().encode(hello))
        }
        guard case MCCLError.invalidArgument(let what) = error else {
            return XCTFail("expected .invalidArgument, got \(error)")
        }
        XCTAssertTrue(what.contains("world size 8"), what)
        XCTAssertTrue(what.contains("rank 0 says 4"),
                      "both sides of the disagreement belong in the message: \(what)")
    }

    // MARK: - A joining rank rejects a malformed table

    func testAJoiningRankRejectsAReplyThatIsNotTheAddressTable() throws {
        let transport = LoopbackTransport()
        let fakeRankZero = try transport.listen(host: "loopback", port: 0)
        defer { fakeRankZero.close() }
        let id = UniqueID(nonce: 0xABCD_EF01, address: fakeRankZero.address)

        let mine = try transport.listen(host: "loopback", port: 0)
        defer { mine.close() }

        let box = Collector<Error>()
        let finished = expectation(description: "rank 1 returned")
        DispatchQueue.global().async {
            do {
                _ = try Rendezvous.join(id: id, rank: 1, worldSize: 2,
                                        dataListener: mine, transport: transport, timeout: 5)
            } catch {
                box.set(1, error)
            }
            finished.fulfill()
        }

        let channel = try fakeRankZero.accept(timeout: 5)
        defer { channel.close() }
        let scratch = ScratchBuffer(capacity: 4096)
        let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 4096)
        XCTAssertEqual(header.tag, Self.helloTag, "a joining rank must lead with a hello")

        // What rank 1 announced is what rank 0 would have put in the table.
        let hello = try JSONDecoder().decode(
            Hello.self, from: Data(bytes: scratch.base, count: Int(header.payloadBytes)))
        XCTAssertEqual(hello.nonce, id.nonce)
        XCTAssertEqual(hello.rank, 1)
        XCTAssertEqual(hello.worldSize, 2)
        XCTAssertEqual(hello.address.port, mine.address.port,
                       "a rank advertises the data listener it already bound")

        // Reply with something that is not the table.
        try sendFrame(channel, tag: Self.tableTag &+ 9, payload: Data("nope".utf8))

        wait(for: [finished], timeout: 10)
        guard case .some(MCCLError.protocolViolation(let what)) = box.get(1) as? MCCLError else {
            return XCTFail("expected .protocolViolation, got \(String(describing: box.get(1)))")
        }
        XCTAssertTrue(what.contains("expected the address table"), what)
    }

    // MARK: - A token whose listener this process did not create

    func testRankZeroRebindsATokenItDidNotCreate() throws {
        // `mcclGetUniqueId` may run in a launcher and the token be handed to a
        // different process, which then has to rebind the port itself. Discard
        // releases the listener without consuming the token, which is exactly
        // the state that process would see.
        let id = try Rendezvous.createUniqueID(host: "127.0.0.1")
        Rendezvous.discard(id)

        let worldSize = 2
        var listeners: [Listener] = []
        for _ in 0..<worldSize { listeners.append(try TCPTransport().listen(host: "127.0.0.1", port: 0)) }
        defer { listeners.forEach { $0.close() } }

        let tables = ResultSlots<[PeerAddress]>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "rendezvous.rebind", attributes: .concurrent)
        for rank in 0..<worldSize {
            let listener = listeners[rank]
            queue.async(group: group) {
                do {
                    tables.set(rank, .success(try Rendezvous.join(
                        id: id, rank: rank, worldSize: worldSize,
                        dataListener: listener, transport: TCPTransport(), timeout: 20)))
                } catch {
                    tables.set(rank, .failure(error))
                }
            }
        }
        group.wait()

        let collected = try tables.collect()
        XCTAssertEqual(collected[0], collected[1], "both ranks must agree on the world")
        XCTAssertEqual(collected[0].map(\.port), listeners.map(\.address.port))
    }

    // MARK: - Helpers

    /// Runs rank 0's half of the rendezvous in the background, dials it once
    /// with the frame `make` produces, and returns whatever rank 0 threw.
    private func rankZeroError(
        worldSize: Int,
        _ make: (UniqueID) throws -> (UInt32, Data)
    ) throws -> Error {
        let transport = LoopbackTransport()
        let id = try Rendezvous.createUniqueID(transport: transport, host: "loopback")
        let listener = try transport.listen(host: "loopback", port: 0)
        defer { listener.close() }

        let box = Collector<Error>()
        let finished = expectation(description: "rank 0 returned")
        DispatchQueue.global().async {
            do {
                _ = try Rendezvous.join(id: id, rank: 0, worldSize: worldSize,
                                        dataListener: listener, transport: transport, timeout: 5)
            } catch {
                box.set(0, error)
            }
            finished.fulfill()
        }

        // The listener was bound by createUniqueID, so this dial cannot race
        // rank 0 reaching its accept loop.
        let channel = try transport.connect(to: id.address, timeout: 5)
        defer { channel.close() }
        let (tag, payload) = try make(id)
        try sendFrame(channel, tag: tag, payload: payload)

        wait(for: [finished], timeout: 10)
        guard let error = box.get(0) else {
            throw MCCLError.protocolViolation("rank 0 accepted a frame it should have rejected")
        }
        return error
    }

    private func sendFrame(_ channel: Channel, tag: UInt32, payload: Data) throws {
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount + max(payload.count, 1))
        var header = WireHeader()
        header.tag = tag
        header.dataType = DataType.int8.wireCode
        header.elementCount = UInt32(payload.count)
        payload.withUnsafeBytes { source in
            if !payload.isEmpty {
                (scratch.base + WireHeader.byteCount).copyMemory(
                    from: source.baseAddress!, byteCount: payload.count)
            }
        }
        try channel.sendPreparedFrame(header, payloadBytes: payload.count, in: scratch)
    }
}
