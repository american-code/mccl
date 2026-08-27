import XCTest
@testable import MCCL

/// Per-pair addressing: what a rank advertises, how a dialer chooses among a
/// peer's addresses, and what happens when an announcement does not match where
/// it came from.
///
/// The interesting fabric — a Thunderbolt island bridged to a laptop over Wi-Fi
/// — cannot be built on one machine, so these split into two halves. The
/// selection rule is exercised against synthetic interface tables, where every
/// media combination can be written down exactly. The *mechanism* is exercised
/// over real loopback TCP with deliberately dead addresses mixed in, which is
/// the property that actually matters: an advertisement whose first entry is
/// unreachable must still produce a world.
final class PeerAdvertisementTests: XCTestCase {

    // MARK: - Building an advertisement

    /// `mask` matters: subnet membership is how a dialer works out which of its
    /// own interfaces an advertised address would leave on. The default /24 is
    /// an ordinary LAN.
    private func interface(
        _ name: String, _ media: InterfaceMedia, _ addresses: [String],
        mask: [UInt8] = [255, 255, 255, 0], up: Bool = true
    ) -> NetworkInterface {
        NetworkInterface(
            name: name, media: media, displayName: nil, isUp: up,
            isLoopback: media == .loopback, linkSpeedBitsPerSecond: 0,
            addresses: addresses.map { text in
                let parsed = NetworkInterfaces.parse(literal: text)!
                return InterfaceAddress(text: text, isIPv6: parsed.isIPv6,
                                        octets: parsed.octets,
                                        mask: parsed.isIPv6 ? parsed.mask : mask)
            })
    }

    /// The /16 a Thunderbolt bridge self-assigns — the subnet that cannot tell
    /// three TB ports apart.
    private func linkLocalInterface(_ name: String, _ addresses: [String]) -> NetworkInterface {
        interface(name, .thunderbolt, addresses, mask: [255, 255, 0, 0])
    }

    func testLocalAdvertisementCarriesEveryUsableAddressFastestFirst() {
        let advertisement = PeerAdvertisement.local(rank: 2, port: 9000, interfaces: [
            interface("en0", .wifi, ["192.168.1.5"]),
            interface("en1", .ethernet, ["10.0.0.5"]),
            interface("en4", .thunderbolt, ["169.254.152.222"]),
            interface("lo0", .loopback, ["127.0.0.1"]),
        ])
        XCTAssertEqual(advertisement.rank, 2)
        XCTAssertEqual(advertisement.hosts, ["169.254.152.222", "10.0.0.5", "192.168.1.5"],
                       "thunderbolt, then ethernet, then wi-fi; loopback is not advertised")
        XCTAssertTrue(advertisement.endpoints.allSatisfy { $0.address.port == 9000 },
                      "one listener means one port on every address")
        XCTAssertEqual(advertisement.endpoints.map(\.media), [.thunderbolt, .ethernet, .wifi])
    }

    func testASelfAssignedAddressCountsOnlyOnThunderbolt() {
        // 169.254/16 is how a point-to-point TB bridge with no DHCP server
        // always looks, and how an Ethernet port that failed to configure looks.
        // The same number means opposite things, so the media decides.
        let advertisement = PeerAdvertisement.local(rank: 0, port: 1, interfaces: [
            interface("en1", .ethernet, ["169.254.9.9"]),
            interface("en0", .wifi, ["192.168.1.5"]),
            interface("en4", .thunderbolt, ["169.254.152.222"]),
        ])
        XCTAssertEqual(advertisement.hosts, ["169.254.152.222", "192.168.1.5"])
    }

    func testADownInterfaceIsNotAdvertised() {
        let advertisement = PeerAdvertisement.local(rank: 0, port: 1, interfaces: [
            interface("en5", .ethernet, ["10.0.0.9"], up: false),
            interface("en0", .wifi, ["192.168.1.5"]),
        ])
        XCTAssertEqual(advertisement.hosts, ["192.168.1.5"])
    }

    func testExplicitHostsAreClassifiedAgainstTheLocalTable() {
        let table = [
            interface("en0", .wifi, ["192.168.1.5"]),
            interface("en4", .thunderbolt, ["169.254.152.222"]),
        ]
        let advertisement = PeerAdvertisement.explicit(
            rank: 1, hosts: ["192.168.1.5", "169.254.152.222", "10.9.9.9"],
            port: 700, interfaces: table)
        XCTAssertEqual(advertisement.hosts, ["192.168.1.5", "169.254.152.222", "10.9.9.9"],
                       "an explicit list keeps the operator's order")
        XCTAssertEqual(advertisement.endpoints.map(\.media), [.wifi, .thunderbolt, .other],
                       "a host this machine does not own is tagged .other, not guessed at")
    }

    func testExplicitHostsDropDuplicates() {
        let advertisement = PeerAdvertisement.explicit(
            rank: 0, hosts: ["10.0.0.1", "10.0.0.1"], port: 5, interfaces: [])
        XCTAssertEqual(advertisement.hosts, ["10.0.0.1"])
    }

    // MARK: - The selection rule

    func testDialerRanksPathsByItsOwnEgressInterface() {
        // The dialer is on Thunderbolt and Wi-Fi. The peer offers one address
        // on each. Thunderbolt must come first — and it must come first because
        // of the *dialer's* interface, not the peer's claim about its own.
        let local = [
            interface("en0", .wifi, ["192.168.1.5"]),
            linkLocalInterface("en4", ["169.254.152.222"]),
        ]
        let peer = PeerAdvertisement(rank: 0, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "192.168.1.250", port: 7), media: .wifi),
            PeerEndpoint(address: PeerAddress(host: "169.254.23.203", port: 7), media: .thunderbolt),
        ])
        let ordered = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), ["169.254.23.203", "192.168.1.250"])
        XCTAssertEqual(ordered[0].egressInterface, "en4")
        XCTAssertEqual(ordered[0].egressMedia, .thunderbolt)
    }

    func testAPeersOwnMediaClaimCannotPromoteAPath() {
        // The peer insists both of its addresses are Thunderbolt. Only one of
        // them is on the dialer's TB subnet, and that is the one that wins:
        // an unverifiable claim never outranks an observed egress interface.
        let local = [
            interface("en0", .wifi, ["192.168.1.5"]),
            linkLocalInterface("en4", ["169.254.152.222"]),
        ]
        let peer = PeerAdvertisement(rank: 0, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "192.168.1.250", port: 7), media: .thunderbolt),
            PeerEndpoint(address: PeerAddress(host: "169.254.23.203", port: 7), media: .thunderbolt),
        ])
        let ordered = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), ["169.254.23.203", "192.168.1.250"])
    }

    func testAnAddressNoLocalSubnetClaimsSortsLast() {
        let local = [interface("en0", .wifi, ["192.168.1.5"])]
        let peer = PeerAdvertisement(rank: 0, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "203.0.113.7", port: 7), media: .ethernet),
            PeerEndpoint(address: PeerAddress(host: "192.168.1.250", port: 7), media: .wifi),
        ])
        let ordered = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), ["192.168.1.250", "203.0.113.7"])
        XCTAssertNil(ordered[1].egressMedia, "nothing local claims it, so nothing is claimed about it")
        XCTAssertTrue(ordered[1].description.contains("default route"))
    }

    func testTheAdvertisersOrderBreaksATie() {
        // Three link-local addresses on one /16: the dialer's subnet match
        // cannot separate them, so the advertiser's own order decides which is
        // dialed first — and the dial itself decides which one works.
        let local = [linkLocalInterface("en4", ["169.254.152.222"])]
        let hosts = ["169.254.32.129", "169.254.74.184", "169.254.23.203"]
        let peer = PeerAdvertisement(rank: 0, endpoints: hosts.map {
            PeerEndpoint(address: PeerAddress(host: $0, port: 7), media: .thunderbolt)
        })
        let ordered = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), hosts)
        XCTAssertEqual(ordered.map(\.advertisedIndex), [0, 1, 2])
    }

    func testSelectionIsDeterministicAcrossRanks() {
        // Two ranks with identical interface tables must derive the same order
        // for the same advertisement, or a dial and its accept disagree about
        // which cable the pair is on.
        let local = [
            interface("en0", .wifi, ["192.168.1.5"]),
            interface("en1", .ethernet, ["10.0.0.5"]),
            linkLocalInterface("en4", ["169.254.152.222"]),
        ]
        let peer = PeerAdvertisement(rank: 3, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "192.168.1.250", port: 7), media: .wifi),
            PeerEndpoint(address: PeerAddress(host: "169.254.23.203", port: 7), media: .thunderbolt),
            PeerEndpoint(address: PeerAddress(host: "10.0.0.9", port: 7), media: .ethernet),
        ])
        let a = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        let b = PathSelection.candidates(for: peer, own: nil, interfaces: local)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.map(\.endpoint.address.host),
                       ["169.254.23.203", "10.0.0.9", "192.168.1.250"])
    }

    func testARemoteLoopbackAddressIsIgnoredWhenSomethingRealIsOffered() {
        // 127.0.0.1 in a remote peer's advertisement names *this* machine.
        // Honouring it would dial the wrong host entirely.
        let local = [
            interface("lo0", .loopback, ["127.0.0.1"]),
            interface("en0", .wifi, ["192.168.1.5"]),
        ]
        let peer = PeerAdvertisement(rank: 0, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "127.0.0.1", port: 7), media: .loopback),
            PeerEndpoint(address: PeerAddress(host: "192.168.1.250", port: 7), media: .wifi),
        ])
        let own = PeerAdvertisement(rank: 1, address: PeerAddress(host: "192.168.1.5", port: 7))
        let ordered = PathSelection.candidates(for: peer, own: own, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), ["192.168.1.250"])
    }

    func testLoopbackSurvivesWhenTheWholeWorldIsOnOneMachine() {
        let local = [interface("lo0", .loopback, ["127.0.0.1"])]
        let peer = PeerAdvertisement(rank: 0, address: PeerAddress(host: "127.0.0.1", port: 7))
        let own = PeerAdvertisement(rank: 1, address: PeerAddress(host: "127.0.0.1", port: 8))
        XCTAssertTrue(own.isLoopbackOnly)
        let ordered = PathSelection.candidates(for: peer, own: own, interfaces: local)
        XCTAssertEqual(ordered.map(\.endpoint.address.host), ["127.0.0.1"])
    }

    func testAPeerOfferingOnlyLoopbackIsStillTried() {
        // Dropping its only entry would leave nothing to dial, which is a worse
        // failure than dialing something that turns out to be ourselves.
        let local = [interface("en0", .wifi, ["192.168.1.5"])]
        let peer = PeerAdvertisement(rank: 0, address: PeerAddress(host: "127.0.0.1", port: 7))
        let own = PeerAdvertisement(rank: 1, address: PeerAddress(host: "192.168.1.5", port: 7))
        let ordered = PathSelection.candidates(for: peer, own: own, interfaces: local)
        XCTAssertEqual(ordered.count, 1)
    }

    // MARK: - Forged and mismatched advertisements

    func testAnAnnouncementFromAnUnadvertisedAddressIsDisavowed() {
        let advertised = PeerAdvertisement(rank: 1, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "169.254.23.203", port: 7), media: .thunderbolt),
            PeerEndpoint(address: PeerAddress(host: "192.168.1.238", port: 7), media: .wifi),
        ])
        XCTAssertTrue(advertised.disavows(source: PeerAddress(host: "192.168.1.99", port: 55_001)),
                      "a rank dialing from an address it never advertised is not that rank")
    }

    func testAnAnnouncementFromAnAdvertisedAddressIsAccepted() {
        let advertised = PeerAdvertisement(rank: 1, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "169.254.23.203", port: 7), media: .thunderbolt),
            PeerEndpoint(address: PeerAddress(host: "192.168.1.238", port: 7), media: .wifi),
        ])
        // The source port is ephemeral and carries no information; only the
        // host is checked.
        XCTAssertFalse(advertised.disavows(source: PeerAddress(host: "192.168.1.238", port: 61_204)))
        XCTAssertFalse(advertised.disavows(source: PeerAddress(host: "169.254.23.203", port: 143)))
    }

    func testTheCheckIsSkippedRatherThanGuessedAt() {
        let advertised = PeerAdvertisement(rank: 1, address: PeerAddress(host: "192.168.1.238", port: 7))
        XCTAssertFalse(advertised.disavows(source: nil),
                       "a transport with no addressing has nothing to check")
        XCTAssertFalse(advertised.disavows(source: PeerAddress(host: "127.0.0.1", port: 9)),
                       "this machine talking to itself is not something a third party can forge")
        XCTAssertFalse(PeerAdvertisement(rank: 1, endpoints: []).disavows(
            source: PeerAddress(host: "10.0.0.1", port: 9)),
                       "an empty advertisement contradicts nothing")
    }

    func testBootstrapRejectsAdvertisementsInTheWrongSlots() {
        let transport = LoopbackTransport()
        let listener = try! transport.listen(host: "loopback", port: 0)
        defer { listener.close() }
        // Rank 1's advertisement in rank 0's slot: the table is not the table
        // rank 0 published, so refuse it rather than dial the wrong peers.
        let scrambled = [
            PeerAdvertisement(rank: 1, address: PeerAddress(host: "loopback", port: 1)),
            PeerAdvertisement(rank: 0, address: PeerAddress(host: "loopback", port: 2)),
        ]
        XCTAssertThrowsError(try MeshFabric.bootstrap(
            rank: 0, worldSize: 2, advertisements: scrambled,
            listener: listener, transport: transport, timeout: 0.5)
        ) { error in
            guard case MCCLError.protocolViolation(let what) = error else {
                return XCTFail("expected .protocolViolation, got \(error)")
            }
            XCTAssertTrue(what.contains("claims rank 1"), what)
        }
    }

    func testDialingAPeerThatAdvertisedNothingIsAnError() {
        XCTAssertThrowsError(try MeshFabric.dial(
            to: PeerAdvertisement(rank: 3, endpoints: []),
            transport: LoopbackTransport(), timeout: 0.2)
        ) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("rank 3"), what)
        }
    }

    // MARK: - A multi-address world over real sockets

    /// A port nothing is listening on. Binding and immediately closing is the
    /// cheapest way to name one the kernel just confirmed was free, and a dial
    /// to it is refused immediately rather than hanging — which is what keeps
    /// this test fast.
    private func deadPort() throws -> Int {
        let listener = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        let port = listener.address.port
        listener.close()
        return port
    }

    /// A dial has to be cuttable short, or per-pair addressing cannot work: the
    /// order it walks deliberately contains addresses that may be unreachable,
    /// and a blocking `connect(2)` to one that silently drops SYNs parks for the
    /// kernel's own TCP timeout — around 75 seconds — however little the caller
    /// asked for.
    ///
    /// 192.0.2.0/24 is TEST-NET-1: reserved for documentation, so nothing
    /// answers. Whether this machine's stack rejects it instantly or routes it
    /// into a black hole, the call must come back inside its own deadline.
    func testAConnectToABlackHoleRespectsItsTimeout() {
        let start = Date()
        XCTAssertThrowsError(try TCPTransport().connect(
            to: PeerAddress(host: "192.0.2.1", port: 9), timeout: 1))
        XCTAssertLessThan(Date().timeIntervalSince(start), 15,
                          "the deadline has to bound the connect, not just the retry")
    }

    /// Every rank advertises a dead address first and its real one second, then
    /// the world is brought up over real loopback TCP.
    ///
    /// This is the mechanism the mixed fabric depends on: on a link-local
    /// Thunderbolt fabric three TB ports share one /16, so the ordering rule
    /// alone cannot say which address reaches this peer, and the dial has to.
    func testAWorldFormsWhenEveryPeersFirstAddressIsDead() throws {
        let previous = MeshFabric.dialAttemptSeconds
        MeshFabric.dialAttemptSeconds = 0.25
        defer { MeshFabric.dialAttemptSeconds = previous }

        let worldSize = 3
        let transport = TCPTransport()
        var listeners: [Listener] = []
        for _ in 0..<worldSize { listeners.append(try transport.listen(host: "127.0.0.1", port: 0)) }
        defer { listeners.forEach { $0.close() } }

        let dead = try deadPort()
        let advertisements = (0..<worldSize).map { rank in
            PeerAdvertisement(rank: rank, endpoints: [
                PeerEndpoint(address: PeerAddress(host: "127.0.0.1", port: dead), media: .thunderbolt),
                PeerEndpoint(address: listeners[rank].address, media: .wifi),
            ])
        }

        let results = ResultSlots<MeshFabric>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "advertisement.bootstrap", attributes: .concurrent)
        for rank in 0..<worldSize {
            let listener = listeners[rank]
            queue.async(group: group) {
                do {
                    results.set(rank, .success(try MeshFabric.bootstrap(
                        rank: rank, worldSize: worldSize, advertisements: advertisements,
                        listener: listener, transport: transport, timeout: 20)))
                } catch {
                    results.set(rank, .failure(error))
                }
            }
        }
        group.wait()

        let fabrics = try results.collect()
        defer { fabrics.forEach { $0.shutdown() } }
        for fabric in fabrics {
            for peer in 0..<worldSize where peer != fabric.rank {
                XCTAssertNoThrow(try fabric.channel(to: peer),
                                 "rank \(fabric.rank) has no channel to \(peer)")
            }
        }
    }

    /// The same world, driven through a collective, so the channels are shown to
    /// carry traffic and not merely to exist.
    func testAMultiAddressWorldAllReducesCorrectly() async throws {
        let previous = MeshFabric.dialAttemptSeconds
        MeshFabric.dialAttemptSeconds = 0.25
        defer { MeshFabric.dialAttemptSeconds = previous }

        let worldSize = 3
        let transport = TCPTransport()
        var listeners: [Listener] = []
        for _ in 0..<worldSize { listeners.append(try transport.listen(host: "127.0.0.1", port: 0)) }
        defer { listeners.forEach { $0.close() } }

        let dead = try deadPort()
        let advertisements = (0..<worldSize).map { rank in
            PeerAdvertisement(rank: rank, endpoints: [
                PeerEndpoint(address: PeerAddress(host: "127.0.0.1", port: dead), media: .thunderbolt),
                PeerEndpoint(address: listeners[rank].address, media: .ethernet),
            ])
        }

        let results = ResultSlots<Communicator>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "advertisement.comm", attributes: .concurrent)
        for rank in 0..<worldSize {
            let listener = listeners[rank]
            queue.async(group: group) {
                do {
                    results.set(rank, .success(try Communicator.bootstrap(
                        rank: rank, worldSize: worldSize, advertisements: advertisements,
                        listener: listener, transport: transport, timeout: 20)))
                } catch {
                    results.set(rank, .failure(error))
                }
            }
        }
        group.wait()

        let comms = try results.collect()
        defer { comms.forEach { $0.shutdown() } }

        let count = 512
        try await withThrowingTaskGroup(of: Void.self) { tasks in
            for comm in comms {
                tasks.addTask {
                    let buffer = UnsafeMutableRawBufferPointer.allocate(
                        byteCount: count * 4, alignment: 64)
                    defer { buffer.deallocate() }
                    let floats = buffer.bindMemory(to: Float.self)
                    for i in 0..<count { floats[i] = Float(comm.rank + 1) }
                    try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)
                    for i in 0..<count {
                        XCTAssertEqual(floats[i], 6, accuracy: 1e-5, "element \(i)")
                    }
                }
            }
            try await tasks.waitForAll()
        }
    }
}
