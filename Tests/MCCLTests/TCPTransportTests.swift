import XCTest
#if canImport(Darwin)
import Darwin
#endif
@testable import MCCL

/// The failure side of the POSIX socket transport.
///
/// `TransportTests` covers the happy path — bind, dial, move bytes. Everything
/// here is a way bring-up goes wrong on a real cluster: a bind address that
/// belongs to another machine, a port already in use, a hostname that does not
/// resolve, a peer that went away mid-collective. Each one has to arrive as a
/// typed `MCCLError` carrying the offending detail, because that string is all
/// an operator gets when a 16-node job fails to come up.
final class TCPTransportTests: XCTestCase {

    // MARK: - Binding

    func testBindingAnAddressThisMachineDoesNotOwnFailsWithTheAddressInTheMessage() {
        // 203.0.113.0/24 is TEST-NET-3 (RFC 5737): never assigned to a host.
        XCTAssertThrowsError(try TCPTransport().listen(host: "203.0.113.7", port: 0)) { error in
            guard case MCCLError.socketFailure(let what, let code) = error else {
                return XCTFail("expected .socketFailure, got \(error)")
            }
            XCTAssertTrue(what.hasPrefix("bind(203.0.113.7:0)"),
                          "the message must name the address that failed, got '\(what)'")
            XCTAssertEqual(code, EADDRNOTAVAIL)
            XCTAssertTrue("\(error)".contains("errno \(EADDRNOTAVAIL)"), "\(error)")
        }
    }

    func testBindingAPortSomethingElseIsAlreadyListeningOnFails() throws {
        let held = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        defer { held.close() }
        XCTAssertGreaterThan(held.address.port, 0)

        XCTAssertThrowsError(
            try TCPTransport().listen(host: "127.0.0.1", port: held.address.port)
        ) { error in
            guard case MCCLError.socketFailure(let what, let code) = error else {
                return XCTFail("expected .socketFailure, got \(error)")
            }
            XCTAssertTrue(what.hasPrefix("bind("), what)
            XCTAssertEqual(code, EADDRINUSE, "two listeners on one port is EADDRINUSE, not a silent success")
        }
    }

    func testAHostnameThatDoesNotResolveIsAnInvalidArgument() {
        // `.invalid` is reserved by RFC 6761 precisely so it never resolves.
        XCTAssertThrowsError(
            try SocketSupport.resolve(host: "mccl-no-such-node.invalid", port: 7777, passive: false)
        ) { error in
            guard case MCCLError.invalidArgument(let what) = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
            XCTAssertTrue(what.contains("getaddrinfo(mccl-no-such-node.invalid:7777)"),
                          "the message must repeat what could not be resolved, got '\(what)'")
        }
        // And the same failure surfaces through the public entry points.
        XCTAssertThrowsError(try TCPTransport().listen(host: "mccl-no-such-node.invalid", port: 0))
        XCTAssertThrowsError(try TCPTransport().connect(
            to: PeerAddress(host: "mccl-no-such-node.invalid", port: 7777), timeout: 0.1))
    }

    func testResolutionPrefersIPv4ForAMixedCluster() throws {
        let candidates = try SocketSupport.resolve(host: "localhost", port: 7777, passive: false)
        XCTAssertFalse(candidates.isEmpty)
        // localhost usually resolves to both ::1 and 127.0.0.1; whichever this
        // machine reports, the v4 candidates must come first.
        let families = candidates.map(\.family)
        if let lastV4 = families.lastIndex(of: AF_INET), let firstV6 = families.firstIndex(of: AF_INET6) {
            XCTAssertLessThan(lastV4, firstV6, "IPv4 candidates must sort ahead of IPv6")
        }
    }

    // MARK: - Socket introspection

    func testBoundPortReportsAFailureRatherThanAGarbagePort() {
        XCTAssertThrowsError(try SocketSupport.boundPort(fd: -1)) { error in
            guard case MCCLError.socketFailure(let what, _) = error else {
                return XCTFail("expected .socketFailure, got \(error)")
            }
            XCTAssertEqual(what, "getsockname()")
        }
    }

    func testPeerNameFallsBackToTheDescriptorWhenThereIsNoPeer() throws {
        XCTAssertEqual(SocketSupport.peerName(fd: -1), "fd-1",
                       "an unusable descriptor must still produce a printable name")

        // A bound-but-never-connected socket has no peer either.
        let fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        XCTAssertEqual(SocketSupport.peerName(fd: fd), "fd\(fd)")
    }

    func testAcceptedChannelsNameTheirPeer() throws {
        let listener = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        defer { listener.close() }
        let accepted = expectation(description: "accepted")
        var serverSide: Channel?
        DispatchQueue.global().async {
            serverSide = try? listener.accept(timeout: 5)
            accepted.fulfill()
        }
        let client = try TCPTransport().connect(to: listener.address, timeout: 5)
        defer { client.close() }
        wait(for: [accepted], timeout: 10)
        defer { serverSide?.close() }
        XCTAssertEqual(client.peerDescription, "127.0.0.1:\(listener.address.port)")
        XCTAssertTrue(serverSide?.peerDescription.hasPrefix("127.0.0.1:") == true,
                      "got \(serverSide?.peerDescription ?? "nil")")
    }

    // MARK: - Dialling

    func testDiallingAClosedPortGivesUpAtTheDeadlineWithTheLastErrno() {
        let start = Date()
        XCTAssertThrowsError(
            try TCPTransport().connect(to: PeerAddress(host: "127.0.0.1", port: 9), timeout: 0.3)
        ) { error in
            guard case MCCLError.socketFailure(let what, let code) = error else {
                return XCTFail("expected .socketFailure, got \(error)")
            }
            XCTAssertTrue(what.contains("connect to 127.0.0.1:9"), what)
            XCTAssertEqual(code, ECONNREFUSED, "the last errno must survive the retry loop")
        }
        XCTAssertGreaterThanOrEqual(Date().timeIntervalSince(start), 0.3,
                                    "connect retries until the deadline — bring-up races are normal")
    }

    // MARK: - Transfers

    func testZeroLengthTransfersAreNoOps() throws {
        let (client, server) = try connectedPair()
        defer { client.close(); server.close() }

        var empty = [UInt8]()
        XCTAssertNoThrow(try empty.withUnsafeBytes { try client.sendBytes($0) })
        XCTAssertNoThrow(try empty.withUnsafeMutableBytes { try client.receiveBytes(into: $0) })
    }

    func testReceivingFromAPeerThatHungUpReportsTheClosure() throws {
        let (client, server) = try connectedPair()
        defer { client.close() }
        server.close()

        var buffer = [UInt8](repeating: 0, count: 64)
        XCTAssertThrowsError(try buffer.withUnsafeMutableBytes { try client.receiveBytes(into: $0) }) { error in
            XCTAssertEqual(error as? MCCLError, .connectionClosed,
                           "a rank that died mid-collective must not look like a short read")
        }
    }

    func testSendingToAPeerThatHungUpFailsInsteadOfKillingTheProcess() throws {
        let (client, server) = try connectedPair()
        defer { client.close() }
        server.close()

        // SO_NOSIGPIPE is set, so this must come back as an error rather than
        // SIGPIPE'ing the whole job. More than one write is needed: the first
        // lands in the socket buffer before the RST arrives.
        let payload = [UInt8](repeating: 0xAB, count: 1 << 20)
        var caught: Error?
        for _ in 0..<32 where caught == nil {
            do { try payload.withUnsafeBytes { try client.sendBytes($0) } } catch { caught = error }
        }
        guard let caught else { return XCTFail("expected the send to a dead peer to fail") }
        switch caught {
        case MCCLError.connectionClosed:
            break
        case MCCLError.socketFailure(let what, let code):
            XCTAssertEqual(what, "send()")
            XCTAssertTrue(code == EPIPE || code == ECONNRESET, "unexpected errno \(code)")
        default:
            XCTFail("expected .connectionClosed or .socketFailure, got \(caught)")
        }
    }

    func testClosingAChannelTwiceIsSafe() throws {
        let (client, server) = try connectedPair()
        client.close()
        client.close()
        server.close()
        server.close()
    }

    func testListenerClosesAreIdempotent() throws {
        let listener = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        listener.close()
        listener.close()
        // A closed listener is not a working one.
        XCTAssertThrowsError(try listener.accept(timeout: 0.1))
    }

    func testAcceptTimesOutRatherThanBlockingForever() throws {
        let listener = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        defer { listener.close() }
        XCTAssertThrowsError(try listener.accept(timeout: 0.2)) { error in
            guard case MCCLError.timedOut(let what) = error else {
                return XCTFail("expected .timedOut, got \(error)")
            }
            XCTAssertTrue(what.contains("accept on 127.0.0.1:\(listener.address.port)"), what)
        }
    }

    func testSocketBufferSizeIsConfigurable() throws {
        // The default is 4 MiB; a caller tuning for a slower link must be able
        // to change it and still get a working transport.
        let transport = TCPTransport(socketBufferBytes: 64 << 10)
        XCTAssertEqual(transport.socketBufferBytes, 64 << 10)
        let listener = try transport.listen(host: "127.0.0.1", port: 0)
        defer { listener.close() }
        let accepted = expectation(description: "accepted")
        DispatchQueue.global().async {
            let channel = try? listener.accept(timeout: 5)
            channel?.close()
            accepted.fulfill()
        }
        let client = try transport.connect(to: listener.address, timeout: 5)
        client.close()
        wait(for: [accepted], timeout: 10)
    }

    // MARK: - Helpers

    /// A connected TCP pair over loopback, both ends returned to the caller.
    private func connectedPair() throws -> (Channel, Channel) {
        let listener = try TCPTransport().listen(host: "127.0.0.1", port: 0)
        defer { listener.close() }
        let box = Collector<Channel>()
        let accepted = expectation(description: "accepted")
        DispatchQueue.global().async {
            if let channel = try? listener.accept(timeout: 5) { box.set(0, channel) }
            accepted.fulfill()
        }
        let client = try TCPTransport().connect(to: listener.address, timeout: 5)
        wait(for: [accepted], timeout: 10)
        guard let server = box.get(0) else {
            client.close()
            throw MCCLError.timedOut("test peer never accepted")
        }
        return (client, server)
    }
}
