import XCTest
@testable import MCCL

final class TransportTests: XCTestCase {

    func testPeerAddressParsing() {
        XCTAssertEqual(PeerAddress("10.0.0.4:7777"), PeerAddress(host: "10.0.0.4", port: 7777))
        XCTAssertEqual(PeerAddress("7777"), PeerAddress(host: "127.0.0.1", port: 7777))
        XCTAssertEqual(PeerAddress("[::1]:9000"), PeerAddress(host: "::1", port: 9000))
        XCTAssertNil(PeerAddress("not-a-port"))
        XCTAssertTrue(PeerAddress(host: "127.0.0.1", port: 1).isLoopback)
        XCTAssertFalse(PeerAddress(host: "10.0.0.4", port: 1).isLoopback)
    }

    func testTCPRoundTrip() throws {
        try roundTrip(transport: TCPTransport(), host: "127.0.0.1")
    }

    func testLoopbackRoundTrip() throws {
        try roundTrip(transport: LoopbackTransport(), host: "loopback")
    }

    private func roundTrip(transport: Transport, host: String) throws {
        let listener = try transport.listen(host: host, port: 0)
        defer { listener.close() }
        XCTAssertGreaterThan(listener.address.port, 0, "ephemeral port must be reported back")

        let payload = (0..<4096).map { UInt8(truncatingIfNeeded: $0 &* 7) }
        let serverDone = expectation(description: "server echoed")

        DispatchQueue.global().async {
            do {
                let channel = try listener.accept(timeout: 10)
                var received = [UInt8](repeating: 0, count: payload.count)
                try received.withUnsafeMutableBytes { try channel.receiveBytes(into: $0) }
                try received.withUnsafeBytes { try channel.sendBytes($0) }
                serverDone.fulfill()
            } catch {
                XCTFail("server side: \(error)")
                serverDone.fulfill()
            }
        }

        let client = try transport.connect(to: listener.address, timeout: 10)
        defer { client.close() }
        try payload.withUnsafeBytes { try client.sendBytes($0) }
        var echoed = [UInt8](repeating: 0, count: payload.count)
        try echoed.withUnsafeMutableBytes { try client.receiveBytes(into: $0) }
        XCTAssertEqual(echoed, payload)

        wait(for: [serverDone], timeout: 15)
    }

    func testFrameHeaderRoundTrip() throws {
        var header = WireHeader()
        header.codec = WireCodecID.int8Blockwise.rawValue
        header.dataType = DataType.bfloat16.wireCode
        header.elementCount = 123_456
        header.payloadBytes = 99
        header.blockSize = 256
        header.tag = 42

        let scratch = ScratchBuffer(capacity: WireHeader.byteCount)
        header.encode(into: scratch.base)
        let decoded = try WireHeader.decode(from: scratch.base)

        XCTAssertEqual(decoded.codec, header.codec)
        XCTAssertEqual(decoded.dataType, header.dataType)
        XCTAssertEqual(decoded.elementCount, header.elementCount)
        XCTAssertEqual(decoded.payloadBytes, header.payloadBytes)
        XCTAssertEqual(decoded.blockSize, header.blockSize)
        XCTAssertEqual(decoded.tag, header.tag)
    }

    func testFrameRejectsGarbage() {
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount)
        scratch.base.storeBytes(of: UInt32(0xDEADBEEF), toByteOffset: 0, as: UInt32.self)
        XCTAssertThrowsError(try WireHeader.decode(from: scratch.base))
    }

    func testMeshBootstrapConnectsEveryPair() throws {
        let fabrics = try MeshFabric.makeGroup(worldSize: 4, transport: TCPTransport(), host: "127.0.0.1")
        defer { fabrics.forEach { $0.shutdown() } }

        XCTAssertEqual(fabrics.map(\.rank), [0, 1, 2, 3])
        for fabric in fabrics {
            for peer in 0..<4 where peer != fabric.rank {
                XCTAssertNoThrow(try fabric.channel(to: peer))
            }
            XCTAssertThrowsError(try fabric.channel(to: fabric.rank))
            XCTAssertThrowsError(try fabric.channel(to: 4))
        }
    }

    func testCommunicatorWithoutFabricRejectsCollectives() async throws {
        let comm = Communicator(rank: 0, worldSize: 4, topology: Topology.uniform(nodeCount: 4))
        let buffer = Buf.allocate(count: 8, dataType: .float32)
        defer { buffer.deallocate() }
        do {
            try await comm.allReduce(buffer, count: 8, dataType: .float32)
            XCTFail("expected .noFabric")
        } catch let error as MCCLError {
            XCTAssertEqual(error, .noFabric)
        }
    }

    func testSingleRankCommunicatorIsANoOp() async throws {
        let comm = Communicator(rank: 0, worldSize: 1, topology: Topology.uniform(nodeCount: 1))
        let buffer = Buf.allocate(count: 4, dataType: .float32)
        defer { buffer.deallocate() }
        Buf.fill(buffer, count: 4, dataType: .float32) { Float($0) + 1 }
        try await comm.allReduce(buffer, count: 4, dataType: .float32)
        XCTAssertEqual(Buf.read(buffer, count: 4, dataType: .float32), [1, 2, 3, 4])
    }
}
