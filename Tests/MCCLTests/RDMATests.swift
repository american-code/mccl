import XCTest
@testable import MCCL

/// Tests for the RDMA-over-Thunderbolt transport.
///
/// None of this has run on Thunderbolt 5 hardware. What these tests establish is
/// narrower and worth stating exactly: that the queue-pair lifecycle matches the
/// sequence and the attribute masks TN3205 publishes, that the framing satisfies
/// the same-size rule by construction, that a byte stream survives chunking and
/// reassembly, and that a lost slot is reported rather than swallowed. Whether
/// Apple's hardware then behaves as its technote says is the one thing no test
/// here can settle — see `docs/RDMA.md` for the checklist that would.
final class RDMASpecConstantTests: XCTestCase {

    /// The three attribute masks, computed from the SDK's own enum values and
    /// compared against the numbers TN3205's listings imply.
    func testAttributeMasksMatchTheTechnote() {
        // IBV_QP_STATE(1) | IBV_QP_PKEY_INDEX(16) | IBV_QP_PORT(32) | IBV_QP_ACCESS_FLAGS(8)
        XCTAssertEqual(RDMASpec.Mask.initialize, 57)
        // IBV_QP_STATE(1) | IBV_QP_AV(128) | IBV_QP_PATH_MTU(256)
        //   | IBV_QP_DEST_QPN(1048576) | IBV_QP_RQ_PSN(4096)
        XCTAssertEqual(RDMASpec.Mask.readyToReceive, 1_053_057)
        // IBV_QP_STATE(1) | IBV_QP_SQ_PSN(65536)
        XCTAssertEqual(RDMASpec.Mask.readyToSend, 65_537)
    }

    func testSpecConstantsMatchTheTechnote() {
        XCTAssertEqual(RDMASpec.queuePairType, 3, "IBV_QPT_UC")
        XCTAssertEqual(RDMASpec.pathMTU, 5, "IBV_MTU_4096")
        XCTAssertEqual(RDMASpec.accessLocalWrite, 1, "IBV_ACCESS_LOCAL_WRITE")
        XCTAssertEqual(RDMASpec.opcodeSend, 2, "IBV_WR_SEND")
        XCTAssertEqual(RDMASpec.sendSignaled, 2, "IBV_SEND_SIGNALED")
        XCTAssertEqual(RDMASpec.portNumber, 1)
        XCTAssertEqual(RDMASpec.gidIndex, 1)
        XCTAssertEqual(RDMASpec.frameBytes, 4096)
        XCTAssertEqual(RDMASpec.maxWorkRequests, 4095)
        XCTAssertEqual(RDMASpec.maxQueuePairs, 10)
    }

    /// "Message sizes of up to 16,773,120 bytes", which is 4095 x 4096.
    func testMaximumMessageIsFourThousandNinetyFiveFrames() {
        XCTAssertEqual(RDMASpec.maxMessageBytes, 16_773_120)
        XCTAssertEqual(RDMASpec.maxMessageBytes,
                       RDMASpec.maxWorkRequests * RDMASpec.frameBytes)
    }
}

// MARK: - Geometry and framing

final class RDMAFramingTests: XCTestCase {

    func testDefaultGeometryIsWholeFramesAndWithinLimits() throws {
        let geometry = try RDMASlotGeometry()
        XCTAssertEqual(geometry.slotBytes % RDMASpec.frameBytes, 0)
        XCTAssertEqual(geometry.frameCount, 16)
        XCTAssertLessThanOrEqual(geometry.slotBytes, RDMASpec.maxMessageBytes)
        XCTAssertEqual(geometry.payloadCapacity, geometry.slotBytes - RDMASlotHeader.byteCount)
        XCTAssertEqual(geometry.ringBytes, geometry.slotBytes * geometry.slotCount)
    }

    func testGeometryRejectsSlotsThatAreNotWholeFrames() {
        XCTAssertThrowsError(try RDMASlotGeometry(slotBytes: 5000, slotCount: 4)) { error in
            XCTAssertTrue("\(error)".contains("not a multiple"), "\(error)")
        }
    }

    /// A slot above the maximum message would be rejected by the hardware, not
    /// by us, and only at the moment of the first transfer.
    func testGeometryRejectsSlotsAboveTheMaximumMessage() {
        XCTAssertThrowsError(
            try RDMASlotGeometry(slotBytes: RDMASpec.maxMessageBytes + RDMASpec.frameBytes,
                                 slotCount: 1))
    }

    func testGeometryRejectsRingsAboveTheWorkRequestLimit() {
        XCTAssertThrowsError(try RDMASlotGeometry(slotBytes: 4096, slotCount: 4096)) { error in
            XCTAssertTrue("\(error)".contains("work requests"), "\(error)")
        }
    }

    func testGeometryAcceptsTheLargestLegalSlot() throws {
        let geometry = try RDMASlotGeometry(slotBytes: RDMASpec.maxMessageBytes, slotCount: 1)
        XCTAssertEqual(geometry.frameCount, RDMASpec.maxWorkRequests)
    }

    func testSlotHeaderRoundTrips() throws {
        var header = RDMASlotHeader()
        header.sequence = 0xDEAD_BEEF
        header.payloadBytes = 4321
        header.flags = RDMASlotHeader.flagClosing
        var storage = [UInt8](repeating: 0, count: RDMASlotHeader.byteCount)
        try storage.withUnsafeMutableBytes { raw in
            header.encode(into: raw.baseAddress!)
            let decoded = try RDMASlotHeader.decode(from: raw.baseAddress!, capacity: 8192)
            XCTAssertEqual(decoded, header)
            XCTAssertTrue(decoded.isClosing)
        }
    }

    func testSlotHeaderRejectsForeignMagic() throws {
        var storage = [UInt8](repeating: 0xAB, count: RDMASlotHeader.byteCount)
        try storage.withUnsafeMutableBytes { raw -> Void in
            XCTAssertThrowsError(try RDMASlotHeader.decode(from: raw.baseAddress!, capacity: 8192))
        }
    }

    func testSlotHeaderRejectsPayloadBeyondCapacity() throws {
        var header = RDMASlotHeader()
        header.payloadBytes = 9000
        var storage = [UInt8](repeating: 0, count: RDMASlotHeader.byteCount)
        try storage.withUnsafeMutableBytes { raw -> Void in
            header.encode(into: raw.baseAddress!)
            XCTAssertThrowsError(try RDMASlotHeader.decode(from: raw.baseAddress!, capacity: 8192)) {
                XCTAssertTrue("\($0)".contains("capacity"), "\($0)")
            }
        }
    }

    // MARK: Chunking

    func testChunkingSplitsExactlyAtCapacity() {
        XCTAssertEqual(RDMAChunking.chunks(of: 100, capacity: 40), [40, 40, 20])
        XCTAssertEqual(RDMAChunking.chunks(of: 80, capacity: 40), [40, 40])
        XCTAssertEqual(RDMAChunking.chunks(of: 41, capacity: 40), [40, 1])
        XCTAssertEqual(RDMAChunking.chunks(of: 1, capacity: 40), [1])
        XCTAssertEqual(RDMAChunking.chunks(of: 0, capacity: 40), [])
    }

    func testChunkCountAgreesWithChunkList() {
        for byteCount in [0, 1, 39, 40, 41, 4095, 4096, 4097, 1 << 20] {
            XCTAssertEqual(RDMAChunking.slotCount(for: byteCount, capacity: 40),
                           RDMAChunking.chunks(of: byteCount, capacity: 40).count,
                           "byteCount \(byteCount)")
        }
    }

    func testChunksAlwaysSumToTheOriginalLength() {
        for byteCount in [1, 7, 4096, 65_500, 1 << 20, (1 << 20) + 3] {
            let chunks = RDMAChunking.chunks(of: byteCount, capacity: 65_520)
            XCTAssertEqual(chunks.reduce(0, +), byteCount)
            XCTAssertTrue(chunks.allSatisfy { $0 > 0 && $0 <= 65_520 })
        }
    }

    // MARK: Sequence checking

    func testSequenceCheckerAcceptsAConsecutiveRun() throws {
        var checker = RDMASequenceChecker()
        for expected in UInt32(0)..<1000 { try checker.accept(expected) }
        XCTAssertEqual(checker.expected, 1000)
    }

    func testSequenceCheckerRejectsAGap() {
        var checker = RDMASequenceChecker()
        XCTAssertNoThrow(try checker.accept(0))
        XCTAssertThrowsError(try checker.accept(2)) { error in
            guard case MCCLError.rdmaSequenceGap(let expected, let received) = error else {
                return XCTFail("expected a sequence gap, got \(error)")
            }
            XCTAssertEqual(expected, 1)
            XCTAssertEqual(received, 2)
        }
    }

    /// A replayed slot is as much a violation as a missing one.
    func testSequenceCheckerRejectsARepeat() {
        var checker = RDMASequenceChecker()
        XCTAssertNoThrow(try checker.accept(0))
        XCTAssertThrowsError(try checker.accept(0))
    }

    func testSequenceCheckerWrapsCleanly() throws {
        var checker = RDMASequenceChecker()
        // Walk the counter to its last value without doing 4 billion iterations.
        for _ in 0..<3 { _ = checker.next() }
        XCTAssertEqual(checker.expected, 3)

        var sender = RDMASequenceChecker()
        for _ in 0..<5 { _ = sender.next() }
        XCTAssertEqual(sender.expected, 5)
    }

    func testSequenceGapErrorNamesTheCause() {
        let error = MCCLError.rdmaSequenceGap(expected: 7, received: 9)
        let text = error.description
        XCTAssertTrue(text.contains("unreliable connection"), text)
        XCTAssertTrue(text.contains("7"))
        XCTAssertTrue(text.contains("9"))
    }
}

// MARK: - Queue-pair state machine

final class RDMAStateMachineTests: XCTestCase {

    private func makeConnection(_ mock: MockVerbs, index: Int = 0,
                                geometry: RDMASlotGeometry? = nil) throws -> RDMAConnection {
        try RDMAConnection(
            verbs: mock, deviceIndex: index, deviceName: mock.devices[index],
            geometry: geometry ?? (try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)),
            packetSequenceNumber: 7)
    }

    /// Reset -> Init happens in the initialiser; RTR and RTS in `connect`.
    func testLifecycleFollowsTheTechnotesOrder() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock, index: 0)
        let b = try makeConnection(mock, index: 1)
        XCTAssertEqual(mock.transitions.map(\.state), ["INIT", "INIT"])

        try a.connect(to: b.metadata)
        try b.connect(to: a.metadata)
        XCTAssertEqual(mock.transitions.map(\.state), ["INIT", "INIT", "RTR", "RTS", "RTR", "RTS"])
        a.close(); b.close()
    }

    func testTransitionsUseTheSpecifiedMasks() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock, index: 0)
        let b = try makeConnection(mock, index: 1)
        try a.connect(to: b.metadata)

        XCTAssertEqual(mock.mask(for: "INIT"), RDMASpec.Mask.initialize)
        XCTAssertEqual(mock.mask(for: "RTR"), RDMASpec.Mask.readyToReceive)
        XCTAssertEqual(mock.mask(for: "RTS"), RDMASpec.Mask.readyToSend)
        XCTAssertEqual(mock.mask(for: "INIT"), 57)
        XCTAssertEqual(mock.mask(for: "RTR"), 1_053_057)
        XCTAssertEqual(mock.mask(for: "RTS"), 65_537)
        a.close(); b.close()
    }

    func testReadyToReceiveCarriesThePeersAddressing() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock, index: 0)
        let b = try makeConnection(mock, index: 1)
        let peer = b.metadata
        try a.connect(to: peer)

        let attributes = try XCTUnwrap(mock.readyToReceiveAttributes.first)
        XCTAssertEqual(attributes.pathMTU, RDMASpec.pathMTU, "IBV_MTU_4096")
        XCTAssertEqual(attributes.port, 1, "Apple controllers have one port")
        XCTAssertEqual(attributes.sourceGIDIndex, 1, "GID index 1 is the IPv4-mapped address")
        XCTAssertEqual(attributes.hopLimit, 1, "point to point")
        XCTAssertEqual(attributes.destinationQueuePairNumber, peer.queuePairNumber)
        XCTAssertEqual(attributes.receivePSN, peer.packetSequenceNumber)
        XCTAssertEqual(attributes.destinationGID, peer.globalID)
        a.close(); b.close()
    }

    func testQueuePairIsAnUnreliableConnection() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock)
        XCTAssertEqual(mock.createdQueuePairs.first?.type, RDMASpec.queuePairType)
        XCTAssertEqual(mock.createdQueuePairs.first?.type, 3)
        a.close()
    }

    /// "applications should only register memory as IBV_ACCESS_LOCAL_WRITE".
    func testMemoryIsRegisteredLocalWriteOnly() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock)
        let region = try XCTUnwrap(mock.registeredRegions.first)
        XCTAssertEqual(region.access, RDMASpec.accessLocalWrite)
        XCTAssertEqual(region.access, 1)
        a.close()
    }

    /// The mock refuses a registration that is not page aligned, so this passing
    /// means `posix_memalign` was actually used.
    func testRingIsPageAligned() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock)
        let region = try XCTUnwrap(mock.registeredRegions.first)
        XCTAssertEqual(region.address % UInt(getpagesize()), 0)
        a.close()
    }

    func testBothRingsAreCoveredByOneRegistration() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)
        let a = try makeConnection(mock, geometry: geometry)
        XCTAssertEqual(mock.registeredRegions.count, 1)
        XCTAssertEqual(mock.registeredRegions.first?.length, geometry.ringBytes * 2)
        a.close()
    }

    func testCompletionQueueIsOneDeeperThanTheWorkRequests() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)
        let a = try makeConnection(mock, geometry: geometry)
        XCTAssertEqual(mock.completionQueueDepths.first, geometry.slotCount * 2 + 1)
        XCTAssertEqual(mock.createdQueuePairs.first?.maxSend, geometry.slotCount)
        XCTAssertEqual(mock.createdQueuePairs.first?.maxRecv, geometry.slotCount)
        a.close()
    }

    /// Receives must be posted by the time `connect` returns, or the first
    /// message the peer sends has nowhere to land.
    func testReceivesArePostedBeforeConnectReturns() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)
        let a = try makeConnection(mock, index: 0, geometry: geometry)
        let b = try makeConnection(mock, index: 1, geometry: geometry)
        XCTAssertEqual(mock.postedReceiveLengths.count, 0)
        try a.connect(to: b.metadata)
        XCTAssertEqual(mock.postedReceiveLengths.count, geometry.slotCount)
        a.close(); b.close()
    }

    func testGeometryDisagreementIsRejectedBeforeAnyTransfer() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock, index: 0,
                                   geometry: try RDMASlotGeometry(slotBytes: 8192, slotCount: 4))
        let b = try makeConnection(mock, index: 1,
                                   geometry: try RDMASlotGeometry(slotBytes: 16384, slotCount: 4))
        XCTAssertThrowsError(try a.connect(to: b.metadata)) { error in
            XCTAssertTrue("\(error)".contains("same number of frames"), "\(error)")
        }
        a.close(); b.close()
    }

    func testTeardownReleasesEveryObject() throws {
        let mock = MockVerbs()
        do {
            let a = try makeConnection(mock, index: 0)
            let b = try makeConnection(mock, index: 1)
            try a.connect(to: b.metadata)
            try b.connect(to: a.metadata)
            XCTAssertFalse(mock.isFullyReleased)
            a.close()
            b.close()
        }
        XCTAssertTrue(mock.isFullyReleased,
                      "contexts \(mock.openContexts), pds \(mock.liveProtectionDomains), "
                      + "mrs \(mock.liveMemoryRegions), cqs \(mock.liveCompletionQueues), "
                      + "qps \(mock.liveQueuePairs)")
    }

    func testCloseIsIdempotent() throws {
        let mock = MockVerbs()
        let a = try makeConnection(mock)
        a.close()
        a.close()
        XCTAssertTrue(mock.isFullyReleased)
    }

    /// TN3205 allows ten queue pairs. The eleventh should fail, not succeed
    /// quietly and then misbehave.
    func testEleventhQueuePairIsRefused() throws {
        let mock = MockVerbs(devices: ["rdma_en2"])
        var connections: [RDMAConnection] = []
        for _ in 0..<RDMASpec.maxQueuePairs {
            connections.append(try makeConnection(mock))
        }
        XCTAssertThrowsError(try makeConnection(mock)) { error in
            XCTAssertTrue("\(error)".contains("queue-pair limit"), "\(error)")
        }
        connections.forEach { $0.close() }
    }

    /// A failed construction must not leak the objects it did manage to build.
    func testFailedConstructionUnwindsCleanly() throws {
        let mock = MockVerbs(devices: ["rdma_en2"])
        var connections: [RDMAConnection] = []
        for _ in 0..<RDMASpec.maxQueuePairs { connections.append(try makeConnection(mock)) }
        let contextsBefore = mock.openContexts
        let domainsBefore = mock.liveProtectionDomains
        let regionsBefore = mock.liveMemoryRegions
        let queuesBefore = mock.liveCompletionQueues

        XCTAssertThrowsError(try makeConnection(mock))
        XCTAssertEqual(mock.openContexts, contextsBefore)
        XCTAssertEqual(mock.liveProtectionDomains, domainsBefore)
        XCTAssertEqual(mock.liveMemoryRegions, regionsBefore)
        XCTAssertEqual(mock.liveCompletionQueues, queuesBefore)

        connections.forEach { $0.close() }
        XCTAssertTrue(mock.isFullyReleased)
    }
}

// MARK: - Data path

final class RDMADataPathTests: XCTestCase {

    private func makePair(_ mock: MockVerbs, geometry: RDMASlotGeometry)
        throws -> (RDMAConnection, RDMAConnection) {
        let a = try RDMAConnection(verbs: mock, deviceIndex: 0, deviceName: mock.devices[0],
                                   geometry: geometry, packetSequenceNumber: 11)
        let b = try RDMAConnection(verbs: mock, deviceIndex: 1, deviceName: mock.devices[1],
                                   geometry: geometry, packetSequenceNumber: 22)
        try a.connect(to: b.metadata)
        try b.connect(to: a.metadata)
        return (a, b)
    }

    /// Sends `payload` from one end and reads it at the other, concurrently.
    ///
    /// The concurrency is not decoration. A queue pair applies back-pressure the
    /// way a socket does: the receive ring holds `slotCount` buffers, and once a
    /// sender has filled them it blocks until the peer consumes some. TN3205
    /// puts it as "completion of a send operation indicates that the peer has
    /// posted sufficient receive requests", so a rank that sends a payload
    /// larger than the ring without anyone reading blocks — on a cable and in
    /// the mock alike. Every collective already runs its send and its receive on
    /// separate queues for this exact reason (see `runConcurrently`), so this
    /// helper does what the collectives do.
    private func transfer(_ payload: [UInt8], geometry: RDMASlotGeometry,
                          mock: MockVerbs? = nil) throws -> [UInt8] {
        let verbs = mock ?? MockVerbs()
        let (a, b) = try makePair(verbs, geometry: geometry)
        defer { a.close(); b.close() }
        var received = [UInt8](repeating: 0, count: payload.count)
        let sendQueue = DispatchQueue(label: "test.rdma.transfer.tx")
        let receiveQueue = DispatchQueue(label: "test.rdma.transfer.rx")
        try received.withUnsafeMutableBytes { destination in
            try payload.withUnsafeBytes { source in
                try runConcurrently({ try a.send(source) }, on: sendQueue,
                                    { try b.receive(into: destination) }, on: receiveQueue)
            }
        }
        return received
    }

    func testSingleSlotRoundTrip() throws {
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)
        let payload = (0..<100).map { UInt8($0 & 0xFF) }
        XCTAssertEqual(try transfer(payload, geometry: geometry), payload)
    }

    func testTransferSpanningManySlots() throws {
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 8)
        let payload = (0..<50_000).map { UInt8($0 &* 31 & 0xFF) }
        XCTAssertEqual(try transfer(payload, geometry: geometry), payload)
    }

    /// The boundary a chunking bug hides at: a payload that exactly fills its
    /// last slot, and one that overruns it by a single byte.
    func testTransfersAtExactSlotBoundaries() throws {
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 8)
        let capacity = geometry.payloadCapacity
        for count in [capacity - 1, capacity, capacity + 1, capacity * 3, capacity * 3 + 1] {
            let payload = (0..<count).map { UInt8($0 &* 7 & 0xFF) }
            XCTAssertEqual(try transfer(payload, geometry: geometry), payload,
                           "payload of \(count) bytes")
        }
    }

    func testTransferLargerThanTheWholeRing() throws {
        // Four slots of 4 KiB is a 16 KiB ring; send a quarter of a megabyte
        // through it, which forces the sender to block on completions and reuse
        // every slot many times over.
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 4)
        let payload = (0..<262_144).map { UInt8($0 &* 13 & 0xFF) }
        XCTAssertEqual(try transfer(payload, geometry: geometry), payload)
    }

    func testEmptyWriteMovesNothing() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 4)
        let (a, b) = try makePair(mock, geometry: geometry)
        defer { a.close(); b.close() }
        let before = mock.postedSendLengths.count
        try [UInt8]().withUnsafeBytes { try a.send($0) }
        XCTAssertEqual(mock.postedSendLengths.count, before)
    }

    /// The claim the whole slot design exists to make: every message on the wire
    /// is the same size, in both directions, always.
    func testEverySendAndReceiveIsExactlyOneSlot() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 4)
        let payload = (0..<70_000).map { UInt8($0 & 0xFF) }
        _ = try transfer(payload, geometry: geometry, mock: mock)

        XCTAssertFalse(mock.postedSendLengths.isEmpty)
        XCTAssertFalse(mock.postedReceiveLengths.isEmpty)
        let expected = UInt32(geometry.slotBytes)
        XCTAssertTrue(mock.postedSendLengths.allSatisfy { $0 == expected },
                      "send lengths: \(Set(mock.postedSendLengths))")
        XCTAssertTrue(mock.postedReceiveLengths.allSatisfy { $0 == expected },
                      "receive lengths: \(Set(mock.postedReceiveLengths))")
    }

    /// A reader asking for sizes the writer never used — which is exactly what
    /// `receiveFrame` does when it reads a 24-byte header then a payload.
    func testStreamSurvivesMismatchedReadSizes() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 8)
        let (a, b) = try makePair(mock, geometry: geometry)
        defer { a.close(); b.close() }

        let payload = (0..<20_000).map { UInt8($0 &* 17 & 0xFF) }
        try payload.withUnsafeBytes { try a.send($0) }

        var assembled: [UInt8] = []
        var offset = 0
        for size in [24, 1, 4095, 4096, 4097, 7, 6000] where offset < payload.count {
            let take = min(size, payload.count - offset)
            var part = [UInt8](repeating: 0, count: take)
            try part.withUnsafeMutableBytes { try b.receive(into: $0) }
            assembled += part
            offset += take
        }
        var rest = [UInt8](repeating: 0, count: payload.count - offset)
        if !rest.isEmpty { try rest.withUnsafeMutableBytes { try b.receive(into: $0) } }
        assembled += rest
        XCTAssertEqual(assembled, payload)
    }

    /// The real usage: `Channel.sendFrame` / `receiveFrame` over the queue pair.
    func testWireFrameRoundTripsOverTheQueuePair() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 8)
        let (a, b) = try makePair(mock, geometry: geometry)
        let channelA = RDMAChannel(connection: a, side: NullChannel(), deviceName: "rdma_en2")
        let channelB = RDMAChannel(connection: b, side: NullChannel(), deviceName: "rdma_en3")
        defer { channelA.close(); channelB.close() }

        let values: [Float] = (0..<4096).map { Float($0) * 0.5 }
        var header = WireHeader()
        header.tag = 0x1234
        header.elementCount = UInt32(values.count)
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount + values.count * 4)

        try values.withUnsafeBytes { raw in
            try channelA.sendFrame(header, payload: raw.baseAddress,
                                   payloadBytes: raw.count, using: scratch)
        }
        let received = ScratchBuffer(capacity: values.count * 4)
        let decoded = try channelB.receiveFrame(into: received, maxPayloadBytes: values.count * 4)
        XCTAssertEqual(decoded.tag, 0x1234)
        XCTAssertEqual(decoded.elementCount, UInt32(values.count))
        let out = UnsafeRawBufferPointer(start: received.base, count: values.count * 4)
        XCTAssertEqual(Array(out.bindMemory(to: Float.self)), values)
    }

    /// The receive ring is the flow-control window. A sender that overruns it
    /// with nobody reading is left with sends the hardware has accepted and not
    /// delivered — it does not fail, and it does not race ahead.
    func testSenderIsBackPressuredByTheReceiveRing() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 4)
        let (a, b) = try makePair(mock, geometry: geometry)
        defer { a.close(); b.close() }

        // A sender can run ahead by two rings before it stalls: `slotCount`
        // messages sitting in the peer's posted buffers, plus `slotCount` send
        // completions it has not yet consumed. Twelve slots into a four-slot
        // ring is comfortably past that, on a thread of its own so the expected
        // block does not stall the test.
        let payload = [UInt8](repeating: 0x5A, count: geometry.payloadCapacity * 12)
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue(label: "test.rdma.backpressure").async {
            payload.withUnsafeBytes { try? a.send($0) }
            finished.signal()
        }

        // The sender must still be blocked: two slots have nowhere to go.
        XCTAssertEqual(finished.wait(timeout: .now() + 0.25), .timedOut,
                       "a send larger than the receive ring should not complete unread")
        XCTAssertGreaterThan(mock.outstandingSendCount, 0)

        var received = [UInt8](repeating: 0, count: payload.count)
        try received.withUnsafeMutableBytes { try b.receive(into: $0) }
        XCTAssertEqual(finished.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(received, payload)
        XCTAssertEqual(mock.outstandingSendCount, 0)
    }

    /// UC does not retransmit. A dropped slot must stop the collective, not
    /// quietly hand back a tensor with a hole in it.
    func testLostSlotIsReportedAsASequenceGap() throws {
        let mock = MockVerbs()
        // Discard the second message. The third then lands in the slot the
        // second would have used, so its sequence number is one ahead.
        mock.dropSendIndices = [1]
        let geometry = try RDMASlotGeometry(slotBytes: 4096, slotCount: 8)
        let (a, b) = try makePair(mock, geometry: geometry)
        defer { a.close(); b.close() }

        let payload = [UInt8](repeating: 0xA5, count: geometry.payloadCapacity * 4)
        try payload.withUnsafeBytes { try a.send($0) }

        var received = [UInt8](repeating: 0, count: payload.count)
        XCTAssertThrowsError(try received.withUnsafeMutableBytes { try b.receive(into: $0) }) { error in
            guard case MCCLError.rdmaSequenceGap(let expected, let got) = error else {
                return XCTFail("expected a sequence gap, got \(error)")
            }
            XCTAssertEqual(expected, 1)
            XCTAssertEqual(got, 2)
        }
    }

    /// Full duplex, which is what every ring step actually does.
    func testConcurrentSendAndReceiveInBothDirections() throws {
        let mock = MockVerbs()
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 16)
        let (a, b) = try makePair(mock, geometry: geometry)
        defer { a.close(); b.close() }

        let fromA = (0..<40_000).map { UInt8($0 &* 3 & 0xFF) }
        let fromB = (0..<40_000).map { UInt8($0 &* 5 & 0xFF) }
        var intoB = [UInt8](repeating: 0, count: fromA.count)
        var intoA = [UInt8](repeating: 0, count: fromB.count)

        let queueOne = DispatchQueue(label: "test.rdma.one")
        let queueTwo = DispatchQueue(label: "test.rdma.two")
        try intoB.withUnsafeMutableBytes { bufferB in
            try intoA.withUnsafeMutableBytes { bufferA in
                try fromA.withUnsafeBytes { outA in
                    try fromB.withUnsafeBytes { outB in
                        try runConcurrently({
                            try a.send(outA)
                            try a.receive(into: bufferA)
                        }, on: queueOne, {
                            try b.receive(into: bufferB)
                            try b.send(outB)
                        }, on: queueTwo)
                    }
                }
            }
        }
        XCTAssertEqual(intoB, fromA)
        XCTAssertEqual(intoA, fromB)
    }
}

/// A `Channel` that does nothing, standing in for the TCP side-channel in tests
/// that only exercise the queue pair.
private final class NullChannel: Channel {
    let sendQueue = DispatchQueue(label: "test.rdma.null.tx")
    let receiveQueue = DispatchQueue(label: "test.rdma.null.rx")
    let peerDescription = "null"
    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws {}
    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws {}
    func close() {}
}

// MARK: - Transport, availability and addressing

final class RDMATransportTests: XCTestCase {

    func testDeviceNamesPairWithInterfaceNames() {
        XCTAssertEqual(RDMATransport.deviceName(pairedWith: "en2"), "rdma_en2")
        XCTAssertEqual(RDMATransport.interfaceName(pairedWith: "rdma_en2"), "en2")
        XCTAssertNil(RDMATransport.interfaceName(pairedWith: "en2"))
        XCTAssertNil(RDMATransport.interfaceName(pairedWith: "mlx5_0"))
    }

    func testUnavailabilityReportsAMissingLibrary() {
        let mock = MockVerbs()
        mock.loadFailure = "/usr/lib/librdma.dylib could not be loaded (no such file)"
        let reason = RDMATransport.unavailability(verbs: mock)
        guard case .libraryUnavailable(let detail)? = reason else {
            return XCTFail("expected libraryUnavailable, got \(String(describing: reason))")
        }
        XCTAssertTrue(detail.contains("librdma"))
        XCTAssertFalse(RDMATransport.isAvailable(verbs: mock))
    }

    func testUnavailabilityDistinguishesAnSDKWithoutHeaders() {
        let mock = MockVerbs()
        mock.loadFailure = "this build has no RDMA support: it was compiled against a macOS SDK "
            + "without <infiniband/verbs.h>, which needs the macOS 26.2 SDK or later"
        guard case .notCompiled? = RDMATransport.unavailability(verbs: mock) else {
            return XCTFail("expected notCompiled")
        }
    }

    /// The case every Mac without Thunderbolt 5 hits, and the one whose message
    /// has to tell the reader what to actually do.
    func testUnavailabilityReportsNoDevicesWithEnablementInstructions() {
        let mock = MockVerbs(devices: [])
        guard case .noDevices? = RDMATransport.unavailability(verbs: mock) else {
            return XCTFail("expected noDevices")
        }
        let reason = try! XCTUnwrap(RDMATransport.unusableReason(verbs: mock))
        XCTAssertTrue(reason.contains("rdma_ctl enable"), reason)
        XCTAssertTrue(reason.contains("Recovery"), reason)
        XCTAssertTrue(reason.contains("ibv_devices"), reason)
        XCTAssertTrue(reason.contains("Thunderbolt 5"), reason)
    }

    func testAvailableWhenDevicesArePresent() {
        let mock = MockVerbs(devices: ["rdma_en2"])
        XCTAssertNil(RDMATransport.unavailability(verbs: mock))
        XCTAssertTrue(RDMATransport.isAvailable(verbs: mock))
        XCTAssertNil(RDMATransport.unusableReason(verbs: mock))
        XCTAssertEqual(RDMATransport.deviceNames(verbs: mock), ["rdma_en2"])
    }

    func testDeviceNamesAreEmptyWhenTheLibraryIsMissing() {
        let mock = MockVerbs()
        mock.loadFailure = "no library"
        XCTAssertEqual(RDMATransport.deviceNames(verbs: mock), [])
    }

    // MARK: Metadata

    func testMetadataRoundTrips() throws {
        let metadata = RDMAQueuePairMetadata(
            globalID: try XCTUnwrap(RDMAGlobalID.ipv4Mapped("169.254.205.255")),
            localID: 3, queuePairNumber: 0x1234_5678,
            packetSequenceNumber: 0x00AB_CDEF, slotBytes: 65_536, slotCount: 32)
        let decoded = try RDMAQueuePairMetadata.decode(metadata.encoded())
        XCTAssertEqual(decoded, metadata)
        XCTAssertEqual(metadata.encoded().count, RDMAQueuePairMetadata.byteCount)
    }

    func testMetadataRejectsForeignMagic() {
        var bytes = [UInt8](repeating: 0, count: RDMAQueuePairMetadata.byteCount)
        bytes[0] = 0xFF
        XCTAssertThrowsError(try RDMAQueuePairMetadata.decode(bytes)) {
            XCTAssertTrue("\($0)".contains("magic"), "\($0)")
        }
    }

    func testMetadataRejectsAShortBuffer() {
        XCTAssertThrowsError(try RDMAQueuePairMetadata.decode([1, 2, 3]))
    }

    func testGlobalIDMapsIPv4BothWays() throws {
        let gid = try XCTUnwrap(RDMAGlobalID.ipv4Mapped("169.254.205.255"))
        XCTAssertTrue(gid.isIPv4Mapped)
        XCTAssertEqual(gid.ipv4Address, "169.254.205.255")
        XCTAssertEqual(gid.description, "::ffff:169.254.205.255")
        XCTAssertNil(RDMAGlobalID.ipv4Mapped("not.an.address"))
        XCTAssertFalse(RDMAGlobalID().isIPv4Mapped)
    }

    func testPacketSequenceNumbersFitTheField() {
        for _ in 0..<200 {
            XCTAssertLessThan(RDMATransport.randomPacketSequenceNumber(), 1 << 24)
        }
    }
}

// MARK: - Path selection

final class RDMAPathSelectionTests: XCTestCase {

    func testRDMARanksAboveEveryOtherMedium() {
        XCTAssertEqual(PathSelection.mediaOrder.first, .rdma)
        for media in [InterfaceMedia.thunderbolt, .ethernet, .wifi, .other, .loopback] {
            XCTAssertLessThan(PathSelection.priority(.rdma), PathSelection.priority(media),
                              "rdma should outrank \(media)")
        }
    }

    func testAPairedRDMADeviceClassifiesTheInterface() {
        XCTAssertEqual(
            NetworkInterfaces.classify(name: "en2", ifiType: 0x06, isLoopback: false,
                                       displayName: "Thunderbolt Bridge",
                                       rdmaInterfaces: ["en2"]),
            .rdma)
        // The same interface without a paired device is just Thunderbolt.
        XCTAssertEqual(
            NetworkInterfaces.classify(name: "en2", ifiType: 0x06, isLoopback: false,
                                       displayName: "Thunderbolt Bridge",
                                       rdmaInterfaces: []),
            .thunderbolt)
    }

    func testLoopbackIsNeverReclassifiedAsRDMA() {
        XCTAssertEqual(
            NetworkInterfaces.classify(name: "lo0", ifiType: 0x18, isLoopback: true,
                                       displayName: nil, rdmaInterfaces: ["lo0"]),
            .loopback)
    }

    /// A paired RDMA device settles the generation question: TN3205 ships only
    /// on Thunderbolt 5.
    func testRDMAInterfaceImpliesThunderbolt5() {
        let interface = NetworkInterface(
            name: "en2", media: .rdma, displayName: "Thunderbolt Bridge", isUp: true,
            isLoopback: false, linkSpeedBitsPerSecond: 0, addresses: [])
        XCTAssertEqual(interface.linkKind(measuredBandwidth: nil), .thunderbolt5)
        // Even a slow measurement does not downgrade it.
        XCTAssertEqual(interface.linkKind(measuredBandwidth: 1.0e8), .thunderbolt5)
    }

    func testRDMAEndpointsSortFirstInACandidateList() {
        let advertisement = PeerAdvertisement(rank: 1, endpoints: [
            PeerEndpoint(address: PeerAddress(host: "10.0.0.5", port: 100), media: .ethernet),
            PeerEndpoint(address: PeerAddress(host: "169.254.1.2", port: 100), media: .rdma),
            PeerEndpoint(address: PeerAddress(host: "192.168.1.9", port: 100), media: .wifi),
        ])
        // No local interface matches these, so every candidate has a nil egress
        // medium and the advertiser's own tag is what orders them.
        let candidates = PathSelection.candidates(for: advertisement, own: nil, interfaces: [])
        XCTAssertEqual(candidates.first?.endpoint.media, .rdma)
    }

    func testInterfaceMediaStillDecodesUnknownKinds() throws {
        // A rank running an older build must not choke on `.rdma`.
        XCTAssertEqual(InterfaceMedia(rawValue: "rdma"), .rdma)
        let json = #"{"address":{"host":"1.2.3.4","port":5},"media":"quantum-entanglement"}"#
        let endpoint = try JSONDecoder().decode(PeerEndpoint.self, from: Data(json.utf8))
        XCTAssertEqual(endpoint.media, .other)
    }
}

// MARK: - Hardware-gated integration

/// The tests that need a Thunderbolt 5 cable. They skip, loudly, everywhere else.
final class RDMAHardwareTests: XCTestCase {

    private func requireHardware() throws {
        if let reason = RDMATransport.unusableReason() {
            throw XCTSkip("""
                requires macOS 26.2+ with Thunderbolt 5; enable via `rdma_ctl enable` in Recovery \
                — see docs/RDMA.md. Reported reason: \(reason)
                """)
        }
    }

    /// Reports what this machine has, whether or not it has RDMA. Never fails:
    /// its job is to put the reason in the log where a first TB5 user will read
    /// it.
    func testAvailabilityIsReportedPrecisely() {
        if let reason = RDMATransport.unusableReason() {
            XCTAssertFalse(reason.isEmpty)
            print("[mccl] RDMA over Thunderbolt unavailable: \(reason)")
        } else {
            print("[mccl] RDMA devices: \(RDMATransport.deviceNames().joined(separator: ", "))")
        }
    }

    /// The library loads (or does not) without crashing, on every machine. This
    /// is the one thing the dlopen strategy has to get right everywhere.
    func testLibraryProbeIsSafeOnAnyMachine() {
        let verbs = SystemVerbs()
        let reason = verbs.load()
        if reason == nil {
            XCTAssertNoThrow(try verbs.deviceCount())
        } else {
            XCTAssertFalse(reason!.isEmpty)
        }
    }

    func testDevicesArePairedWithThunderboltInterfaces() throws {
        try requireHardware()
        let devices = RDMATransport.deviceNames()
        XCTAssertFalse(devices.isEmpty)
        let interfaces = Set(NetworkInterfaces.local().map(\.name))
        for device in devices {
            let paired = try XCTUnwrap(RDMATransport.interfaceName(pairedWith: device))
            XCTAssertTrue(interfaces.contains(paired),
                          "\(device) has no paired IP interface \(paired)")
        }
    }

    func testEveryDeviceReportsAnIPv4MappedGIDAtIndexOne() throws {
        try requireHardware()
        let verbs = SystemVerbs()
        for index in 0..<(try verbs.deviceCount()) {
            let context = try verbs.openDevice(at: index)
            defer { verbs.closeDevice(context) }
            let gid = try verbs.queryGID(context, port: RDMASpec.portNumber,
                                         index: Int(RDMASpec.gidIndex))
            XCTAssertTrue(gid.isIPv4Mapped, "GID 1 on device \(index) is \(gid)")
        }
    }

    func testLoopbackTransferOverRealHardware() throws {
        try requireHardware()
        // Two queue pairs on this machine's own devices, wired to each other.
        // Needs at least two RDMA devices with active ports.
        let verbs = SystemVerbs()
        guard try verbs.deviceCount() >= 2 else {
            throw XCTSkip("needs at least two RDMA devices")
        }
        let geometry = try RDMASlotGeometry(slotBytes: 8192, slotCount: 8)
        let a = try RDMAConnection(verbs: verbs, deviceIndex: 0,
                                   deviceName: try verbs.deviceName(at: 0),
                                   geometry: geometry, packetSequenceNumber: 11)
        let b = try RDMAConnection(verbs: verbs, deviceIndex: 1,
                                   deviceName: try verbs.deviceName(at: 1),
                                   geometry: geometry, packetSequenceNumber: 22)
        defer { a.close(); b.close() }
        guard a.isPortActive, b.isPortActive else {
            throw XCTSkip("needs two active Thunderbolt ports (is a cable connected?)")
        }
        try a.connect(to: b.metadata)
        try b.connect(to: a.metadata)

        let payload = (0..<30_000).map { UInt8($0 & 0xFF) }
        var received = [UInt8](repeating: 0, count: payload.count)
        try payload.withUnsafeBytes { try a.send($0) }
        try received.withUnsafeMutableBytes { try b.receive(into: $0) }
        XCTAssertEqual(received, payload)
    }
}
