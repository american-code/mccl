import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// What one side publishes about its queue pair so the other can reach it.
///
/// TN3205: "Applications should use an out-of-band communication mechanism to
/// share GID, LID, QPN, and PSN with a peer. Typically applications use
/// Thunderbolt IP and a TCP socket connection for this purpose." mccl already
/// has that TCP connection — it is the bootstrap channel `MeshFabric` dials —
/// so this rides on it rather than introducing a second rendezvous.
///
/// The slot geometry travels with it: both sides must agree on the one message
/// size before either posts a buffer (see `RDMASlotGeometry`).
public struct RDMAQueuePairMetadata: Equatable, Sendable {
    public var globalID: RDMAGlobalID
    public var localID: UInt16
    public var queuePairNumber: UInt32
    public var packetSequenceNumber: UInt32
    public var slotBytes: UInt32
    public var slotCount: UInt32

    public init(globalID: RDMAGlobalID, localID: UInt16, queuePairNumber: UInt32,
                packetSequenceNumber: UInt32, slotBytes: UInt32, slotCount: UInt32) {
        self.globalID = globalID
        self.localID = localID
        self.queuePairNumber = queuePairNumber
        self.packetSequenceNumber = packetSequenceNumber
        self.slotBytes = slotBytes
        self.slotCount = slotCount
    }

    /// Fixed 36-byte little-endian encoding: magic(4) gid(16) lid(2) pad(2)
    /// qpn(4) psn(4) slotBytes(4) slotCount(4). Sent over the TCP bootstrap
    /// channel as one write.
    static let magic: UInt32 = 0x304D_4452   // 'R','D','M','0' little-endian
    public static let byteCount = 40

    public func encoded() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: RDMAQueuePairMetadata.byteCount)
        out.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!
            p.storeBytes(of: RDMAQueuePairMetadata.magic.littleEndian, toByteOffset: 0, as: UInt32.self)
            for i in 0..<16 { p.storeBytes(of: globalID.bytes[i], toByteOffset: 4 + i, as: UInt8.self) }
            p.storeBytes(of: localID.littleEndian, toByteOffset: 20, as: UInt16.self)
            p.storeBytes(of: UInt16(0), toByteOffset: 22, as: UInt16.self)
            p.storeBytes(of: queuePairNumber.littleEndian, toByteOffset: 24, as: UInt32.self)
            p.storeBytes(of: packetSequenceNumber.littleEndian, toByteOffset: 28, as: UInt32.self)
            p.storeBytes(of: slotBytes.littleEndian, toByteOffset: 32, as: UInt32.self)
            p.storeBytes(of: slotCount.littleEndian, toByteOffset: 36, as: UInt32.self)
        }
        return out
    }

    public static func decode(_ bytes: [UInt8]) throws -> RDMAQueuePairMetadata {
        guard bytes.count >= byteCount else {
            throw MCCLError.protocolViolation(
                "RDMA metadata is \(bytes.count) bytes, expected \(byteCount)")
        }
        return try bytes.withUnsafeBytes { raw -> RDMAQueuePairMetadata in
            let p = raw.baseAddress!
            let magic = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
            guard magic == RDMAQueuePairMetadata.magic else {
                throw MCCLError.protocolViolation(
                    String(format: "bad RDMA metadata magic 0x%08x — the peer is not speaking "
                           + "mccl's RDMA bootstrap", magic))
            }
            var gid = [UInt8](repeating: 0, count: 16)
            for i in 0..<16 { gid[i] = p.loadUnaligned(fromByteOffset: 4 + i, as: UInt8.self) }
            return RDMAQueuePairMetadata(
                globalID: RDMAGlobalID(bytes: gid),
                localID: UInt16(littleEndian: p.loadUnaligned(fromByteOffset: 20, as: UInt16.self)),
                queuePairNumber: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 24, as: UInt32.self)),
                packetSequenceNumber: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 28, as: UInt32.self)),
                slotBytes: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 32, as: UInt32.self)),
                slotCount: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 36, as: UInt32.self)))
        }
    }
}

/// A page-aligned, registered ring of fixed-size slots.
///
/// TN3205 requires page alignment because each Thunderbolt controller sits
/// behind an IOMMU and can only reach memory mapped for it, so the allocation
/// goes through `posix_memalign` exactly as the technote's listing does.
///
/// Send and receive rings live in one allocation and one memory region: the
/// registration is the expensive part, and `IBV_ACCESS_LOCAL_WRITE` is the only
/// access flag either direction may carry, so there is nothing to separate them
/// for. The send ring occupies the first `ringBytes`, the receive ring the next.
final class RDMARing {
    let geometry: RDMASlotGeometry
    let base: UnsafeMutableRawPointer
    let totalBytes: Int
    private(set) var memoryRegion: RDMAMemoryRegionHandle?
    private(set) var localKey: UInt32 = 0
    private unowned let verbs: RDMAVerbs

    init(geometry: RDMASlotGeometry, verbs: RDMAVerbs,
         protectionDomain: RDMAProtectionDomainHandle) throws {
        self.geometry = geometry
        self.verbs = verbs
        self.totalBytes = geometry.ringBytes * 2

        let pageSize = Int(getpagesize())
        var allocation: UnsafeMutableRawPointer?
        let rc = posix_memalign(&allocation, pageSize, totalBytes)
        guard rc == 0, let allocation else {
            throw MCCLError.socketFailure("posix_memalign(\(totalBytes) bytes for RDMA ring)", errno: rc)
        }
        self.base = allocation

        do {
            let region = try verbs.registerMemoryRegion(
                protectionDomain, address: allocation, length: totalBytes,
                access: RDMASpec.accessLocalWrite)
            self.memoryRegion = region
            self.localKey = verbs.localKey(region)
        } catch {
            free(allocation)
            throw error
        }
    }

    /// Address of send slot `index`.
    func sendSlot(_ index: Int) -> UnsafeMutableRawPointer {
        base + index * geometry.slotBytes
    }

    /// Address of receive slot `index`.
    func receiveSlot(_ index: Int) -> UnsafeMutableRawPointer {
        base + geometry.ringBytes + index * geometry.slotBytes
    }

    func release() {
        if let region = memoryRegion {
            verbs.deregisterMemoryRegion(region)
            memoryRegion = nil
        }
        free(base)
    }
}

/// One RDMA connection: a device, a protection domain, a completion queue, a UC
/// queue pair and its ring.
///
/// The lifecycle is TN3205's, verbatim, and each step is asserted by
/// `RDMAStateMachineTests` against a mock:
///
/// ```
/// create  ->  Reset
/// modify  ->  Init   (IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)
/// modify  ->  RTR    (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)
/// modify  ->  RTS    (IBV_QP_STATE | IBV_QP_SQ_PSN)
/// ```
final class RDMAConnection {
    /// Which queue a completion came from, encoded into the top half of `wr_id`.
    enum Direction: UInt64 {
        case send = 0
        case receive = 1
    }

    let verbs: RDMAVerbs
    let geometry: RDMASlotGeometry
    let deviceName: String

    private let context: RDMAContextHandle
    private let protectionDomain: RDMAProtectionDomainHandle
    private let completionQueue: RDMACompletionQueueHandle
    private let queuePair: RDMAQueuePairHandle
    private let ring: RDMARing
    private let portAttributes: RDMAPortAttributes
    private let globalID: RDMAGlobalID
    private let localPSN: UInt32

    /// Guards the completion queue and the two pending lists. A `Channel` is
    /// driven from two threads at once — `sendQueue` and `receiveQueue` — and a
    /// single completion queue carries both directions, exactly as TN3205's
    /// example builds it. Whichever thread gets here first does the polling and
    /// files what it finds for the other.
    private let condition = NSCondition()
    private var pendingSend: [RDMACompletion] = []
    private var pendingReceive: [RDMACompletion] = []
    private var isPolling = false
    private var failure: Error?

    /// Send-side slot allocation.
    private var nextSendSlot = 0
    private var sendsInFlight = 0
    private var sendSequence = RDMASequenceChecker()

    /// Receive-side reassembly.
    private var receiveSequence = RDMASequenceChecker()
    private var leftover: [UInt8] = []
    private var leftoverOffset = 0
    private var peerClosed = false

    /// How long a single blocking send or receive waits before giving up.
    ///
    /// A socket has the kernel's own timeouts underneath it; a queue pair has
    /// none, so a cable pulled out mid-collective would otherwise poll for ever.
    var operationTimeout: TimeInterval = 60

    init(verbs: RDMAVerbs, deviceIndex: Int, deviceName: String,
         geometry: RDMASlotGeometry, packetSequenceNumber: UInt32) throws {
        self.verbs = verbs
        self.geometry = geometry
        self.deviceName = deviceName
        self.localPSN = packetSequenceNumber

        // Everything is built into locals first, so that a failure part-way
        // through can unwind what already exists without `self` — which is not
        // yet fully initialised — appearing in a cleanup closure.
        let context = try verbs.openDevice(at: deviceIndex)
        var unwinds: [() -> Void] = [{ verbs.closeDevice(context) }]
        func rollback(_ error: Error) -> Error {
            for undo in unwinds.reversed() { undo() }
            return error
        }

        do {
            let port = try verbs.queryPort(context, port: RDMASpec.portNumber)
            let gid = try verbs.queryGID(context, port: RDMASpec.portNumber,
                                         index: Int(RDMASpec.gidIndex))

            let pd = try verbs.allocateProtectionDomain(context)
            unwinds.append { verbs.deallocateProtectionDomain(pd) }

            let ring = try RDMARing(geometry: geometry, verbs: verbs, protectionDomain: pd)
            unwinds.append { ring.release() }

            // TN3205 creates the completion queue one entry deeper than the
            // requested depth. Both directions share it, as its listing does.
            let cq = try verbs.createCompletionQueue(context, depth: geometry.slotCount * 2 + 1)
            unwinds.append { verbs.destroyCompletionQueue(cq) }

            let qp = try verbs.createQueuePair(
                pd, completionQueue: cq,
                maxSendWorkRequests: geometry.slotCount,
                maxReceiveWorkRequests: geometry.slotCount,
                type: RDMASpec.queuePairType)
            unwinds.append { verbs.destroyQueuePair(qp) }

            // Reset -> Init.
            try verbs.modifyToInit(
                qp, port: RDMASpec.portNumber, pkeyIndex: RDMASpec.pkeyIndex,
                // "applications should only register memory as IBV_ACCESS_LOCAL_WRITE";
                // a UC queue pair that grants no remote access needs no access flags.
                accessFlags: 0, mask: RDMASpec.Mask.initialize)

            self.context = context
            self.portAttributes = port
            self.globalID = gid
            self.protectionDomain = pd
            self.ring = ring
            self.completionQueue = cq
            self.queuePair = qp
        } catch {
            throw rollback(error)
        }
    }

    /// What the peer needs to reach this queue pair.
    var metadata: RDMAQueuePairMetadata {
        RDMAQueuePairMetadata(
            globalID: globalID,
            localID: portAttributes.localID,
            queuePairNumber: verbs.queuePairNumber(queuePair),
            packetSequenceNumber: localPSN,
            slotBytes: UInt32(geometry.slotBytes),
            slotCount: UInt32(geometry.slotCount))
    }

    var isPortActive: Bool { portAttributes.isActive }

    /// Init -> RTR -> RTS, then fills the receive queue.
    ///
    /// Receives are posted before the function returns, which is what makes the
    /// same-size rule hold from the first message: the peer reaches RTS only
    /// after its own bootstrap read completes, and by then this side already has
    /// `slotCount` correctly-sized buffers waiting.
    func connect(to peer: RDMAQueuePairMetadata) throws {
        guard Int(peer.slotBytes) == geometry.slotBytes,
              Int(peer.slotCount) == geometry.slotCount else {
            throw MCCLError.protocolViolation(
                "RDMA slot geometry disagreement: this side has \(geometry.slotBytes)-byte slots "
                + "x\(geometry.slotCount), the peer announced \(peer.slotBytes) x\(peer.slotCount). "
                + "TN3205 requires both sides to post the same number of frames per message.")
        }

        try verbs.modifyToReadyToReceive(
            queuePair,
            attributes: RDMAReadyToReceive(
                pathMTU: RDMASpec.pathMTU,
                receivePSN: peer.packetSequenceNumber,
                destinationQueuePairNumber: peer.queuePairNumber,
                destinationGID: peer.globalID,
                destinationLocalID: peer.localID,
                sourceGIDIndex: RDMASpec.gidIndex,
                hopLimit: RDMASpec.hopLimit,
                port: RDMASpec.portNumber),
            mask: RDMASpec.Mask.readyToReceive)

        try verbs.modifyToReadyToSend(queuePair, sendPSN: localPSN,
                                      mask: RDMASpec.Mask.readyToSend)

        for slot in 0..<geometry.slotCount { try postReceive(slot: slot) }
    }

    // MARK: - Data path

    private func workRequestID(_ direction: Direction, slot: Int) -> UInt64 {
        (direction.rawValue << 32) | UInt64(slot)
    }

    private func decodeWorkRequestID(_ id: UInt64) -> (Direction, Int) {
        (Direction(rawValue: id >> 32) ?? .send, Int(id & 0xFFFF_FFFF))
    }

    private func postReceive(slot: Int) throws {
        try verbs.postReceive(
            queuePair,
            workRequestID: workRequestID(.receive, slot: slot),
            address: UInt64(UInt(bitPattern: ring.receiveSlot(slot))),
            length: UInt32(geometry.slotBytes),
            localKey: ring.localKey)
    }

    /// Writes `buffer` into as many slots as it takes and posts each one.
    func send(_ buffer: UnsafeRawBufferPointer) throws {
        guard let base = buffer.baseAddress, buffer.count > 0 else { return }
        var offset = 0
        while offset < buffer.count {
            let take = min(geometry.payloadCapacity, buffer.count - offset)
            try sendSlot(payload: base + offset, byteCount: take, closing: false)
            offset += take
        }
    }

    /// Emits one slot. Blocks only when every send slot is still in flight.
    private func sendSlot(payload: UnsafeRawPointer?, byteCount: Int, closing: Bool) throws {
        precondition(byteCount <= geometry.payloadCapacity)
        // Every send is signalled, so one completion frees exactly one slot.
        // Blocking here is the ring's back-pressure: the wire, not the caller,
        // decides how far ahead a sender may run.
        if sendsInFlight >= geometry.slotCount {
            try awaitCompletion(.send)
            sendsInFlight -= 1
        }

        let slot = nextSendSlot
        let address = ring.sendSlot(slot)
        var header = RDMASlotHeader()
        header.sequence = sendSequence.next()
        header.payloadBytes = UInt32(byteCount)
        header.flags = closing ? RDMASlotHeader.flagClosing : 0
        header.encode(into: address)
        if byteCount > 0, let payload {
            (address + RDMASlotHeader.byteCount).copyMemory(from: payload, byteCount: byteCount)
        }

        try verbs.postSend(
            queuePair,
            workRequestID: workRequestID(.send, slot: slot),
            address: UInt64(UInt(bitPattern: address)),
            // Every message on the wire is a full slot: that is the same-size
            // rule, and it is why `payloadBytes` in the header rather than the
            // work request's length is what bounds the useful bytes.
            length: UInt32(geometry.slotBytes),
            localKey: ring.localKey,
            sendFlags: RDMASpec.sendSignaled,
            opcode: RDMASpec.opcodeSend)

        sendsInFlight += 1
        nextSendSlot = (nextSendSlot + 1) % geometry.slotCount
    }

    /// Announces a clean shutdown so the peer can tell close from cable-pull.
    func sendClose() {
        try? sendSlot(payload: nil, byteCount: 0, closing: true)
    }

    /// Fills `buffer` completely, pulling and reposting slots as needed.
    func receive(into buffer: UnsafeMutableRawBufferPointer) throws {
        guard let base = buffer.baseAddress, buffer.count > 0 else { return }
        var filled = 0
        while filled < buffer.count {
            if leftoverOffset == leftover.count {
                try pullSlot()
            }
            let available = leftover.count - leftoverOffset
            guard available > 0 else { continue }
            let take = min(available, buffer.count - filled)
            leftover.withUnsafeBytes { raw in
                (base + filled).copyMemory(from: raw.baseAddress! + leftoverOffset, byteCount: take)
            }
            leftoverOffset += take
            filled += take
        }
    }

    /// Waits for one receive completion, validates it and stages its payload.
    private func pullSlot() throws {
        if peerClosed { throw MCCLError.connectionClosed }
        let completion = try awaitCompletion(.receive)
        let (_, slot) = decodeWorkRequestID(completion.workRequestID)
        guard slot >= 0, slot < geometry.slotCount else {
            throw MCCLError.protocolViolation(
                "RDMA completion names receive slot \(slot), outside 0..<\(geometry.slotCount)")
        }
        let address = ring.receiveSlot(slot)
        let header = try RDMASlotHeader.decode(from: address, capacity: geometry.payloadCapacity)
        try receiveSequence.accept(header.sequence)

        if header.isClosing {
            peerClosed = true
            try postReceive(slot: slot)
            if header.payloadBytes == 0 { throw MCCLError.connectionClosed }
        }

        let count = Int(header.payloadBytes)
        leftover = [UInt8](repeating: 0, count: count)
        if count > 0 {
            leftover.withUnsafeMutableBytes { raw in
                raw.baseAddress!.copyMemory(from: address + RDMASlotHeader.byteCount, byteCount: count)
            }
        }
        leftoverOffset = 0
        // The slot is free the moment its bytes are copied out, and reposting it
        // immediately is what keeps `slotCount` buffers permanently available —
        // the invariant the same-size rule depends on.
        try postReceive(slot: slot)
    }

    // MARK: - Completion handling

    /// Blocks until a completion for `direction` is available.
    ///
    /// One completion queue, two waiting threads. Whichever thread arrives first
    /// takes the polling role; the other waits on the condition and is woken
    /// with whatever was found. A failed completion poisons the connection for
    /// both directions, because a UC queue pair that has errored will not
    /// recover on its own.
    @discardableResult
    private func awaitCompletion(_ direction: Direction) throws -> RDMACompletion {
        let deadline = Date().addingTimeInterval(operationTimeout)
        condition.lock()
        defer { condition.unlock() }

        while true {
            if let error = failure { throw error }
            if let completion = take(direction) { return completion }
            guard Date() < deadline else {
                throw MCCLError.timedOut(
                    "no RDMA \(direction == .send ? "send" : "receive") completion on \(deviceName) "
                    + "in \(Int(operationTimeout))s")
            }

            if isPolling {
                condition.wait(until: Date().addingTimeInterval(0.001))
                continue
            }

            isPolling = true
            condition.unlock()
            var polled: [RDMACompletion] = []
            var pollError: Error?
            do {
                polled = try verbs.pollCompletionQueue(completionQueue, maxEntries: 16)
            } catch {
                pollError = error
            }
            // An empty poll is the common case; yield rather than spin hot,
            // because two mccl threads per peer would otherwise pin two cores.
            if polled.isEmpty && pollError == nil { sched_yield() }
            condition.lock()
            isPolling = false

            if let pollError {
                failure = pollError
                condition.broadcast()
                throw pollError
            }
            for completion in polled {
                guard completion.isSuccess else {
                    let text = verbs.statusDescription(completion.status)
                    let error = MCCLError.rdmaCompletionFailed(
                        status: completion.status, description: text)
                    failure = error
                    condition.broadcast()
                    throw error
                }
                switch decodeWorkRequestID(completion.workRequestID).0 {
                case .send: pendingSend.append(completion)
                case .receive: pendingReceive.append(completion)
                }
            }
            condition.broadcast()
        }
    }

    private func take(_ direction: Direction) -> RDMACompletion? {
        switch direction {
        case .send: return pendingSend.isEmpty ? nil : pendingSend.removeFirst()
        case .receive: return pendingReceive.isEmpty ? nil : pendingReceive.removeFirst()
        }
    }

    // MARK: - Teardown

    private var isClosed = false

    func close() {
        guard !isClosed else { return }
        isClosed = true
        verbs.destroyQueuePair(queuePair)
        verbs.destroyCompletionQueue(completionQueue)
        ring.release()
        verbs.deallocateProtectionDomain(protectionDomain)
        verbs.closeDevice(context)
    }

    deinit { close() }
}
