import Foundation
@testable import MCCL

/// An in-process implementation of `RDMAVerbs` that behaves like TN3205 says
/// Apple's hardware behaves.
///
/// This is the whole answer to "how do you test a transport for hardware you do
/// not own". It is not a stub that returns success: it enforces the rules the
/// real device enforces, and the interesting tests are the ones where it
/// *refuses*.
///
/// What it models, and why each one is here:
///
/// * **The queue-pair state machine.** Posting a send before RTS, or receiving
///   before RTR, fails — which is the failure a mis-ordered bring-up would
///   produce on hardware.
/// * **The same-size rule.** "If the sender sends a 16K message and the receiver
///   posts a 4K or 32K receive buffer the receive operation will fail." The mock
///   fails it, with a distinguishable status, so the framing's central claim is
///   checked rather than assumed.
/// * **Receive-buffer exhaustion.** A send with no receive posted fails, as
///   TN3205 implies ("Completion of a send operation indicates that the peer has
///   posted sufficient receive requests").
/// * **The published ceilings.** Ten queue pairs, 4095 work requests,
///   16,773,120-byte messages.
/// * **UC's unreliability**, via `dropSendIndices` — a send that completes
///   successfully and never arrives, which is exactly the failure a sequence
///   number exists to catch.
///
/// Deliberately *not* modelled: reordering and corruption. A gap check cannot
/// detect corruption, and pretending to test that would be worse than admitting
/// it. See `RDMASequenceChecker`.
final class MockVerbs: RDMAVerbs, @unchecked Sendable {

    // MARK: Configuration

    /// Device names, in the order `ibv_get_device_list` would return them.
    var devices: [String]
    /// Set non-nil to make `load()` report the library as unusable.
    var loadFailure: String?
    /// Port state reported by `queryPort`; `IBV_PORT_ACTIVE` is 4.
    var portState: UInt32 = 4
    /// The IPv4 addresses whose v4-mapped form each device reports at GID 1.
    var gidAddresses: [String: String] = [:]

    /// Zero-based indices of sends to accept, complete successfully, and then
    /// silently discard — the shape of a lost message on a UC queue pair.
    var dropSendIndices: Set<Int> = []

    // MARK: Recording

    /// Every verbs call, in order, as a short string. Assertions read this.
    private(set) var transcript: [String] = []
    /// Each `ibv_modify_qp` call: the state asked for and the mask passed.
    private(set) var transitions: [(state: String, mask: Int32)] = []
    private(set) var registeredRegions: [(address: UInt, length: Int, access: Int32)] = []
    private(set) var createdQueuePairs: [(maxSend: Int, maxRecv: Int, type: Int32)] = []
    private(set) var readyToReceiveAttributes: [RDMAReadyToReceive] = []
    private(set) var completionQueueDepths: [Int] = []
    /// Every `length` ever handed to `ibv_post_send` / `ibv_post_recv`. The
    /// same-size rule is a claim about these two lists.
    private(set) var postedSendLengths: [UInt32] = []
    private(set) var postedReceiveLengths: [UInt32] = []

    /// Live-object counters, so teardown can be checked for leaks.
    private(set) var openContexts = 0
    private(set) var liveProtectionDomains = 0
    private(set) var liveMemoryRegions = 0
    private(set) var liveCompletionQueues = 0
    private(set) var liveQueuePairs = 0

    private let lock = NSRecursiveLock()
    private var nextHandle: UInt = 1
    private var sendCounter = 0

    init(devices: [String] = ["rdma_en2", "rdma_en3"]) {
        self.devices = devices
    }

    // MARK: Internal object tables

    private enum State: Int, Comparable {
        case reset = 0, initialized = 1, readyToReceive = 2, readyToSend = 3
        static func < (a: State, b: State) -> Bool { a.rawValue < b.rawValue }
    }

    private struct PostedReceive {
        var workRequestID: UInt64
        var address: UInt
        var length: UInt32
    }

    private final class QueuePair {
        var number: UInt32
        var state: State = .reset
        var completionQueue: UInt
        var maxSend: Int
        var maxRecv: Int
        var posted: [PostedReceive] = []
        var peerNumber: UInt32?
        init(number: UInt32, completionQueue: UInt, maxSend: Int, maxRecv: Int) {
            self.number = number
            self.completionQueue = completionQueue
            self.maxSend = maxSend
            self.maxRecv = maxRecv
        }
    }

    /// A send the hardware has accepted but not yet delivered, because the peer
    /// has no receive buffer posted.
    ///
    /// TN3205: "Completion of a send operation indicates that the peer has posted
    /// sufficient receive requests to receive all sent frames and that the
    /// Thunderbolt hardware has sent the buffers to the peer." So a send with
    /// nowhere to land does not fail — it stays outstanding, and completes when
    /// the peer posts. Modelling that faithfully is what makes the sender's
    /// back-pressure real: a rank that only sends, never receiving, blocks, in
    /// the mock exactly as it would on a cable.
    private struct PendingSend {
        var sourceAddress: UInt
        var length: UInt32
        var senderCompletionQueue: UInt
        var senderWorkRequestID: UInt64
    }
    private var pendingSends: [UInt32: [PendingSend]] = [:]

    private var contexts: Set<UInt> = []
    private var contextDevice: [UInt: String] = [:]
    private var protectionDomains: Set<UInt> = []
    private var memoryRegions: [UInt: UInt32] = [:]
    private var completionQueues: [UInt: [RDMACompletion]] = [:]
    private var queuePairs: [UInt: QueuePair] = [:]
    private var queuePairsByNumber: [UInt32: UInt] = [:]

    private func allocate() -> UInt {
        defer { nextHandle += 1 }
        return nextHandle
    }

    private func record(_ what: String) { transcript.append(what) }

    // MARK: - RDMAVerbs

    func load() -> String? {
        lock.lock(); defer { lock.unlock() }
        record("load")
        return loadFailure
    }

    func deviceCount() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        if let loadFailure { throw MCCLError.rdmaUnavailable(loadFailure) }
        return devices.count
    }

    func deviceName(at index: Int) throws -> String {
        lock.lock(); defer { lock.unlock() }
        guard devices.indices.contains(index) else {
            throw MCCLError.rdmaFailure("no device \(index)", code: -1)
        }
        return devices[index]
    }

    func openDevice(at index: Int) throws -> RDMAContextHandle {
        lock.lock(); defer { lock.unlock() }
        guard devices.indices.contains(index) else {
            throw MCCLError.rdmaFailure("no device \(index)", code: -1)
        }
        let handle = allocate()
        contexts.insert(handle)
        contextDevice[handle] = devices[index]
        openContexts += 1
        record("open(\(devices[index]))")
        return RDMAContextHandle(raw: handle)
    }

    func closeDevice(_ context: RDMAContextHandle) {
        lock.lock(); defer { lock.unlock() }
        if contexts.remove(context.raw) != nil { openContexts -= 1 }
        record("close-device")
    }

    func queryPort(_ context: RDMAContextHandle, port: UInt8) throws -> RDMAPortAttributes {
        lock.lock(); defer { lock.unlock() }
        record("query-port(\(port))")
        return RDMAPortAttributes(localID: 1, state: portState)
    }

    func queryGID(_ context: RDMAContextHandle, port: UInt8, index: Int) throws -> RDMAGlobalID {
        lock.lock(); defer { lock.unlock() }
        record("query-gid(port \(port), index \(index))")
        let device = contextDevice[context.raw] ?? ""
        let address = gidAddresses[device] ?? "169.254.\(port).\(index)"
        return RDMAGlobalID.ipv4Mapped(address) ?? RDMAGlobalID()
    }

    func allocateProtectionDomain(_ context: RDMAContextHandle) throws -> RDMAProtectionDomainHandle {
        lock.lock(); defer { lock.unlock() }
        let handle = allocate()
        protectionDomains.insert(handle)
        liveProtectionDomains += 1
        record("alloc-pd")
        return RDMAProtectionDomainHandle(raw: handle)
    }

    func deallocateProtectionDomain(_ pd: RDMAProtectionDomainHandle) {
        lock.lock(); defer { lock.unlock() }
        if protectionDomains.remove(pd.raw) != nil { liveProtectionDomains -= 1 }
        record("dealloc-pd")
    }

    func registerMemoryRegion(_ pd: RDMAProtectionDomainHandle, address: UnsafeMutableRawPointer,
                              length: Int, access: Int32) throws -> RDMAMemoryRegionHandle {
        lock.lock(); defer { lock.unlock() }
        // TN3205: page-aligned, because the controller sits behind an IOMMU.
        let pageSize = Int(getpagesize())
        guard UInt(bitPattern: address) % UInt(pageSize) == 0 else {
            throw MCCLError.rdmaFailure("ibv_reg_mr: buffer is not page-aligned", code: -22)
        }
        let handle = allocate()
        memoryRegions[handle] = UInt32(handle &* 7 &+ 1)
        liveMemoryRegions += 1
        registeredRegions.append((UInt(bitPattern: address), length, access))
        record("reg-mr(\(length), access \(access))")
        return RDMAMemoryRegionHandle(raw: handle)
    }

    func localKey(_ mr: RDMAMemoryRegionHandle) -> UInt32 {
        lock.lock(); defer { lock.unlock() }
        return memoryRegions[mr.raw] ?? 0
    }

    func deregisterMemoryRegion(_ mr: RDMAMemoryRegionHandle) {
        lock.lock(); defer { lock.unlock() }
        if memoryRegions.removeValue(forKey: mr.raw) != nil { liveMemoryRegions -= 1 }
        record("dereg-mr")
    }

    func createCompletionQueue(_ context: RDMAContextHandle, depth: Int) throws -> RDMACompletionQueueHandle {
        lock.lock(); defer { lock.unlock() }
        let handle = allocate()
        completionQueues[handle] = []
        liveCompletionQueues += 1
        completionQueueDepths.append(depth)
        record("create-cq(\(depth))")
        return RDMACompletionQueueHandle(raw: handle)
    }

    func destroyCompletionQueue(_ cq: RDMACompletionQueueHandle) {
        lock.lock(); defer { lock.unlock() }
        if completionQueues.removeValue(forKey: cq.raw) != nil { liveCompletionQueues -= 1 }
        record("destroy-cq")
    }

    func createQueuePair(_ pd: RDMAProtectionDomainHandle, completionQueue: RDMACompletionQueueHandle,
                         maxSendWorkRequests: Int, maxReceiveWorkRequests: Int,
                         type: Int32) throws -> RDMAQueuePairHandle {
        lock.lock(); defer { lock.unlock() }
        guard liveQueuePairs < RDMASpec.maxQueuePairs else {
            throw MCCLError.rdmaFailure(
                "ibv_create_qp: at the \(RDMASpec.maxQueuePairs) queue-pair limit", code: -12)
        }
        guard maxSendWorkRequests <= RDMASpec.maxWorkRequests,
              maxReceiveWorkRequests <= RDMASpec.maxWorkRequests else {
            throw MCCLError.rdmaFailure(
                "ibv_create_qp: above the \(RDMASpec.maxWorkRequests) work-request limit", code: -22)
        }
        let handle = allocate()
        let number = UInt32(truncatingIfNeeded: handle &+ 1000)
        queuePairs[handle] = QueuePair(number: number, completionQueue: completionQueue.raw,
                                       maxSend: maxSendWorkRequests, maxRecv: maxReceiveWorkRequests)
        queuePairsByNumber[number] = handle
        liveQueuePairs += 1
        createdQueuePairs.append((maxSendWorkRequests, maxReceiveWorkRequests, type))
        record("create-qp(type \(type))")
        return RDMAQueuePairHandle(raw: handle)
    }

    func queuePairNumber(_ qp: RDMAQueuePairHandle) -> UInt32 {
        lock.lock(); defer { lock.unlock() }
        return queuePairs[qp.raw]?.number ?? 0
    }

    func destroyQueuePair(_ qp: RDMAQueuePairHandle) {
        lock.lock(); defer { lock.unlock() }
        if let pair = queuePairs.removeValue(forKey: qp.raw) {
            queuePairsByNumber.removeValue(forKey: pair.number)
            liveQueuePairs -= 1
        }
        record("destroy-qp")
    }

    func modifyToInit(_ qp: RDMAQueuePairHandle, port: UInt8, pkeyIndex: UInt16,
                      accessFlags: UInt32, mask: Int32) throws {
        lock.lock(); defer { lock.unlock() }
        guard let pair = queuePairs[qp.raw] else {
            throw MCCLError.rdmaFailure("modify_qp: no such queue pair", code: -22)
        }
        guard pair.state == .reset else {
            throw MCCLError.rdmaFailure("modify_qp -> INIT from \(pair.state)", code: -22)
        }
        pair.state = .initialized
        transitions.append(("INIT", mask))
        record("modify->INIT(port \(port), mask \(mask))")
    }

    func modifyToReadyToReceive(_ qp: RDMAQueuePairHandle, attributes: RDMAReadyToReceive,
                                mask: Int32) throws {
        lock.lock(); defer { lock.unlock() }
        guard let pair = queuePairs[qp.raw] else {
            throw MCCLError.rdmaFailure("modify_qp: no such queue pair", code: -22)
        }
        guard pair.state == .initialized else {
            throw MCCLError.rdmaFailure("modify_qp -> RTR from \(pair.state)", code: -22)
        }
        pair.state = .readyToReceive
        pair.peerNumber = attributes.destinationQueuePairNumber
        transitions.append(("RTR", mask))
        readyToReceiveAttributes.append(attributes)
        record("modify->RTR(mask \(mask))")
    }

    func modifyToReadyToSend(_ qp: RDMAQueuePairHandle, sendPSN: UInt32, mask: Int32) throws {
        lock.lock(); defer { lock.unlock() }
        guard let pair = queuePairs[qp.raw] else {
            throw MCCLError.rdmaFailure("modify_qp: no such queue pair", code: -22)
        }
        guard pair.state == .readyToReceive else {
            throw MCCLError.rdmaFailure("modify_qp -> RTS from \(pair.state)", code: -22)
        }
        pair.state = .readyToSend
        transitions.append(("RTS", mask))
        record("modify->RTS(mask \(mask))")
    }

    func postReceive(_ qp: RDMAQueuePairHandle, workRequestID: UInt64,
                     address: UInt64, length: UInt32, localKey: UInt32) throws {
        lock.lock(); defer { lock.unlock() }
        guard let pair = queuePairs[qp.raw] else {
            throw MCCLError.rdmaFailure("post_recv: no such queue pair", code: -22)
        }
        guard pair.state >= .initialized else {
            throw MCCLError.rdmaFailure("post_recv before INIT", code: -22)
        }
        guard pair.posted.count < pair.maxRecv else {
            throw MCCLError.rdmaFailure("post_recv: receive queue is full", code: -12)
        }
        pair.posted.append(PostedReceive(workRequestID: workRequestID,
                                         address: UInt(address), length: length))
        postedReceiveLengths.append(length)
        // A buffer arriving may unblock a sender that has been waiting for one.
        drainPendingSends(to: pair)
    }

    func postSend(_ qp: RDMAQueuePairHandle, workRequestID: UInt64, address: UInt64,
                  length: UInt32, localKey: UInt32, sendFlags: UInt32, opcode: Int32) throws {
        lock.lock(); defer { lock.unlock() }
        guard let pair = queuePairs[qp.raw] else {
            throw MCCLError.rdmaFailure("post_send: no such queue pair", code: -22)
        }
        guard pair.state == .readyToSend else {
            throw MCCLError.rdmaFailure("post_send before RTS (state \(pair.state))", code: -22)
        }
        guard opcode == RDMASpec.opcodeSend else {
            throw MCCLError.rdmaFailure("post_send: only IBV_WR_SEND is supported", code: -95)
        }
        guard Int(length) <= RDMASpec.maxMessageBytes else {
            throw MCCLError.rdmaFailure(
                "post_send: \(length) bytes exceeds the \(RDMASpec.maxMessageBytes)-byte maximum",
                code: -22)
        }

        postedSendLengths.append(length)
        let index = sendCounter
        sendCounter += 1

        if dropSendIndices.contains(index) {
            // A lost message on a UC queue pair: the send completes — the
            // hardware did put the frames on the wire — and nothing arrives.
            // The peer's buffer is left posted, so the *next* message lands in
            // it and its sequence number is one ahead. That is precisely the
            // discontinuity `RDMASequenceChecker` exists to catch.
            complete(on: pair.completionQueue, workRequestID: workRequestID,
                     status: 0, opcode: UInt32(RDMASpec.opcodeSend), byteCount: length)
            return
        }

        guard let peerNumber = pair.peerNumber,
              let peerHandle = queuePairsByNumber[peerNumber],
              let peer = queuePairs[peerHandle], peer.state >= .readyToReceive else {
            throw MCCLError.rdmaFailure("post_send: peer queue pair is not ready", code: -22)
        }

        pendingSends[peerNumber, default: []].append(
            PendingSend(sourceAddress: UInt(address), length: length,
                        senderCompletionQueue: pair.completionQueue,
                        senderWorkRequestID: workRequestID))
        drainPendingSends(to: peer)
    }

    /// Delivers as many outstanding sends as `pair` has buffers posted for.
    private func drainPendingSends(to pair: QueuePair) {
        while !pair.posted.isEmpty, var queue = pendingSends[pair.number], !queue.isEmpty {
            let send = queue.removeFirst()
            pendingSends[pair.number] = queue
            let receive = pair.posted.removeFirst()

            // The same-size rule. This is the assertion the whole slot design
            // exists to satisfy, so the mock enforces it exactly as TN3205
            // states it: "if the sender sends a 16K message and the receiver
            // posts a 4K or 32K receive buffer the receive operation will fail".
            guard receive.length == send.length else {
                complete(on: pair.completionQueue, workRequestID: receive.workRequestID,
                         status: MockVerbs.lengthMismatchStatus, opcode: 128, byteCount: 0)
                complete(on: send.senderCompletionQueue, workRequestID: send.senderWorkRequestID,
                         status: MockVerbs.lengthMismatchStatus,
                         opcode: UInt32(RDMASpec.opcodeSend), byteCount: 0)
                continue
            }

            if let destination = UnsafeMutableRawPointer(bitPattern: receive.address),
               let source = UnsafeRawPointer(bitPattern: send.sourceAddress) {
                destination.copyMemory(from: source, byteCount: Int(send.length))
            }
            complete(on: pair.completionQueue, workRequestID: receive.workRequestID,
                     status: 0, opcode: 128, byteCount: send.length)
            complete(on: send.senderCompletionQueue, workRequestID: send.senderWorkRequestID,
                     status: 0, opcode: UInt32(RDMASpec.opcodeSend), byteCount: send.length)
        }
    }

    /// Sends the hardware has accepted but not yet delivered, for assertions
    /// about back-pressure.
    var outstandingSendCount: Int {
        lock.lock(); defer { lock.unlock() }
        return pendingSends.values.reduce(0) { $0 + $1.count }
    }

    /// `IBV_WC_LOC_LEN_ERR` is 1 in rdma-core; any non-zero status would do, but
    /// using the real one keeps the failure recognisable.
    static let lengthMismatchStatus: UInt32 = 1

    private func complete(on cq: UInt, workRequestID: UInt64, status: UInt32,
                          opcode: UInt32, byteCount: UInt32) {
        completionQueues[cq, default: []].append(
            RDMACompletion(workRequestID: workRequestID, status: status,
                           opcode: opcode, byteCount: byteCount))
    }

    func pollCompletionQueue(_ cq: RDMACompletionQueueHandle, maxEntries: Int) throws -> [RDMACompletion] {
        lock.lock(); defer { lock.unlock() }
        guard var queue = completionQueues[cq.raw], !queue.isEmpty else { return [] }
        let take = min(maxEntries, queue.count)
        let out = Array(queue.prefix(take))
        queue.removeFirst(take)
        completionQueues[cq.raw] = queue
        return out
    }

    func statusDescription(_ status: UInt32) -> String {
        status == 0 ? "success"
            : (status == MockVerbs.lengthMismatchStatus ? "local length error" : "error \(status)")
    }

    // MARK: - Test helpers

    /// True when every object this mock handed out has been given back.
    var isFullyReleased: Bool {
        openContexts == 0 && liveProtectionDomains == 0 && liveMemoryRegions == 0
            && liveCompletionQueues == 0 && liveQueuePairs == 0
    }

    /// The masks passed to `ibv_modify_qp`, keyed by target state.
    func mask(for state: String) -> Int32? {
        transitions.first(where: { $0.state == state })?.mask
    }
}
