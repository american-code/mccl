import Foundation
import CRDMA

/// The verbs surface mccl uses, as a protocol.
///
/// Everything above this — the queue-pair state machine, the slot ring, the
/// framing, the gap check — is written against `RDMAVerbs` rather than against
/// `CRDMA` directly. That is the whole reason a transport for hardware we do not
/// own can be tested at all: `MockVerbs` in the test target implements this and
/// drives the same state machine the real device would.
///
/// The vocabulary is deliberately Apple's, not a friendlier abstraction of it.
/// TN3205's constraints — UC queue pairs, one port, `IBV_MTU_4096`, send/recv
/// only, `IBV_ACCESS_LOCAL_WRITE` only — are the shape of this API, and hiding
/// them would only mean rediscovering them on hardware.
public protocol RDMAVerbs: AnyObject {
    /// Binds the library. Idempotent. Returns nil on success, or the reason.
    func load() -> String?

    func deviceCount() throws -> Int
    func deviceName(at index: Int) throws -> String

    func openDevice(at index: Int) throws -> RDMAContextHandle
    func closeDevice(_ context: RDMAContextHandle)

    func queryPort(_ context: RDMAContextHandle, port: UInt8) throws -> RDMAPortAttributes
    func queryGID(_ context: RDMAContextHandle, port: UInt8, index: Int) throws -> RDMAGlobalID

    func allocateProtectionDomain(_ context: RDMAContextHandle) throws -> RDMAProtectionDomainHandle
    func deallocateProtectionDomain(_ pd: RDMAProtectionDomainHandle)

    func registerMemoryRegion(
        _ pd: RDMAProtectionDomainHandle,
        address: UnsafeMutableRawPointer,
        length: Int,
        access: Int32
    ) throws -> RDMAMemoryRegionHandle
    func localKey(_ mr: RDMAMemoryRegionHandle) -> UInt32
    func deregisterMemoryRegion(_ mr: RDMAMemoryRegionHandle)

    func createCompletionQueue(_ context: RDMAContextHandle, depth: Int) throws -> RDMACompletionQueueHandle
    func destroyCompletionQueue(_ cq: RDMACompletionQueueHandle)

    func createQueuePair(
        _ pd: RDMAProtectionDomainHandle,
        completionQueue: RDMACompletionQueueHandle,
        maxSendWorkRequests: Int,
        maxReceiveWorkRequests: Int,
        type: Int32
    ) throws -> RDMAQueuePairHandle
    func queuePairNumber(_ qp: RDMAQueuePairHandle) -> UInt32
    func destroyQueuePair(_ qp: RDMAQueuePairHandle)

    func modifyToInit(_ qp: RDMAQueuePairHandle, port: UInt8, pkeyIndex: UInt16,
                      accessFlags: UInt32, mask: Int32) throws
    func modifyToReadyToReceive(_ qp: RDMAQueuePairHandle, attributes: RDMAReadyToReceive,
                                mask: Int32) throws
    func modifyToReadyToSend(_ qp: RDMAQueuePairHandle, sendPSN: UInt32, mask: Int32) throws

    func postSend(_ qp: RDMAQueuePairHandle, workRequestID: UInt64,
                  address: UInt64, length: UInt32, localKey: UInt32,
                  sendFlags: UInt32, opcode: Int32) throws
    func postReceive(_ qp: RDMAQueuePairHandle, workRequestID: UInt64,
                     address: UInt64, length: UInt32, localKey: UInt32) throws

    /// Returns up to `maxEntries` completions; an empty array means "nothing
    /// ready", which is the overwhelmingly common answer and is not an error.
    func pollCompletionQueue(_ cq: RDMACompletionQueueHandle, maxEntries: Int) throws -> [RDMACompletion]

    func statusDescription(_ status: UInt32) -> String
}

// MARK: - Handles

/// Opaque handles. The real implementation stores the `struct ibv_*` pointer;
/// the mock stores an index into its own tables. Neither is interpreted here.
public struct RDMAContextHandle: Hashable, Sendable { public let raw: UInt; public init(raw: UInt) { self.raw = raw } }
public struct RDMAProtectionDomainHandle: Hashable, Sendable { public let raw: UInt; public init(raw: UInt) { self.raw = raw } }
public struct RDMAMemoryRegionHandle: Hashable, Sendable { public let raw: UInt; public init(raw: UInt) { self.raw = raw } }
public struct RDMACompletionQueueHandle: Hashable, Sendable { public let raw: UInt; public init(raw: UInt) { self.raw = raw } }
public struct RDMAQueuePairHandle: Hashable, Sendable { public let raw: UInt; public init(raw: UInt) { self.raw = raw } }

// MARK: - Value types

public struct RDMAPortAttributes: Sendable, Equatable {
    public var localID: UInt16
    public var state: UInt32
    public init(localID: UInt16, state: UInt32) {
        self.localID = localID
        self.state = state
    }
    /// `IBV_PORT_ACTIVE`. TN3205's `ibv_devinfo` output shows `PORT_ACTIVE (4)`
    /// for a Thunderbolt port with a cable in it.
    public var isActive: Bool { state == UInt32(MCCL_IBV_PORT_ACTIVE) }
}

/// A 128-bit GID. TN3205: index 1 is the IPv4-mapped address of the paired
/// IP-over-Thunderbolt interface (`::ffff:169.254.205.255`), index 2 the
/// link-local IPv6. mccl exchanges index 1.
public struct RDMAGlobalID: Hashable, Sendable, CustomStringConvertible {
    public var bytes: [UInt8]

    public init(bytes: [UInt8]) {
        precondition(bytes.count == 16, "a GID is 16 bytes")
        self.bytes = bytes
    }

    public init() { self.bytes = [UInt8](repeating: 0, count: 16) }

    /// True when this is an IPv4-mapped GID (`::ffff:a.b.c.d`), which is the
    /// form index 1 takes.
    public var isIPv4Mapped: Bool {
        bytes[0..<10].allSatisfy { $0 == 0 } && bytes[10] == 0xFF && bytes[11] == 0xFF
    }

    /// The paired interface's IPv4 address, when this is an IPv4-mapped GID.
    public var ipv4Address: String? {
        guard isIPv4Mapped else { return nil }
        return "\(bytes[12]).\(bytes[13]).\(bytes[14]).\(bytes[15])"
    }

    /// Builds the IPv4-mapped GID for a dotted-quad address. Used by the mock
    /// and by tests; the real path always reads the GID off the device.
    public static func ipv4Mapped(_ address: String) -> RDMAGlobalID? {
        let parts = address.split(separator: ".").compactMap { UInt8($0) }
        guard parts.count == 4 else { return nil }
        var bytes = [UInt8](repeating: 0, count: 16)
        bytes[10] = 0xFF
        bytes[11] = 0xFF
        bytes[12...15] = ArraySlice(parts)
        return RDMAGlobalID(bytes: bytes)
    }

    public var description: String {
        if let v4 = ipv4Address { return "::ffff:\(v4)" }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
}

/// Everything `ibv_modify_qp` needs for the Reset→Init→RTR step, which is the
/// only transition that depends on the peer.
public struct RDMAReadyToReceive: Sendable, Equatable {
    public var pathMTU: UInt32
    public var receivePSN: UInt32
    public var destinationQueuePairNumber: UInt32
    public var destinationGID: RDMAGlobalID
    public var destinationLocalID: UInt16
    public var sourceGIDIndex: UInt8
    public var hopLimit: UInt8
    public var port: UInt8

    public init(pathMTU: UInt32, receivePSN: UInt32, destinationQueuePairNumber: UInt32,
                destinationGID: RDMAGlobalID, destinationLocalID: UInt16,
                sourceGIDIndex: UInt8, hopLimit: UInt8, port: UInt8) {
        self.pathMTU = pathMTU
        self.receivePSN = receivePSN
        self.destinationQueuePairNumber = destinationQueuePairNumber
        self.destinationGID = destinationGID
        self.destinationLocalID = destinationLocalID
        self.sourceGIDIndex = sourceGIDIndex
        self.hopLimit = hopLimit
        self.port = port
    }
}

public struct RDMACompletion: Sendable, Equatable {
    public var workRequestID: UInt64
    public var status: UInt32
    public var opcode: UInt32
    public var byteCount: UInt32

    public init(workRequestID: UInt64, status: UInt32, opcode: UInt32, byteCount: UInt32) {
        self.workRequestID = workRequestID
        self.status = status
        self.opcode = opcode
        self.byteCount = byteCount
    }

    public var isSuccess: Bool { status == UInt32(MCCL_IBV_WC_SUCCESS) }
}

// MARK: - The spec's constants, named once

/// The parts of TN3205 that are numbers.
///
/// Every one of these is cross-checked against Apple's own header by a
/// `_Static_assert` in `Sources/CRDMA/shim.c`, on any machine whose SDK has one.
public enum RDMASpec {
    /// `IBV_QPT_UC`. TN3205 supports unreliable-connection queue pairs only.
    public static let queuePairType = Int32(MCCL_IBV_QPT_UC)
    /// `IBV_MTU_4096` — Thunderbolt moves 4 KiB frames.
    public static let pathMTU = UInt32(MCCL_IBV_MTU_4096)
    /// The only access flag TN3205 permits: the hardware does no remote writes.
    public static let accessLocalWrite = Int32(MCCL_IBV_ACCESS_LOCAL_WRITE)
    /// `IBV_WR_SEND` — the only supported opcode.
    public static let opcodeSend = Int32(MCCL_IBV_WR_SEND)
    public static let sendSignaled = UInt32(MCCL_IBV_SEND_SIGNALED)

    /// Apple Thunderbolt controllers have a single port, always numbered 1.
    public static let portNumber: UInt8 = 1
    /// GID index 1 is the IPv4-mapped address of the paired IP interface.
    public static let gidIndex: UInt8 = 1
    /// Point-to-point: a Thunderbolt peer is exactly one hop away.
    public static let hopLimit: UInt8 = 1
    /// TN3205 shows `pkey_index = 0` in the Init transition.
    public static let pkeyIndex: UInt16 = 0

    /// The wire frame Thunderbolt queues are measured in.
    public static let frameBytes = 4096
    /// "Maximum of 4095 work requests at a time."
    public static let maxWorkRequests = 4095
    /// "Message sizes of up to 16,773,120 bytes" — 4095 frames of 4 KiB.
    public static let maxMessageBytes = 4095 * 4096
    /// "Maximum of 10 unreliable connection (UC) queue pairs."
    public static let maxQueuePairs = 10

    /// The three `ibv_modify_qp` attribute masks, exactly as TN3205 lists them.
    /// Named here so a unit test can assert them without a device.
    public enum Mask {
        /// `IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS`
        public static let initialize =
            Int32(MCCL_IBV_QP_STATE) | Int32(MCCL_IBV_QP_PKEY_INDEX)
            | Int32(MCCL_IBV_QP_PORT) | Int32(MCCL_IBV_QP_ACCESS_FLAGS)

        /// `IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN`
        public static let readyToReceive =
            Int32(MCCL_IBV_QP_STATE) | Int32(MCCL_IBV_QP_AV)
            | Int32(MCCL_IBV_QP_PATH_MTU) | Int32(MCCL_IBV_QP_DEST_QPN)
            | Int32(MCCL_IBV_QP_RQ_PSN)

        /// `IBV_QP_STATE | IBV_QP_SQ_PSN`
        public static let readyToSend =
            Int32(MCCL_IBV_QP_STATE) | Int32(MCCL_IBV_QP_SQ_PSN)
    }
}

// MARK: - The real implementation

/// `RDMAVerbs` over `CRDMA`, i.e. over Apple's librdma.
public final class SystemVerbs: RDMAVerbs, @unchecked Sendable {
    public static let shared = SystemVerbs()
    public init() {}

    public func load() -> String? {
        mccl_rdma_load() == Int32(MCCL_RDMA_OK) ? nil : (mccl_rdma_load_error().map(String.init(cString:)))
    }

    private func check(_ code: Int32, _ what: String) throws {
        guard code != Int32(MCCL_RDMA_OK) else { return }
        throw MCCLError.rdmaFailure(what, code: code)
    }

    public func deviceCount() throws -> Int {
        let count = mccl_rdma_device_count()
        guard count >= 0 else { throw MCCLError.rdmaFailure("ibv_get_device_list", code: count) }
        return Int(count)
    }

    public func deviceName(at index: Int) throws -> String {
        var buffer = [CChar](repeating: 0, count: 128)
        try check(mccl_rdma_device_name(Int32(index), &buffer, buffer.count),
                  "ibv_get_device_name(\(index))")
        return String(cString: buffer)
    }

    public func openDevice(at index: Int) throws -> RDMAContextHandle {
        guard let pointer = mccl_rdma_open_device(Int32(index)) else {
            throw MCCLError.rdmaFailure("ibv_open_device(\(index))", code: Int32(MCCL_RDMA_FAILED))
        }
        return RDMAContextHandle(raw: UInt(bitPattern: pointer))
    }

    public func closeDevice(_ context: RDMAContextHandle) {
        mccl_rdma_close_device(pointer(context.raw))
    }

    public func queryPort(_ context: RDMAContextHandle, port: UInt8) throws -> RDMAPortAttributes {
        var lid: UInt16 = 0
        var state: UInt32 = 0
        try check(mccl_rdma_query_port(pointer(context.raw), port, &lid, &state),
                  "ibv_query_port(port \(port))")
        return RDMAPortAttributes(localID: lid, state: state)
    }

    public func queryGID(_ context: RDMAContextHandle, port: UInt8, index: Int) throws -> RDMAGlobalID {
        var bytes = [UInt8](repeating: 0, count: 16)
        try check(mccl_rdma_query_gid(pointer(context.raw), port, Int32(index), &bytes),
                  "ibv_query_gid(port \(port), index \(index))")
        return RDMAGlobalID(bytes: bytes)
    }

    public func allocateProtectionDomain(_ context: RDMAContextHandle) throws -> RDMAProtectionDomainHandle {
        guard let pointer = mccl_rdma_alloc_pd(pointer(context.raw)) else {
            throw MCCLError.rdmaFailure("ibv_alloc_pd", code: Int32(MCCL_RDMA_FAILED))
        }
        return RDMAProtectionDomainHandle(raw: UInt(bitPattern: pointer))
    }

    public func deallocateProtectionDomain(_ pd: RDMAProtectionDomainHandle) {
        mccl_rdma_dealloc_pd(pointer(pd.raw))
    }

    public func registerMemoryRegion(
        _ pd: RDMAProtectionDomainHandle,
        address: UnsafeMutableRawPointer,
        length: Int,
        access: Int32
    ) throws -> RDMAMemoryRegionHandle {
        guard let pointer = mccl_rdma_reg_mr(pointer(pd.raw), address, length, access) else {
            throw MCCLError.rdmaFailure("ibv_reg_mr(\(length) bytes)", code: Int32(MCCL_RDMA_FAILED))
        }
        return RDMAMemoryRegionHandle(raw: UInt(bitPattern: pointer))
    }

    public func localKey(_ mr: RDMAMemoryRegionHandle) -> UInt32 {
        mccl_rdma_mr_lkey(pointer(mr.raw))
    }

    public func deregisterMemoryRegion(_ mr: RDMAMemoryRegionHandle) {
        mccl_rdma_dereg_mr(pointer(mr.raw))
    }

    public func createCompletionQueue(_ context: RDMAContextHandle, depth: Int) throws -> RDMACompletionQueueHandle {
        guard let pointer = mccl_rdma_create_cq(pointer(context.raw), Int32(depth)) else {
            throw MCCLError.rdmaFailure("ibv_create_cq(depth \(depth))", code: Int32(MCCL_RDMA_FAILED))
        }
        return RDMACompletionQueueHandle(raw: UInt(bitPattern: pointer))
    }

    public func destroyCompletionQueue(_ cq: RDMACompletionQueueHandle) {
        mccl_rdma_destroy_cq(pointer(cq.raw))
    }

    public func createQueuePair(
        _ pd: RDMAProtectionDomainHandle,
        completionQueue: RDMACompletionQueueHandle,
        maxSendWorkRequests: Int,
        maxReceiveWorkRequests: Int,
        type: Int32
    ) throws -> RDMAQueuePairHandle {
        guard let pointer = mccl_rdma_create_qp(
            pointer(pd.raw), pointer(completionQueue.raw),
            Int32(maxSendWorkRequests), Int32(maxReceiveWorkRequests), type
        ) else {
            throw MCCLError.rdmaFailure("ibv_create_qp", code: Int32(MCCL_RDMA_FAILED))
        }
        return RDMAQueuePairHandle(raw: UInt(bitPattern: pointer))
    }

    public func queuePairNumber(_ qp: RDMAQueuePairHandle) -> UInt32 {
        mccl_rdma_qp_num(pointer(qp.raw))
    }

    public func destroyQueuePair(_ qp: RDMAQueuePairHandle) {
        mccl_rdma_destroy_qp(pointer(qp.raw))
    }

    public func modifyToInit(_ qp: RDMAQueuePairHandle, port: UInt8, pkeyIndex: UInt16,
                             accessFlags: UInt32, mask: Int32) throws {
        try check(mccl_rdma_modify_qp_to_init(pointer(qp.raw), port, pkeyIndex, accessFlags, mask),
                  "ibv_modify_qp -> INIT")
    }

    public func modifyToReadyToReceive(_ qp: RDMAQueuePairHandle, attributes: RDMAReadyToReceive,
                                       mask: Int32) throws {
        try attributes.destinationGID.bytes.withUnsafeBufferPointer { gid in
            try check(mccl_rdma_modify_qp_to_rtr(
                pointer(qp.raw), attributes.pathMTU, attributes.receivePSN,
                attributes.destinationQueuePairNumber, gid.baseAddress,
                attributes.destinationLocalID, attributes.sourceGIDIndex,
                attributes.hopLimit, attributes.port, mask), "ibv_modify_qp -> RTR")
        }
    }

    public func modifyToReadyToSend(_ qp: RDMAQueuePairHandle, sendPSN: UInt32, mask: Int32) throws {
        try check(mccl_rdma_modify_qp_to_rts(pointer(qp.raw), sendPSN, mask), "ibv_modify_qp -> RTS")
    }

    public func postSend(_ qp: RDMAQueuePairHandle, workRequestID: UInt64,
                         address: UInt64, length: UInt32, localKey: UInt32,
                         sendFlags: UInt32, opcode: Int32) throws {
        try check(mccl_rdma_post_send(pointer(qp.raw), workRequestID, address, length,
                                      localKey, sendFlags, opcode), "ibv_post_send")
    }

    public func postReceive(_ qp: RDMAQueuePairHandle, workRequestID: UInt64,
                            address: UInt64, length: UInt32, localKey: UInt32) throws {
        try check(mccl_rdma_post_recv(pointer(qp.raw), workRequestID, address, length, localKey),
                  "ibv_post_recv")
    }

    public func pollCompletionQueue(_ cq: RDMACompletionQueueHandle, maxEntries: Int) throws -> [RDMACompletion] {
        var raw = [mccl_rdma_completion](repeating: mccl_rdma_completion(), count: max(1, maxEntries))
        let polled = raw.withUnsafeMutableBufferPointer {
            mccl_rdma_poll_cq(pointer(cq.raw), Int32(maxEntries), $0.baseAddress)
        }
        guard polled >= 0 else { throw MCCLError.rdmaFailure("ibv_poll_cq", code: polled) }
        return (0..<Int(polled)).map {
            RDMACompletion(workRequestID: raw[$0].wr_id, status: raw[$0].status,
                           opcode: raw[$0].opcode, byteCount: raw[$0].byte_len)
        }
    }

    public func statusDescription(_ status: UInt32) -> String {
        String(cString: mccl_rdma_status_string(status))
    }

    private func pointer(_ raw: UInt) -> UnsafeMutableRawPointer? {
        UnsafeMutableRawPointer(bitPattern: raw)
    }
}
