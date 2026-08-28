import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// RDMA over Thunderbolt, as a `Transport`.
///
/// **This transport has never been run on hardware.** It is written to Apple's
/// Technical Note TN3205 and to the verbs header in the macOS 26.2+ SDK, it
/// compiles everywhere, and its state machine and framing are unit-tested
/// against a mock. Nothing here has moved a byte across a Thunderbolt 5 cable,
/// because the machines this was developed on are M1-generation and have TB4.
/// See `docs/RDMA.md` for what a first TB5-equipped user should check.
///
/// ---------------------------------------------------------------------------
/// Shape
/// ---------------------------------------------------------------------------
///
/// A queue pair is not a socket: it has no listener, no accept, and no way to
/// learn about a peer that has not already been described to it. TN3205 says as
/// much and names the fix —
///
/// > Applications should use an out-of-band communication mechanism to share
/// > GID, LID, QPN, and PSN with a peer. Typically applications use Thunderbolt
/// > IP and a TCP socket connection for this purpose.
///
/// — so `RDMATransport` wraps a `TCPTransport` and uses it for exactly that. The
/// TCP channel does the listening, the accepting and the metadata exchange; once
/// both queue pairs reach RTS the TCP channel stays open but idle, carrying only
/// the close notification. Every byte of collective traffic goes over the queue
/// pair.
///
/// That reuse is deliberate: `MeshFabric`'s bootstrap, `PeerAdvertisement`'s
/// per-pair addressing and the rendezvous all keep working unchanged, because
/// from their side this is just another `Transport`.
///
/// ---------------------------------------------------------------------------
/// Device pairing
/// ---------------------------------------------------------------------------
///
/// TN3205: device names like `rdma_en2` correspond to the Thunderbolt IP
/// interface `en2`, and GID index 1 on `rdma_en2` is `en2`'s IPv4 address. So
/// the device to use for a given peer is decided by the same subnet match that
/// `PathSelection` already performs: find the local interface that reaches the
/// peer's address, then open `rdma_<that interface>`. No guessing, and no
/// configuration for the common case.
public final class RDMATransport: Transport, @unchecked Sendable {
    public let name = "rdma"

    /// The out-of-band channel. TCP over IP-over-Thunderbolt, per TN3205.
    public let bootstrap: Transport

    public let geometry: RDMASlotGeometry
    let verbs: RDMAVerbs

    /// How long the metadata exchange may take once the socket is up.
    public var handshakeTimeout: TimeInterval = 30

    public init(bootstrap: Transport = TCPTransport(),
                geometry: RDMASlotGeometry? = nil,
                verbs: RDMAVerbs = SystemVerbs.shared) throws {
        self.bootstrap = bootstrap
        self.geometry = try geometry ?? RDMASlotGeometry()
        self.verbs = verbs
    }

    // MARK: - Availability

    /// Why RDMA cannot be used here, or nil when it can.
    ///
    /// Follows the `tm_unusable_reason` shape from triton-metal: a single
    /// nullable string that says precisely which of the several quite different
    /// things went wrong, because "RDMA unavailable" sends a user looking in the
    /// wrong place three times out of four.
    public enum Unavailability: Equatable, CustomStringConvertible {
        /// Built against an SDK with no `<infiniband/verbs.h>`.
        case notCompiled(String)
        /// `librdma.dylib` would not load — the OS predates macOS 26.2.
        case libraryUnavailable(String)
        /// The library loaded and reported no devices. From user space these two
        /// causes are indistinguishable, so both are named rather than one
        /// guessed at.
        case noDevices

        public var description: String {
            switch self {
            case .notCompiled(let detail), .libraryUnavailable(let detail):
                return detail
            case .noDevices:
                return """
                    librdma is present but reports no RDMA devices. Either this Mac has no \
                    Thunderbolt 5 controller (RDMA over Thunderbolt needs Apple silicon with TB5), \
                    or RDMA is not enabled — enable it by booting into macOS Recovery, opening \
                    Utilities > Terminal, running `rdma_ctl enable`, and rebooting. Check with \
                    `ibv_devices`.
                    """
            }
        }
    }

    /// Nil when RDMA is usable on this machine.
    public static func unavailability(verbs: RDMAVerbs = SystemVerbs.shared) -> Unavailability? {
        if let reason = verbs.load() {
            // The shim distinguishes these two, and they send a reader to very
            // different places: one is a rebuild, the other is an OS upgrade.
            return reason.contains("without <infiniband/verbs.h>")
                ? .notCompiled(reason)
                : .libraryUnavailable(reason)
        }
        let count = (try? verbs.deviceCount()) ?? 0
        return count > 0 ? nil : .noDevices
    }

    public static func isAvailable(verbs: RDMAVerbs = SystemVerbs.shared) -> Bool {
        unavailability(verbs: verbs) == nil
    }

    /// The reason RDMA is unusable, or nil. The string a diagnostic prints.
    public static func unusableReason(verbs: RDMAVerbs = SystemVerbs.shared) -> String? {
        unavailability(verbs: verbs)?.description
    }

    public var isAvailable: Bool { RDMATransport.isAvailable(verbs: verbs) }
    public var unusableReason: String? { RDMATransport.unusableReason(verbs: verbs) }

    /// Every RDMA device this machine has, in the library's order.
    public static func deviceNames(verbs: RDMAVerbs = SystemVerbs.shared) -> [String] {
        guard verbs.load() == nil, let count = try? verbs.deviceCount() else { return [] }
        return (0..<count).compactMap { try? verbs.deviceName(at: $0) }
    }

    // MARK: - Device selection

    /// The device index whose paired IP interface reaches `address`.
    ///
    /// `rdma_en2` pairs with `en2`; the local interface that reaches a peer is
    /// the one whose subnet contains its address. When nothing matches — a peer
    /// several hops away, or an address on an interface with no RDMA device —
    /// this fails rather than picking a device at random, because on a
    /// multi-port machine the wrong device is a cable that does not go where the
    /// caller thinks it does.
    func device(reaching address: PeerAddress) throws -> (index: Int, name: String) {
        let names = RDMATransport.deviceNames(verbs: verbs)
        guard !names.isEmpty else {
            throw MCCLError.rdmaUnavailable(
                RDMATransport.unusableReason(verbs: verbs) ?? "no RDMA devices")
        }
        guard let interface = NetworkInterfaces.interface(reaching: address) else {
            throw MCCLError.rdmaUnavailable(
                "no local interface reaches \(address.host); RDMA over Thunderbolt is "
                + "point-to-point and cannot route (available devices: \(names.joined(separator: ", ")))")
        }
        let wanted = RDMATransport.deviceName(pairedWith: interface.name)
        guard let index = names.firstIndex(of: wanted) else {
            throw MCCLError.rdmaUnavailable(
                "\(address.host) is reached over \(interface.name), which has no paired RDMA "
                + "device \(wanted) (available: \(names.joined(separator: ", ")))")
        }
        return (index, wanted)
    }

    /// `en2` -> `rdma_en2`, TN3205's naming.
    public static func deviceName(pairedWith interface: String) -> String { "rdma_\(interface)" }

    /// `rdma_en2` -> `en2`, the inverse.
    public static func interfaceName(pairedWith device: String) -> String? {
        device.hasPrefix("rdma_") ? String(device.dropFirst("rdma_".count)) : nil
    }

    // MARK: - Transport

    public func listen(host: String, port: Int) throws -> Listener {
        RDMAListener(inner: try bootstrap.listen(host: host, port: port), transport: self)
    }

    public func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel {
        let selected = try device(reaching: address)
        let side = try bootstrap.connect(to: address, timeout: timeout)
        do {
            return try RDMATransport.handshake(
                over: side, transport: self,
                deviceIndex: selected.index, deviceName: selected.name)
        } catch {
            side.close()
            throw error
        }
    }

    // MARK: - Handshake

    /// Builds a queue pair, swaps metadata over `side`, and drives Init->RTR->RTS.
    ///
    /// Symmetric: both ends write their own metadata and then read the peer's.
    /// Writing first is what avoids a deadlock — 40 bytes fit in any socket
    /// buffer, so neither side blocks before the other has spoken.
    static func handshake(over side: Channel, transport: RDMATransport,
                          deviceIndex: Int, deviceName: String) throws -> Channel {
        let connection = try RDMAConnection(
            verbs: transport.verbs,
            deviceIndex: deviceIndex,
            deviceName: deviceName,
            geometry: transport.geometry,
            packetSequenceNumber: RDMATransport.randomPacketSequenceNumber())

        do {
            guard connection.isPortActive else {
                throw MCCLError.rdmaUnavailable(
                    "\(deviceName) port \(RDMASpec.portNumber) is not active — no Thunderbolt "
                    + "cable, or the peer is not powered up")
            }

            let mine = connection.metadata.encoded()
            try mine.withUnsafeBytes { try side.sendBytes($0) }

            var theirs = [UInt8](repeating: 0, count: RDMAQueuePairMetadata.byteCount)
            try theirs.withUnsafeMutableBytes { try side.receiveBytes(into: $0) }
            let peer = try RDMAQueuePairMetadata.decode(theirs)

            try connection.connect(to: peer)
            return RDMAChannel(connection: connection, side: side, deviceName: deviceName)
        } catch {
            connection.close()
            throw error
        }
    }

    /// TN3205's example picks a fixed number ("pick a random number"); a real
    /// one is drawn per connection so a stale queue pair's packets cannot be
    /// mistaken for a new one's. 24 bits, the PSN field's width.
    static func randomPacketSequenceNumber() -> UInt32 {
        UInt32.random(in: 0..<(1 << 24))
    }
}

// MARK: - Listener

/// Accepts a TCP bootstrap connection, then completes the RDMA handshake on it.
final class RDMAListener: Listener {
    private let inner: Listener
    private unowned let transport: RDMATransport

    init(inner: Listener, transport: RDMATransport) {
        self.inner = inner
        self.transport = transport
    }

    var address: PeerAddress { inner.address }

    func accept(timeout: TimeInterval?) throws -> Channel {
        let side = try inner.accept(timeout: timeout)
        do {
            // The accepting side picks its device from where the dialer came
            // from, which is the same subnet match the dialer used in reverse.
            guard let remote = side.remoteAddress else {
                throw MCCLError.rdmaUnavailable(
                    "the bootstrap transport reports no peer address, so no RDMA device can be "
                    + "paired with this connection")
            }
            let selected = try transport.device(reaching: remote)
            return try RDMATransport.handshake(
                over: side, transport: transport,
                deviceIndex: selected.index, deviceName: selected.name)
        } catch {
            side.close()
            throw error
        }
    }

    func close() { inner.close() }
}

// MARK: - Channel

/// A `Channel` whose bytes travel over a queue pair.
///
/// Blocking, like every other `Channel`: the collectives run each one on its own
/// pair of dispatch queues, so a blocking poll here never occupies a Swift
/// concurrency thread. `sendQueue` and `receiveQueue` are distinct, which is
/// what lets a ring step send forward while the previous rank's chunk arrives —
/// and is why `RDMAConnection` has to arbitrate one completion queue between two
/// threads.
final class RDMAChannel: Channel {
    private let connection: RDMAConnection
    /// Kept open for the lifetime of the channel. It carries no collective
    /// traffic; it exists so that a peer going away is observable, and so the
    /// address check in `MeshFabric.bootstrap` still has a source address.
    private let side: Channel
    private let closeLock = NSLock()
    private var closed = false

    let sendQueue: DispatchQueue
    let receiveQueue: DispatchQueue
    let peerDescription: String
    var remoteAddress: PeerAddress? { side.remoteAddress }

    init(connection: RDMAConnection, side: Channel, deviceName: String) {
        self.connection = connection
        self.side = side
        self.peerDescription = "\(side.peerDescription) via \(deviceName)"
        let id = ObjectIdentifier(connection).hashValue & 0xFFFF
        self.sendQueue = DispatchQueue(label: "mccl.rdma.tx.\(id)")
        self.receiveQueue = DispatchQueue(label: "mccl.rdma.rx.\(id)")
    }

    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws {
        try connection.send(buffer)
    }

    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws {
        try connection.receive(into: buffer)
    }

    func close() {
        closeLock.lock(); defer { closeLock.unlock() }
        guard !closed else { return }
        closed = true
        connection.sendClose()
        connection.close()
        side.close()
    }

    deinit { close() }
}
