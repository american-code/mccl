import Foundation

/// Rank-addressed connectivity. The collectives only ever ask "give me the
/// channel to rank k" — which transport answers is not their business.
public protocol Fabric: AnyObject {
    var rank: Int { get }
    var worldSize: Int { get }
    func channel(to peer: Int) throws -> Channel
    func shutdown()
}

/// A full mesh of point-to-point channels, one per ordered pair of ranks.
///
/// A mesh (rather than just ring neighbours) is what makes the planner useful:
/// tree and hierarchical plans need edges that are not ring-adjacent, and the
/// plan is chosen per-operation, after the fabric already exists.
public final class MeshFabric: Fabric, @unchecked Sendable {
    public let rank: Int
    public let worldSize: Int

    private let lock = NSLock()
    private var channels: [Int: Channel]
    private var listener: Listener?
    private var isShutDown = false

    init(rank: Int, worldSize: Int, channels: [Int: Channel], listener: Listener?) {
        self.rank = rank
        self.worldSize = worldSize
        self.channels = channels
        self.listener = listener
    }

    public func channel(to peer: Int) throws -> Channel {
        guard peer >= 0, peer < worldSize else {
            throw MCCLError.rankOutOfRange(rank: peer, worldSize: worldSize)
        }
        guard peer != rank else {
            throw MCCLError.invalidArgument("no channel from rank \(rank) to itself")
        }
        lock.lock(); defer { lock.unlock() }
        guard let channel = channels[peer] else {
            throw MCCLError.invalidArgument("rank \(rank) has no channel to \(peer)")
        }
        return channel
    }

    /// Peer rank -> the address this rank's channel to it actually runs over.
    ///
    /// For a peer this rank dialed, that is the address per-pair selection
    /// chose out of the peer's advertisement. For a peer that dialed us, it is
    /// the address it egressed from. Either way it names the cable, which is
    /// the one thing a mixed-fabric run needs to be able to state rather than
    /// assume: a table of numbers from a world whose paths are unknown says
    /// nothing about the fabric it claims to measure.
    public var paths: [Int: String] {
        lock.lock(); defer { lock.unlock() }
        return channels.mapValues(\.peerDescription)
    }

    public func shutdown() {
        lock.lock()
        guard !isShutDown else { lock.unlock(); return }
        isShutDown = true
        let open = channels
        channels.removeAll()
        let l = listener
        listener = nil
        lock.unlock()
        for (_, c) in open { c.close() }
        l?.close()
    }

    // MARK: - Bootstrap

    /// Tag used for the one-frame rank handshake a dialer sends on connect.
    static let handshakeTag: UInt32 = 0xB007

    /// How long one dial attempt gets before the next candidate is tried.
    ///
    /// Only relevant when a peer advertised more than one address. It has to be
    /// short enough that three dead `169.254/16` addresses do not exhaust the
    /// bring-up budget before the live one is reached, and long enough that a
    /// working link is never abandoned mid-handshake. Each pass doubles it, so
    /// a peer that is merely slow to bind is still found.
    public static var dialAttemptSeconds: TimeInterval = 3

    /// Opens a channel to one peer, choosing the path per pair.
    ///
    /// The order comes from `PathSelection.candidates`; the *decision* comes
    /// from dialing them. On a link-local Thunderbolt fabric a subnet match
    /// cannot distinguish three TB ports on one machine — they all carry
    /// `169.254/16` — so the only honest test of which cable reaches this peer
    /// is to open the socket and see.
    public static func dial(
        to advertisement: PeerAdvertisement,
        own: PeerAdvertisement? = nil,
        transport: Transport,
        timeout: TimeInterval
    ) throws -> Channel {
        let candidates = PathSelection.candidates(for: advertisement, own: own)
        guard let first = candidates.first else {
            throw MCCLError.invalidArgument(
                "rank \(advertisement.rank) advertised no address this rank can dial")
        }
        // One candidate is the old single-address bootstrap exactly: hand it
        // the whole budget and let the transport do its own retrying.
        guard candidates.count > 1 else {
            return try transport.connect(to: first.endpoint.address, timeout: timeout)
        }

        let deadline = Date().addingTimeInterval(timeout)
        var failures: [String] = []
        var pass = 0
        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { break }
            let attempt = min(dialAttemptSeconds * pow(2, Double(pass)), remaining)
            failures.removeAll(keepingCapacity: true)
            for candidate in candidates {
                let slice = min(attempt, max(0, deadline.timeIntervalSinceNow))
                guard slice > 0 else { break }
                do {
                    return try transport.connect(to: candidate.endpoint.address, timeout: slice)
                } catch {
                    failures.append("\(candidate) — \(error)")
                }
            }
            pass += 1
        }
        throw MCCLError.timedOut(
            "no path to rank \(advertisement.rank) in \(Int(timeout))s; tried "
            + failures.joined(separator: "; "))
    }

    /// Connects rank `rank` to every other rank.
    ///
    /// Ordering rule: for each pair (i, j) with i < j, the higher rank dials and
    /// the lower rank accepts. Every listener must already be bound before any
    /// rank calls this (see `makeGroup`), which is what keeps bring-up
    /// deadlock-free — a dial can land in the listen backlog before the peer
    /// reaches its accept loop.
    ///
    /// Each dial picks its own path out of the target's advertisement, so a
    /// world spanning a Thunderbolt island and a Wi-Fi bridge forms one mesh
    /// with each pair on the best cable the two of them share.
    public static func bootstrap(
        rank: Int,
        worldSize: Int,
        advertisements: [PeerAdvertisement],
        listener: Listener,
        transport: Transport,
        timeout: TimeInterval = 30
    ) throws -> MeshFabric {
        guard rank >= 0, rank < worldSize else {
            throw MCCLError.rankOutOfRange(rank: rank, worldSize: worldSize)
        }
        guard advertisements.count == worldSize else {
            throw MCCLError.invalidArgument(
                "expected \(worldSize) peer advertisements, got \(advertisements.count)")
        }
        for (index, advertisement) in advertisements.enumerated() where advertisement.rank != index {
            throw MCCLError.protocolViolation(
                "advertisement at slot \(index) claims rank \(advertisement.rank)")
        }
        let own = advertisements[rank]

        var channels: [Int: Channel] = [:]
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount)

        // Dial everyone below us.
        for peer in 0..<rank {
            let channel = try dial(to: advertisements[peer], own: own,
                                   transport: transport, timeout: timeout)
            var header = WireHeader()
            header.tag = handshakeTag
            header.elementCount = UInt32(rank)
            try channel.sendFrame(header, payload: nil, payloadBytes: 0, using: scratch)
            channels[peer] = channel
        }

        // Accept everyone above us.
        let inbound = worldSize - 1 - rank
        for _ in 0..<inbound {
            let channel = try listener.accept(timeout: timeout)
            let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 0)
            guard header.tag == handshakeTag else {
                channel.close()
                throw MCCLError.protocolViolation("expected bootstrap handshake, got tag \(header.tag)")
            }
            let peer = Int(header.elementCount)
            guard peer > rank, peer < worldSize, channels[peer] == nil else {
                channel.close()
                throw MCCLError.protocolViolation("bootstrap handshake announced impossible rank \(peer)")
            }
            if advertisements[peer].disavows(source: channel.remoteAddress) {
                let observed = channel.remoteAddress?.host ?? "?"
                channel.close()
                throw MCCLError.protocolViolation(
                    "rank \(peer) announced itself from \(observed), which is not among the "
                    + "addresses it advertised (\(advertisements[peer].hosts.joined(separator: ", ")))")
            }
            channels[peer] = channel
        }

        return MeshFabric(rank: rank, worldSize: worldSize, channels: channels, listener: listener)
    }

    /// Single-address bootstrap, unchanged for callers that know one address per
    /// rank. Every address becomes a one-endpoint advertisement.
    public static func bootstrap(
        rank: Int,
        worldSize: Int,
        addresses: [PeerAddress],
        listener: Listener,
        transport: Transport,
        timeout: TimeInterval = 30
    ) throws -> MeshFabric {
        try bootstrap(
            rank: rank, worldSize: worldSize,
            advertisements: addresses.enumerated().map {
                PeerAdvertisement(rank: $0.offset, address: $0.element)
            },
            listener: listener, transport: transport, timeout: timeout)
    }


    /// Brings up a whole world in this process: binds `worldSize` listeners on
    /// ephemeral ports, then bootstraps every rank concurrently.
    ///
    /// This is how the test suite runs N ranks over real loopback TCP — each
    /// rank gets its own port and its own sockets, so the collectives exercise
    /// the same code path they would across machines.
    public static func makeGroup(
        worldSize: Int,
        transport: Transport = TCPTransport(),
        host: String = "127.0.0.1",
        timeout: TimeInterval = 30
    ) throws -> [MeshFabric] {
        guard worldSize > 0 else { throw MCCLError.invalidArgument("worldSize must be > 0") }

        var listeners: [Listener] = []
        listeners.reserveCapacity(worldSize)
        do {
            for _ in 0..<worldSize {
                listeners.append(try transport.listen(host: host, port: 0))
            }
        } catch {
            listeners.forEach { $0.close() }
            throw error
        }
        let addresses = listeners.map(\.address)

        let results = ResultSlots<MeshFabric>(count: worldSize)
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "mccl.bootstrap", attributes: .concurrent)
        for rank in 0..<worldSize {
            let listener = listeners[rank]
            queue.async(group: group) {
                do {
                    let fabric = try bootstrap(
                        rank: rank, worldSize: worldSize, addresses: addresses,
                        listener: listener, transport: transport, timeout: timeout)
                    results.set(rank, .success(fabric))
                } catch {
                    results.set(rank, .failure(error))
                }
            }
        }
        group.wait()
        return try results.collect(onFailure: { listeners.forEach { $0.close() } })
    }
}

/// Small helper for joining fan-out work that must produce one value per rank.
final class ResultSlots<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var slots: [Result<T, Error>?]

    init(count: Int) { slots = Array(repeating: nil, count: count) }

    func set(_ index: Int, _ value: Result<T, Error>) {
        lock.lock(); slots[index] = value; lock.unlock()
    }

    func collect(onFailure: () -> Void = {}) throws -> [T] {
        lock.lock(); defer { lock.unlock() }
        var out: [T] = []
        out.reserveCapacity(slots.count)
        for (i, slot) in slots.enumerated() {
            guard let slot else {
                onFailure()
                throw MCCLError.protocolViolation("bootstrap produced no result for rank \(i)")
            }
            switch slot {
            case .success(let v): out.append(v)
            case .failure(let e): onFailure(); throw e
            }
        }
        return out
    }
}
