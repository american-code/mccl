import Foundation

/// The token that lets `n` ranks find each other, mirroring `ncclUniqueId`.
///
/// It carries exactly two things: where rank 0 is listening, and a nonce that
/// distinguishes concurrent jobs on the same host. Everything else — each
/// rank's own data addresses — is discovered through rank 0, which is why the
/// token stays small enough to broadcast over MPI, a file, or an env var.
///
/// "Where rank 0 is listening" is a *list* of hosts, not one host. Rank 0 binds
/// one listener on one port, but on a mixed fabric the peers that must reach it
/// do not all share a subnet: a Thunderbolt neighbour dials `169.254.x`, a
/// laptop on the LAN dials `192.168.x`, and no single entry serves both.
///
/// ## Text format, and its version marker
///
/// * `mccl1:<nonce hex>:<host>:<port>` — one host. Emitted verbatim whenever
///   there is exactly one, so a single-homed cluster's tokens are byte-identical
///   to the ones mccl produced before per-pair addressing existed.
/// * `mccl2:<nonce hex>:<port>:<host>|<host>|…` — two or more. The port moves
///   ahead of the hosts so an IPv6 literal's colons stay unambiguous, and `|`
///   separates hosts because it appears in no IP literal.
///
/// Both parse. A build older than `mccl2` cannot read an `mccl2` token, and
/// that is the honest consequence of the format growing: the marker is bumped
/// rather than the new hosts being smuggled somewhere an old parser would skip.
public struct UniqueID: Sendable, Hashable, CustomStringConvertible {
    /// Distinguishes two jobs bootstrapping through the same host and port.
    public let nonce: UInt64
    /// Every host rank 0's rendezvous listener answers on, in rank 0's own
    /// preference order. Never empty.
    public let hosts: [String]
    /// The port that one listener is bound to.
    public let port: Int

    public init(nonce: UInt64, address: PeerAddress) {
        self.init(nonce: nonce, hosts: [address.host], port: address.port)
    }

    /// - Note: An empty `hosts` degrades to the loopback rather than producing
    ///   a token nothing can dial.
    public init(nonce: UInt64, hosts: [String], port: Int) {
        self.nonce = nonce
        var seen: Set<String> = []
        let unique = hosts.filter { !$0.isEmpty && seen.insert($0).inserted }
        self.hosts = unique.isEmpty ? ["127.0.0.1"] : unique
        self.port = port
    }

    /// Rank 0's primary address — the first host. Kept for every caller that
    /// wants one address to name the rendezvous by.
    public var address: PeerAddress { PeerAddress(host: hosts[0], port: port) }

    public var addresses: [PeerAddress] { hosts.map { PeerAddress(host: $0, port: port) } }

    /// Rank 0's rendezvous listener as something `MeshFabric.dial` can walk.
    /// The media tags are unknown — a token carries hosts, not hardware — so
    /// the dialer's own egress classification does all the ranking.
    public var advertisement: PeerAdvertisement {
        PeerAdvertisement(rank: 0, endpoints: addresses.map {
            PeerEndpoint(address: $0, media: $0.isLoopback ? .loopback : .other)
        })
    }

    /// Text, so the token survives being pasted into a launcher, an env var, or
    /// a job script.
    public var text: String {
        guard hosts.count > 1 else {
            return String(format: "mccl1:%016llx:%@:%d", nonce, hosts[0], port)
        }
        return String(format: "mccl2:%016llx:%d:%@", nonce, port, hosts.joined(separator: "|"))
    }

    public var description: String { text }

    public init?(text: String) {
        let parts = text.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 4, let nonce = UInt64(parts[1], radix: 16) else { return nil }
        switch parts[0] {
        case "mccl1":
            guard let port = Int(parts[parts.count - 1]) else { return nil }
            let host = parts[2..<(parts.count - 1)].joined(separator: ":")
            guard !host.isEmpty else { return nil }
            self.init(nonce: nonce, hosts: [host], port: port)
        case "mccl2":
            guard let port = Int(parts[2]) else { return nil }
            let joined = parts[3...].joined(separator: ":")
            let hosts = joined.split(separator: "|", omittingEmptySubsequences: true).map(String.init)
            guard !hosts.isEmpty, hosts.allSatisfy({ !$0.isEmpty }) else { return nil }
            self.init(nonce: nonce, hosts: hosts, port: port)
        default:
            return nil
        }
    }
}

/// Turns a `UniqueID` into the full peer-address table every rank needs.
///
/// `Communicator.bootstrap` deliberately takes explicit addresses: it has no
/// opinion about how a cluster is launched. This is the smallest thing that
/// closes the gap for the C ABI, where `mcclCommInitRank` is handed one token
/// and nothing else.
///
/// The protocol is one round trip through rank 0:
///
/// 1. Every rank binds its own data listener first, so no dial can arrive
///    before its target is accepting.
/// 2. Ranks 1…n-1 connect to the token's address and announce
///    `(rank, data address)`.
/// 3. Rank 0 waits for all n-1 announcements, then sends the assembled table
///    back down the same connections.
///
/// What this is not: a rendezvous *service*. There is no discovery, no
/// membership change, no retry across a rank restarting. Rank 0 must be up, and
/// `createUniqueID` must be called in the process that will host rank 0.
public enum Rendezvous {

    /// Rank-0 listeners created by `createUniqueID`, so that the rank-0 process
    /// reuses the socket it already bound instead of racing to rebind the port.
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pending: [UInt64: Listener] = [:]

    /// Binds a rendezvous listener and returns the token describing it.
    /// Call once, on the machine that will run rank 0.
    ///
    /// `advertisedHosts` names every address the token should carry. Left nil
    /// on a wildcard bind, rank 0 advertises every usable local address, so a
    /// peer on any of its cables can find it — which is the whole point of the
    /// multi-host token.
    public static func createUniqueID(
        transport: Transport = TCPTransport(),
        host: String = "0.0.0.0",
        port: Int = 0,
        advertisedHosts: [String]? = nil
    ) throws -> UniqueID {
        let listener = try transport.listen(host: host, port: port)
        var nonce = UInt64.random(in: 1...UInt64.max)
        lock.lock()
        while pending[nonce] != nil { nonce = UInt64.random(in: 1...UInt64.max) }
        pending[nonce] = listener
        lock.unlock()

        let bound = listener.address.port
        if let advertisedHosts, !advertisedHosts.isEmpty {
            return UniqueID(nonce: nonce, hosts: advertisedHosts, port: bound)
        }
        let isWildcard = host == "0.0.0.0" || host == "::" || host.isEmpty
        guard isWildcard else { return UniqueID(nonce: nonce, hosts: [host], port: bound) }
        let discovered = PeerAdvertisement.local(rank: 0, port: bound).hosts
        return UniqueID(nonce: nonce,
                        hosts: discovered.isEmpty ? ["127.0.0.1"] : discovered,
                        port: bound)
    }

    /// Single-host spelling, kept for callers that already know the one address
    /// they mean to advertise.
    public static func createUniqueID(
        transport: Transport = TCPTransport(),
        host: String = "0.0.0.0",
        port: Int = 0,
        advertisedHost: String?
    ) throws -> UniqueID {
        try createUniqueID(transport: transport, host: host, port: port,
                           advertisedHosts: advertisedHost.map { [$0] })
    }

    /// Releases a token's listener without running a rendezvous. Safe to call
    /// for a token that was already consumed.
    public static func discard(_ id: UniqueID) {
        lock.lock()
        let listener = pending.removeValue(forKey: id.nonce)
        lock.unlock()
        listener?.close()
    }

    /// Exchanges advertisements and returns the world's peer table, indexed by
    /// rank.
    ///
    /// `dataListener` must already be bound; the addresses this rank advertises
    /// are that listener's port on every host in `advertisedHosts`, or on every
    /// usable local address when a wildcard bind leaves the choice open.
    public static func join(
        id: UniqueID,
        rank: Int,
        worldSize: Int,
        dataListener: Listener,
        transport: Transport = TCPTransport(),
        advertisedHosts: [String]? = nil,
        timeout: TimeInterval = 60
    ) throws -> [PeerAdvertisement] {
        guard worldSize > 0 else { throw MCCLError.invalidArgument("worldSize must be > 0") }
        guard rank >= 0, rank < worldSize else {
            throw MCCLError.rankOutOfRange(rank: rank, worldSize: worldSize)
        }

        let bound = dataListener.address
        let mine = resolveAdvertisement(
            rank: rank, bound: bound, rendezvous: id.address.host, explicit: advertisedHosts)

        guard worldSize > 1 else {
            if rank == 0 { discard(id) }
            return [mine]
        }
        return rank == 0
            ? try serve(id: id, worldSize: worldSize, own: mine, transport: transport, timeout: timeout)
            : try announce(id: id, rank: rank, worldSize: worldSize, own: mine,
                           transport: transport, timeout: timeout)
    }

    /// Single-host spelling of `join`.
    public static func join(
        id: UniqueID,
        rank: Int,
        worldSize: Int,
        dataListener: Listener,
        transport: Transport = TCPTransport(),
        advertisedHost: String?,
        timeout: TimeInterval = 60
    ) throws -> [PeerAdvertisement] {
        try join(id: id, rank: rank, worldSize: worldSize, dataListener: dataListener,
                 transport: transport, advertisedHosts: advertisedHost.map { [$0] },
                 timeout: timeout)
    }

    // MARK: - Rank 0

    private static func serve(
        id: UniqueID, worldSize: Int, own: PeerAdvertisement, transport: Transport, timeout: TimeInterval
    ) throws -> [PeerAdvertisement] {
        lock.lock()
        let existing = pending.removeValue(forKey: id.nonce)
        lock.unlock()

        // A token generated in another process on this host: rebind its port.
        // The usual case is `existing`, straight from `createUniqueID`.
        let listener = try existing ?? transport.listen(host: "0.0.0.0", port: id.address.port)
        defer { listener.close() }

        var table = [PeerAdvertisement?](repeating: nil, count: worldSize)
        table[0] = own
        var channels: [Int: Channel] = [:]
        defer { channels.values.forEach { $0.close() } }

        let scratch = ScratchBuffer(capacity: 1 << 16)
        let deadline = Date().addingTimeInterval(timeout)
        while channels.count < worldSize - 1 {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else {
                throw MCCLError.timedOut(
                    "rendezvous: \(channels.count + 1) of \(worldSize) ranks checked in at \(id.address)")
            }
            let channel: Channel
            do {
                channel = try listener.accept(timeout: remaining)
            } catch MCCLError.timedOut {
                continue
            }
            let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 1 << 16)
            guard header.tag == Tag.hello else {
                channel.close()
                throw MCCLError.protocolViolation("rendezvous: expected hello, got tag \(header.tag)")
            }
            let hello = try JSONDecoder().decode(
                Hello.self, from: Data(bytes: scratch.base, count: Int(header.payloadBytes)))
            guard hello.nonce == id.nonce else {
                channel.close()
                throw MCCLError.protocolViolation(
                    "rendezvous: rank \(hello.rank) belongs to a different job")
            }
            guard hello.rank > 0, hello.rank < worldSize, table[hello.rank] == nil else {
                channel.close()
                throw MCCLError.protocolViolation(
                    "rendezvous: impossible or duplicate rank \(hello.rank) for world size \(worldSize)")
            }
            guard hello.worldSize == worldSize else {
                channel.close()
                throw MCCLError.invalidArgument(
                    "rendezvous: rank \(hello.rank) says world size \(hello.worldSize), rank 0 says \(worldSize)")
            }
            let advertisement = hello.advertisement
            guard !advertisement.endpoints.isEmpty else {
                channel.close()
                throw MCCLError.protocolViolation(
                    "rendezvous: rank \(hello.rank) advertised no address")
            }
            // The one fact the announcer does not choose. A rank checking in
            // over a cable it did not advertise cannot be dialed back on that
            // cable, so the world would half-form; say so here instead.
            if advertisement.disavows(source: channel.remoteAddress) {
                let observed = channel.remoteAddress?.host ?? "?"
                channel.close()
                throw MCCLError.protocolViolation(
                    "rendezvous: rank \(hello.rank) checked in from \(observed), which is not "
                    + "among the addresses it advertised (\(advertisement.hosts.joined(separator: ", ")))")
            }
            table[hello.rank] = advertisement
            channels[hello.rank] = channel
        }

        let resolved = try table.enumerated().map { index, advertisement -> PeerAdvertisement in
            guard let advertisement else {
                throw MCCLError.protocolViolation("rendezvous: no address for rank \(index)")
            }
            return advertisement
        }
        // Stay in the v1 shape while every rank has exactly one address: an
        // older joiner can still read that table, and a single-homed cluster
        // gets exactly the bytes it got before per-pair addressing existed.
        let multiHomed = resolved.contains { $0.endpoints.count > 1 }
        let tag = multiHomed ? Tag.advertisements : Tag.table
        let payload = multiHomed
            ? try JSONEncoder().encode(resolved)
            : try JSONEncoder().encode(resolved.map { $0.endpoints[0].address })
        for (_, channel) in channels {
            try send(channel, tag: tag, payload: payload)
        }
        return resolved
    }

    // MARK: - Ranks 1…n-1

    private static func announce(
        id: UniqueID, rank: Int, worldSize: Int, own: PeerAdvertisement,
        transport: Transport, timeout: TimeInterval
    ) throws -> [PeerAdvertisement] {
        // Rank 0 may still be starting; the dialer already retries to its own
        // deadline, so pass the rendezvous timeout straight through. A token
        // carrying several hosts is walked in the same per-pair order the data
        // fabric uses, so the rendezvous itself crosses the best shared cable.
        let channel = try MeshFabric.dial(
            to: id.advertisement, own: own, transport: transport, timeout: timeout)
        defer { channel.close() }

        let hello = Hello(nonce: id.nonce, rank: rank, worldSize: worldSize, own: own)
        try send(channel, tag: Tag.hello, payload: try JSONEncoder().encode(hello))

        let scratch = ScratchBuffer(capacity: 1 << 16)
        let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 1 << 20)
        let data = Data(bytes: scratch.base, count: Int(header.payloadBytes))
        switch header.tag {
        case Tag.advertisements:
            return try JSONDecoder().decode([PeerAdvertisement].self, from: data)
        case Tag.table:
            return try JSONDecoder().decode([PeerAddress].self, from: data)
                .enumerated().map { PeerAdvertisement(rank: $0.offset, address: $0.element) }
        default:
            throw MCCLError.protocolViolation(
                "rendezvous: expected the address table, got tag \(header.tag)")
        }
    }

    // MARK: - Wire helpers

    private enum Tag {
        static let hello: UInt32 = 0xBEE1
        /// v1 table: one `PeerAddress` per rank.
        static let table: UInt32 = 0xBEE2
        /// v2 table: one `PeerAdvertisement` per rank. Sent only when some rank
        /// really has more than one address, so a single-homed world never sees
        /// a frame an older build could not read.
        static let advertisements: UInt32 = 0xBEE3
    }

    private struct Hello: Codable {
        var nonce: UInt64
        var rank: Int
        /// Rank 0 fills this in from its own argument; a joining rank sends its
        /// own so a mismatched launch is caught rather than hanging.
        var worldSize: Int
        /// This rank's primary address. Still sent on its own so a rank 0 built
        /// before per-pair addressing can read the hello it was always able to.
        var address: PeerAddress
        /// Every address this rank can be reached at. Absent from a hello sent
        /// by an older build, in which case `address` is the whole story.
        var endpoints: [PeerEndpoint]?

        init(nonce: UInt64, rank: Int, worldSize: Int, own: PeerAdvertisement) {
            self.nonce = nonce
            self.rank = rank
            self.worldSize = worldSize
            self.address = own.primaryAddress ?? PeerAddress(host: "127.0.0.1", port: 0)
            self.endpoints = own.endpoints
        }

        var advertisement: PeerAdvertisement {
            if let endpoints, !endpoints.isEmpty {
                return PeerAdvertisement(rank: rank, endpoints: endpoints)
            }
            return PeerAdvertisement(rank: rank, address: address)
        }
    }

    private static func send(_ channel: Channel, tag: UInt32, payload: Data) throws {
        let scratch = ScratchBuffer(capacity: WireHeader.byteCount + payload.count)
        var header = WireHeader()
        header.tag = tag
        header.dataType = DataType.int8.wireCode
        header.elementCount = UInt32(payload.count)
        try payload.withUnsafeBytes { source in
            if !payload.isEmpty {
                (scratch.base + WireHeader.byteCount).copyMemory(
                    from: source.baseAddress!, byteCount: payload.count)
            }
            try channel.sendPreparedFrame(header, payloadBytes: payload.count, in: scratch)
        }
    }

    /// What this rank should advertise as its host.
    ///
    /// A wildcard bind tells peers nothing, so substitute a real address: the
    /// loopback if the whole job is on one machine, otherwise the fattest cable
    /// this node has. This is the single-address answer; `resolveAdvertisement`
    /// is the one the rendezvous actually uses.
    static func resolveAdvertisedHost(bound: String, rendezvous: String) -> String {
        if let override = ProcessInfo.processInfo.environment["MCCL_HOST"], !override.isEmpty {
            return override.split(separator: ",").first.map {
                $0.trimmingCharacters(in: .whitespaces)
            } ?? override
        }
        let isWildcard = bound == "0.0.0.0" || bound == "::" || bound.isEmpty
        guard isWildcard else { return bound }
        if PeerAddress(host: rendezvous, port: 0).isLoopback { return "127.0.0.1" }
        return NetworkInterfaces.preferredLocalAddress() ?? "127.0.0.1"
    }

    /// Everything this rank should advertise, in preference order.
    ///
    /// Precedence, highest first:
    ///
    /// 1. `explicit` — what `--bind a,b,c` passed.
    /// 2. `MCCL_HOST`, which now accepts a comma-separated list. One value
    ///    still means exactly what it always did.
    /// 3. A non-wildcard bind: the listener answers on that address and no
    ///    other, so there is nothing to choose.
    /// 4. A loopback rendezvous: the whole job is on this machine.
    /// 5. Otherwise every usable local address — the case per-pair addressing
    ///    exists for, and now the default rather than a single guess.
    static func resolveAdvertisement(
        rank: Int, bound: PeerAddress, rendezvous: String,
        explicit: [String]?, interfaces: [NetworkInterface]? = nil
    ) -> PeerAdvertisement {
        if let explicit, !explicit.isEmpty {
            return .explicit(rank: rank, hosts: explicit, port: bound.port, interfaces: interfaces)
        }
        if let override = ProcessInfo.processInfo.environment["MCCL_HOST"], !override.isEmpty {
            let hosts = override.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if !hosts.isEmpty {
                return .explicit(rank: rank, hosts: hosts, port: bound.port, interfaces: interfaces)
            }
        }
        let isWildcard = bound.host == "0.0.0.0" || bound.host == "::" || bound.host.isEmpty
        guard isWildcard else {
            return .explicit(rank: rank, hosts: [bound.host], port: bound.port, interfaces: interfaces)
        }
        if PeerAddress(host: rendezvous, port: 0).isLoopback {
            return PeerAdvertisement(rank: rank, address: PeerAddress(host: "127.0.0.1", port: bound.port))
        }
        let discovered = PeerAdvertisement.local(rank: rank, port: bound.port, interfaces: interfaces)
        guard !discovered.endpoints.isEmpty else {
            return PeerAdvertisement(rank: rank, address: PeerAddress(host: "127.0.0.1", port: bound.port))
        }
        return discovered
    }
}

// MARK: - Communicator bring-up from a token

extension Communicator {
    /// Joins the world described by `id`, binding this rank's own listener and
    /// discovering every peer through rank 0.
    ///
    /// This is what the C shim's `mcclCommInitRank` calls. Swift callers who
    /// already know every address should keep using `bootstrap`, which involves
    /// no rank-0 dependency at all.
    public static func join(
        uniqueID id: UniqueID,
        rank: Int,
        worldSize: Int,
        transport: Transport = TCPTransport(),
        rendezvousTransport: Transport? = nil,
        bindHost: String = "0.0.0.0",
        advertisedHosts: [String]? = nil,
        topology: Topology? = nil,
        timeout: TimeInterval = 60
    ) throws -> Communicator {
        guard rank >= 0, rank < worldSize else {
            throw MCCLError.rankOutOfRange(rank: rank, worldSize: worldSize)
        }
        // Bind before the rendezvous: rank 0 hands out the table only once every
        // rank has announced, so by the time anyone dials, every listener is up.
        // A loopback token means the whole job lives on this machine — bind the
        // token's own host so an in-process transport binds where it expects to.
        let host = PeerAddress(host: id.address.host, port: 0).isLoopback ? id.address.host : bindHost
        let listener = try transport.listen(host: host, port: 0)
        // The rendezvous is a control channel and need not be the data
        // transport. They diverge for RDMA: rank 0's rendezvous listener is a
        // plain socket, and TN3205 has queue-pair metadata travel over TCP
        // anyway, so dialling rank 0 with a queue pair would be both wrong and
        // circular. Defaults to `transport` so every existing caller is
        // unaffected.
        let control = rendezvousTransport ?? transport
        do {
            let advertisements = try Rendezvous.join(
                id: id, rank: rank, worldSize: worldSize,
                dataListener: listener, transport: control,
                advertisedHosts: advertisedHosts, timeout: timeout)
            guard advertisements.count == worldSize else {
                throw MCCLError.protocolViolation(
                    "rendezvous returned \(advertisements.count) addresses for world size \(worldSize)")
            }
            guard worldSize > 1 else {
                listener.close()
                return Communicator(rank: 0, worldSize: 1,
                                    topology: topology ?? Topology.uniform(nodeCount: 1))
            }
            return try bootstrap(
                rank: rank, worldSize: worldSize, advertisements: advertisements,
                listener: listener, transport: transport, topology: topology, timeout: timeout)
        } catch {
            listener.close()
            throw error
        }
    }
}
