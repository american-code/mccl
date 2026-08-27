import Foundation

/// The token that lets `n` ranks find each other, mirroring `ncclUniqueId`.
///
/// It carries exactly two things: where rank 0 is listening, and a nonce that
/// distinguishes concurrent jobs on the same host. Everything else — each
/// rank's own data address — is discovered through rank 0, which is why the
/// token stays small enough to broadcast over MPI, a file, or an env var.
public struct UniqueID: Sendable, Hashable, CustomStringConvertible {
    /// Distinguishes two jobs bootstrapping through the same host and port.
    public let nonce: UInt64
    /// Rank 0's rendezvous listener.
    public let address: PeerAddress

    public init(nonce: UInt64, address: PeerAddress) {
        self.nonce = nonce
        self.address = address
    }

    /// `mccl1:<nonce hex>:<host>:<port>`. Text so it survives being pasted into
    /// a launcher, an env var, or a job script.
    public var text: String {
        String(format: "mccl1:%016llx:%@:%d", nonce, address.host, address.port)
    }

    public var description: String { text }

    public init?(text: String) {
        let parts = text.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 4, parts[0] == "mccl1",
              let nonce = UInt64(parts[1], radix: 16),
              let port = Int(parts[parts.count - 1]) else { return nil }
        let host = parts[2..<(parts.count - 1)].joined(separator: ":")
        guard !host.isEmpty else { return nil }
        self.init(nonce: nonce, address: PeerAddress(host: host, port: port))
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
    public static func createUniqueID(
        transport: Transport = TCPTransport(),
        host: String = "0.0.0.0",
        port: Int = 0,
        advertisedHost: String? = nil
    ) throws -> UniqueID {
        let listener = try transport.listen(host: host, port: port)
        var nonce = UInt64.random(in: 1...UInt64.max)
        lock.lock()
        while pending[nonce] != nil { nonce = UInt64.random(in: 1...UInt64.max) }
        pending[nonce] = listener
        lock.unlock()

        let advertised = advertisedHost
            ?? (host == "0.0.0.0" || host == "::" ? (NetworkInterfaces.preferredLocalAddress() ?? "127.0.0.1") : host)
        return UniqueID(nonce: nonce, address: PeerAddress(host: advertised, port: listener.address.port))
    }

    /// Releases a token's listener without running a rendezvous. Safe to call
    /// for a token that was already consumed.
    public static func discard(_ id: UniqueID) {
        lock.lock()
        let listener = pending.removeValue(forKey: id.nonce)
        lock.unlock()
        listener?.close()
    }

    /// Exchanges addresses and returns the world's peer table, indexed by rank.
    ///
    /// `dataListener` must already be bound; its address is what this rank
    /// advertises, with `advertisedHost` substituted for a wildcard bind.
    public static func join(
        id: UniqueID,
        rank: Int,
        worldSize: Int,
        dataListener: Listener,
        transport: Transport = TCPTransport(),
        advertisedHost: String? = nil,
        timeout: TimeInterval = 60
    ) throws -> [PeerAddress] {
        guard worldSize > 0 else { throw MCCLError.invalidArgument("worldSize must be > 0") }
        guard rank >= 0, rank < worldSize else {
            throw MCCLError.rankOutOfRange(rank: rank, worldSize: worldSize)
        }

        let bound = dataListener.address
        let host = advertisedHost ?? resolveAdvertisedHost(bound: bound.host, rendezvous: id.address.host)
        let mine = PeerAddress(host: host, port: bound.port)

        guard worldSize > 1 else {
            if rank == 0 { discard(id) }
            return [mine]
        }
        return rank == 0
            ? try serve(id: id, worldSize: worldSize, own: mine, transport: transport, timeout: timeout)
            : try announce(id: id, rank: rank, worldSize: worldSize, own: mine,
                           transport: transport, timeout: timeout)
    }

    // MARK: - Rank 0

    private static func serve(
        id: UniqueID, worldSize: Int, own: PeerAddress, transport: Transport, timeout: TimeInterval
    ) throws -> [PeerAddress] {
        lock.lock()
        let existing = pending.removeValue(forKey: id.nonce)
        lock.unlock()

        // A token generated in another process on this host: rebind its port.
        // The usual case is `existing`, straight from `createUniqueID`.
        let listener = try existing ?? transport.listen(host: "0.0.0.0", port: id.address.port)
        defer { listener.close() }

        var table = [PeerAddress?](repeating: nil, count: worldSize)
        table[0] = own
        var channels: [Int: Channel] = [:]
        defer { channels.values.forEach { $0.close() } }

        let scratch = ScratchBuffer(capacity: 4096)
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
            let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 4096)
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
            table[hello.rank] = hello.address
            channels[hello.rank] = channel
        }

        let resolved = try table.enumerated().map { index, address -> PeerAddress in
            guard let address else {
                throw MCCLError.protocolViolation("rendezvous: no address for rank \(index)")
            }
            return address
        }
        let payload = try JSONEncoder().encode(resolved)
        for (_, channel) in channels {
            try send(channel, tag: Tag.table, payload: payload)
        }
        return resolved
    }

    // MARK: - Ranks 1…n-1

    private static func announce(
        id: UniqueID, rank: Int, worldSize: Int, own: PeerAddress,
        transport: Transport, timeout: TimeInterval
    ) throws -> [PeerAddress] {
        // Rank 0 may still be starting; `connect` already retries to its own
        // deadline, so pass the rendezvous timeout straight through.
        let channel = try transport.connect(to: id.address, timeout: timeout)
        defer { channel.close() }

        let hello = Hello(nonce: id.nonce, rank: rank, worldSize: worldSize, address: own)
        try send(channel, tag: Tag.hello, payload: try JSONEncoder().encode(hello))

        let scratch = ScratchBuffer(capacity: 1 << 16)
        let header = try channel.receiveFrame(into: scratch, maxPayloadBytes: 1 << 20)
        guard header.tag == Tag.table else {
            throw MCCLError.protocolViolation("rendezvous: expected the address table, got tag \(header.tag)")
        }
        return try JSONDecoder().decode(
            [PeerAddress].self, from: Data(bytes: scratch.base, count: Int(header.payloadBytes)))
    }

    // MARK: - Wire helpers

    private enum Tag {
        static let hello: UInt32 = 0xBEE1
        static let table: UInt32 = 0xBEE2
    }

    private struct Hello: Codable {
        var nonce: UInt64
        var rank: Int
        /// Rank 0 fills this in from its own argument; a joining rank sends its
        /// own so a mismatched launch is caught rather than hanging.
        var worldSize: Int
        var address: PeerAddress
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
    /// this node has.
    static func resolveAdvertisedHost(bound: String, rendezvous: String) -> String {
        if let override = ProcessInfo.processInfo.environment["MCCL_HOST"], !override.isEmpty {
            return override
        }
        let isWildcard = bound == "0.0.0.0" || bound == "::" || bound.isEmpty
        guard isWildcard else { return bound }
        if PeerAddress(host: rendezvous, port: 0).isLoopback { return "127.0.0.1" }
        return NetworkInterfaces.preferredLocalAddress() ?? "127.0.0.1"
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
        bindHost: String = "0.0.0.0",
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
        do {
            let addresses = try Rendezvous.join(
                id: id, rank: rank, worldSize: worldSize,
                dataListener: listener, transport: transport, timeout: timeout)
            guard addresses.count == worldSize else {
                throw MCCLError.protocolViolation(
                    "rendezvous returned \(addresses.count) addresses for world size \(worldSize)")
            }
            guard worldSize > 1 else {
                listener.close()
                return Communicator(rank: 0, worldSize: 1,
                                    topology: topology ?? Topology.uniform(nodeCount: 1))
            }
            return try bootstrap(
                rank: rank, worldSize: worldSize, addresses: addresses,
                listener: listener, transport: transport, topology: topology, timeout: timeout)
        } catch {
            listener.close()
            throw error
        }
    }
}
