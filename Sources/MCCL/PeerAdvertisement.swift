import Foundation

/// One address a rank can be reached at, tagged with the interface kind the
/// *advertising* rank believes it sits on.
///
/// The tag is advisory and never load-bearing. A dialer ranks a path by the
/// media of its own egress interface — something it can observe — because the
/// peer's claim about its own hardware is unverifiable and, in a mixed fabric,
/// not even the relevant half: what matters is the cable the bytes leave on.
/// The tag is kept because it makes a log line and a rendezvous table readable,
/// and because it breaks ties deterministically when two paths look alike from
/// the dialer's side.
public struct PeerEndpoint: Sendable, Hashable, Codable, CustomStringConvertible {
    public var address: PeerAddress
    public var media: InterfaceMedia

    public init(address: PeerAddress, media: InterfaceMedia) {
        self.address = address
        self.media = media
    }

    public var description: String { "\(address) (\(media.rawValue))" }

    // Tolerant media decoding: a token or table written by a later version that
    // knows a media kind this build does not should degrade to `.other`, not
    // fail the whole rendezvous.
    private enum CodingKeys: String, CodingKey { case address, media }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        address = try container.decode(PeerAddress.self, forKey: .address)
        let raw = try container.decodeIfPresent(String.self, forKey: .media) ?? InterfaceMedia.other.rawValue
        media = InterfaceMedia(rawValue: raw) ?? .other
    }
}

/// Everything one rank publishes about where it can be reached.
///
/// `MeshFabric` used to carry exactly one `PeerAddress` per rank, which is
/// enough for a uniform fabric and wrong for every interesting one: a world
/// spanning a Thunderbolt island and a Wi-Fi bridge has no single address per
/// rank that all of its peers can dial. An advertisement carries all of them and
/// lets each *pair* pick its own best path.
public struct PeerAdvertisement: Sendable, Hashable, Codable {
    public var rank: Int
    /// Every address this rank's listener can be reached at. Order is the
    /// advertiser's own preference and is only a tiebreak for the dialer.
    public var endpoints: [PeerEndpoint]

    public init(rank: Int, endpoints: [PeerEndpoint]) {
        self.rank = rank
        self.endpoints = endpoints
    }

    /// The single-address form, for callers that know one address per rank.
    public init(rank: Int, address: PeerAddress, media: InterfaceMedia? = nil) {
        let kind = media ?? (address.isLoopback ? .loopback : .other)
        self.init(rank: rank, endpoints: [PeerEndpoint(address: address, media: kind)])
    }

    /// The first endpoint, for callers and diagnostics that still want one
    /// address to name a rank by.
    public var primaryAddress: PeerAddress? { endpoints.first?.address }

    public var hosts: [String] { endpoints.map(\.address.host) }

    /// True when this advertisement offers nothing but loopback literals — the
    /// signature of a world that lives entirely on one machine.
    public var isLoopbackOnly: Bool {
        !endpoints.isEmpty && endpoints.allSatisfy(\.address.isLoopback)
    }

    /// Whether a connection claiming to be this rank, arriving from `observed`,
    /// contradicts what this rank advertised.
    ///
    /// The source address is the one fact about an inbound connection that the
    /// announcer does not get to choose, so it is the only part of a bootstrap
    /// handshake that can be checked at all. Every rank advertises every usable
    /// address it owns, so whichever one it egresses from is always in its own
    /// list; an announcement from anywhere else is either a rank launched with
    /// a `--bind` that contradicts its own routing — in which case the world
    /// would half-form and then deadlock — or a third party claiming a seat.
    ///
    /// False whenever the check cannot be made rather than guessed at:
    ///
    /// * `nil` source — a transport with no addressing of its own.
    /// * an empty advertisement — nothing to contradict.
    /// * a loopback source — this machine talking to itself, which every
    ///   in-process and single-host world does and no remote party can forge.
    public func disavows(source observed: PeerAddress?) -> Bool {
        guard let observed, !endpoints.isEmpty, !observed.isLoopback else { return false }
        return !hosts.contains(observed.host)
    }
}

// MARK: - Building this rank's advertisement

extension PeerAdvertisement {
    /// Every usable local address, at `port`, in the advertiser's own
    /// preference order.
    ///
    /// "Usable" means: on an interface that is up and not loopback, IPv4, and
    /// either routable or link-local-on-Thunderbolt. That last clause is the
    /// same rule `NetworkInterfaces.preferredLocalAddress()` already applies —
    /// a 169.254/16 address is normal on a point-to-point TB bridge with no
    /// DHCP server, and a symptom of a failed configuration anywhere else.
    ///
    /// IPv4 only, deliberately: IPv6 temporary-privacy addresses rotate, and an
    /// advertisement whose entries expire is worse than one that is merely
    /// incomplete. A cluster that needs IPv6 passes its addresses explicitly.
    public static func local(
        rank: Int, port: Int, interfaces: [NetworkInterface]? = nil
    ) -> PeerAdvertisement {
        let candidates = (interfaces ?? NetworkInterfaces.local()).filter { $0.isUp && !$0.isLoopback }
        var endpoints: [PeerEndpoint] = []
        var seen: Set<String> = []
        for media in PathSelection.mediaOrder {
            for interface in candidates where interface.media == media {
                for address in interface.addresses where !address.isIPv6 {
                    let linkLocal = NetworkInterfaces.isLinkLocal(address)
                    guard !linkLocal || media == .thunderbolt else { continue }
                    guard seen.insert(address.text).inserted else { continue }
                    endpoints.append(
                        PeerEndpoint(address: PeerAddress(host: address.text, port: port), media: media))
                }
            }
        }
        return PeerAdvertisement(rank: rank, endpoints: endpoints)
    }

    /// This rank's advertisement from an explicit list of hosts — what
    /// `mcclbench --bind a,b,c` produces. Each host is classified against the
    /// local interface table so the dialer's tiebreak still has something true
    /// to work with; a host this machine does not own is tagged `.other`.
    public static func explicit(
        rank: Int, hosts: [String], port: Int, interfaces: [NetworkInterface]? = nil
    ) -> PeerAdvertisement {
        let table = interfaces ?? NetworkInterfaces.local()
        var endpoints: [PeerEndpoint] = []
        var seen: Set<String> = []
        for host in hosts where seen.insert(host).inserted {
            let address = PeerAddress(host: host, port: port)
            let media: InterfaceMedia
            if address.isLoopback {
                media = .loopback
            } else {
                media = table.first(where: { interface in
                    interface.addresses.contains(where: { $0.text == host })
                })?.media ?? .other
            }
            endpoints.append(PeerEndpoint(address: address, media: media))
        }
        return PeerAdvertisement(rank: rank, endpoints: endpoints)
    }
}

// MARK: - Choosing a path between one pair

/// How a dialer orders the endpoints a peer advertised.
///
/// The rule, in full, because a planner that picks a fabric silently is a
/// planner nobody can debug:
///
/// 1. For each advertised endpoint, find the **local egress interface** — the
///    local interface whose subnet contains that address, longest prefix first.
///    The path's kind is *that interface's* media. An endpoint no local subnet
///    matches is `unroutable`: still dialable through the default route, but
///    ranked last, because nothing here can say what it would cross.
/// 2. Sort by kind priority:
///    `loopback > thunderbolt > ethernet > wifi > other > unroutable`.
/// 3. Break ties on the advertiser's own media tag, in the same order, and then
///    on the endpoint's position in the advertisement — the advertiser's stated
///    preference. Both tiebreaks are pure functions of the advertisement, so
///    every rank derives the same order for it.
/// 4. Drop loopback candidates unless the dialer itself has nothing but
///    loopback to offer. `127.0.0.1` in a remote peer's advertisement names the
///    dialer's own machine; honouring it would dial the wrong host entirely.
///
/// Order alone is not enough on a link-local fabric. Three Thunderbolt ports on
/// one machine all carry `169.254/16` addresses, and a /16 subnet match cannot
/// tell which of them has the cable that reaches this peer. So the order is
/// resolved by *dialing*: `MeshFabric.dial` walks it with a short per-attempt
/// timeout and takes the first connection that completes. Reachability is
/// measured, not inferred — which is the same stance the planner takes towards
/// bandwidth.
public enum PathSelection {

    /// Media, fastest first. The one place the ordering is written down.
    public static let mediaOrder: [InterfaceMedia] = [.loopback, .thunderbolt, .ethernet, .wifi, .other]

    /// Rank of a media kind; lower sorts earlier. `nil` means "no local
    /// interface claims this address", which sorts after every known kind.
    public static func priority(_ media: InterfaceMedia?) -> Int {
        guard let media, let index = mediaOrder.firstIndex(of: media) else { return mediaOrder.count }
        return index
    }

    /// One candidate path to a peer, with the reasoning attached.
    public struct Candidate: Sendable, Equatable, CustomStringConvertible {
        public let endpoint: PeerEndpoint
        /// Media of the local interface this address would leave on, or nil
        /// when no local subnet contains it.
        public let egressMedia: InterfaceMedia?
        /// BSD name of that interface, for logs.
        public let egressInterface: String?
        /// Position in the advertisement, the advertiser's own preference.
        public let advertisedIndex: Int

        public var description: String {
            let via = egressInterface.map { "\($0)/\(egressMedia?.rawValue ?? "?")" } ?? "default route"
            return "\(endpoint.address) via \(via)"
        }
    }

    /// The ordered candidate list for dialing `advertisement`.
    ///
    /// `own` is this rank's own advertisement, used only for rule 4.
    public static func candidates(
        for advertisement: PeerAdvertisement,
        own: PeerAdvertisement?,
        interfaces: [NetworkInterface]? = nil
    ) -> [Candidate] {
        let table = interfaces ?? NetworkInterfaces.local()
        let selfIsLoopbackOnly = own.map(\.isLoopbackOnly) ?? false

        var considered = Array(advertisement.endpoints.enumerated())
        if !selfIsLoopbackOnly {
            let routable = considered.filter { !$0.element.address.isLoopback }
            // Only drop the loopback entries if something else remains; a peer
            // that offered nothing else is a peer we still have to try.
            if !routable.isEmpty { considered = routable }
        }

        let scored = considered.map { index, endpoint -> Candidate in
            let interface = NetworkInterfaces.interface(reaching: endpoint.address, among: table)
            return Candidate(endpoint: endpoint,
                             egressMedia: interface?.media,
                             egressInterface: interface?.name,
                             advertisedIndex: index)
        }
        return scored.sorted { a, b in
            let pa = priority(a.egressMedia), pb = priority(b.egressMedia)
            if pa != pb { return pa < pb }
            let ta = priority(a.endpoint.media), tb = priority(b.endpoint.media)
            if ta != tb { return ta < tb }
            return a.advertisedIndex < b.advertisedIndex
        }
    }
}
