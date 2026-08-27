import Foundation
#if canImport(Darwin)
import Darwin
#endif
#if canImport(SystemConfiguration)
import SystemConfiguration
#endif

/// What a local interface actually *is*, as opposed to what it managed to
/// achieve in a bandwidth test.
///
/// Deliberately coarse. The kernel and SystemConfiguration can tell Ethernet
/// from Wi-Fi from a Thunderbolt bridge; neither can tell TB4 from TB5, because
/// both present as ordinary IP links. Generation is left to the measurement.
public enum InterfaceMedia: String, Sendable, Codable, CaseIterable {
    case thunderbolt
    case ethernet
    case wifi
    case loopback
    /// Tunnels, awdl, bridges we could not attribute, anything unrecognised.
    /// Callers fall back to throughput inference for these.
    case other
}

/// One address on one local interface, with the netmask that defines its
/// subnet — which is what lets us decide which interface a peer is reached on.
public struct InterfaceAddress: Sendable, Equatable, CustomStringConvertible {
    public let text: String
    public let isIPv6: Bool
    let octets: [UInt8]
    let mask: [UInt8]

    init(text: String, isIPv6: Bool, octets: [UInt8], mask: [UInt8]) {
        self.text = text
        self.isIPv6 = isIPv6
        self.octets = octets
        self.mask = mask
    }

    public var description: String { text }

    /// True when `other` sits inside this address's subnet, i.e. traffic to it
    /// leaves through this interface.
    public func sameSubnet(as other: InterfaceAddress) -> Bool {
        guard isIPv6 == other.isIPv6, octets.count == other.octets.count,
              mask.count == octets.count, mask.contains(where: { $0 != 0 }) else { return false }
        for i in 0..<octets.count where (octets[i] & mask[i]) != (other.octets[i] & mask[i]) {
            return false
        }
        return true
    }
}

/// A local network interface as the kernel describes it.
public struct NetworkInterface: Sendable, Equatable {
    /// BSD name — `en0`, `bridge0`, `lo0`.
    public let name: String
    public let media: InterfaceMedia
    /// SystemConfiguration's user-visible name, e.g. "Thunderbolt Bridge".
    /// Nil for interfaces SystemConfiguration does not model.
    public let displayName: String?
    public let isUp: Bool
    public let isLoopback: Bool
    /// `ifi_baudrate`, bits/sec. 0 when the driver does not report one.
    public let linkSpeedBitsPerSecond: UInt64
    public let addresses: [InterfaceAddress]

    /// The topology link kind this interface implies, refined by a measured
    /// bandwidth where the media alone cannot distinguish generations.
    ///
    /// Returns nil for `.other`: an honest "I do not know" beats mislabelling a
    /// tunnel as Ethernet, and the caller then falls back to inference.
    public func linkKind(measuredBandwidth: Double?) -> Topology.Link.Kind? {
        switch media {
        case .loopback: return .loopback
        case .ethernet: return .ethernet
        case .wifi: return .wifi
        case .other: return nil
        case .thunderbolt:
            // TB4/USB4 and TB5 are the same kind of link to every API on the
            // machine. Only throughput separates them, so use it when we have
            // it and admit to the older generation when we do not.
            guard let bandwidth = measuredBandwidth, bandwidth > 0 else { return .thunderbolt4 }
            if bandwidth >= 4.0e9 { return .thunderbolt5 }
            if bandwidth >= 1.8e9 { return .thunderbolt4 }
            return .usb4
        }
    }
}

// MARK: - Enumeration

public enum NetworkInterfaces {

    /// Every interface with at least one IP address, newest snapshot.
    ///
    /// `getifaddrs` supplies addresses, flags and `if_data`; SystemConfiguration
    /// supplies the human-facing identity that distinguishes a Thunderbolt
    /// bridge from any other `en`/`bridge` device. Both are OS facilities — mccl
    /// still has no package dependencies.
    public static func local() -> [NetworkInterface] {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return [] }
        defer { freeifaddrs(head) }

        var addresses: [String: [InterfaceAddress]] = [:]
        var flags: [String: UInt32] = [:]
        var speeds: [String: UInt64] = [:]
        var types: [String: UInt8] = [:]
        var order: [String] = []

        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let entry = cursor {
            defer { cursor = entry.pointee.ifa_next }
            let name = String(cString: entry.pointee.ifa_name)
            if !order.contains(name) { order.append(name) }
            flags[name] = entry.pointee.ifa_flags

            guard let sockaddrPointer = entry.pointee.ifa_addr else { continue }
            let family = sockaddrPointer.pointee.sa_family

            if family == UInt8(AF_LINK) {
                // The link-layer entry carries if_data: media type and speed.
                if let data = entry.pointee.ifa_data?.assumingMemoryBound(to: if_data.self) {
                    types[name] = data.pointee.ifi_type
                    speeds[name] = UInt64(data.pointee.ifi_baudrate)
                }
                continue
            }
            guard family == UInt8(AF_INET) || family == UInt8(AF_INET6) else { continue }
            guard let address = parse(sockaddrPointer, netmask: entry.pointee.ifa_netmask) else { continue }
            addresses[name, default: []].append(address)
        }

        let described = systemConfigurationNames()
        return order.compactMap { name -> NetworkInterface? in
            let entries = addresses[name] ?? []
            guard !entries.isEmpty else { return nil }
            let interfaceFlags = flags[name] ?? 0
            let isLoopback = (interfaceFlags & UInt32(IFF_LOOPBACK)) != 0
            return NetworkInterface(
                name: name,
                media: classify(name: name, ifiType: types[name],
                                isLoopback: isLoopback, displayName: described[name]),
                displayName: described[name],
                isUp: (interfaceFlags & UInt32(IFF_UP)) != 0,
                isLoopback: isLoopback,
                linkSpeedBitsPerSecond: speeds[name] ?? 0,
                addresses: entries)
        }
    }

    /// The interface a peer at `address` would be reached on: the one whose
    /// subnet contains it. Nil when nothing matches (a routed peer several hops
    /// away, most often).
    public static func interface(reaching address: PeerAddress,
                                 among interfaces: [NetworkInterface]? = nil) -> NetworkInterface? {
        let candidates = interfaces ?? local()
        guard let target = parse(literal: address.host) else {
            // Not a numeric literal: only a loopback name can be resolved
            // without doing DNS inside the probe's hot path.
            guard address.isLoopback else { return nil }
            return candidates.first(where: \.isLoopback)
        }
        // Longest match first, so a /24 on en0 wins over a /8 that also covers it.
        var best: (NetworkInterface, Int)?
        for interface in candidates {
            for local in interface.addresses where local.sameSubnet(as: target) {
                let bits = local.mask.reduce(0) { $0 + $1.nonzeroBitCount }
                if best == nil || bits > best!.1 { best = (interface, bits) }
            }
        }
        return best?.0
    }

    // MARK: - Classification

    // <net/if_types.h>. Not exposed to Swift, and stable since 4.4BSD.
    private static let iftEther: UInt8 = 0x06
    private static let iftLoop: UInt8 = 0x18
    private static let iftBridge: UInt8 = 0xd1

    static func classify(name: String, ifiType: UInt8?, isLoopback: Bool, displayName: String?) -> InterfaceMedia {
        if isLoopback || ifiType == iftLoop || name.hasPrefix("lo") { return .loopback }

        if let displayName {
            let lowered = displayName.lowercased()
            if lowered.contains("thunderbolt") { return .thunderbolt }
            if lowered.contains("wi-fi") || lowered.contains("wifi") || lowered.contains("airport") {
                return .wifi
            }
            if lowered.contains("ethernet") || lowered.contains("lan") { return .ethernet }
        }

        // macOS names the Thunderbolt IP fabric `bridge0` and nothing else uses
        // a bridge on a stock machine.
        if name.hasPrefix("bridge") || ifiType == iftBridge { return .thunderbolt }
        // Apple Wireless Direct Link, the Wi-Fi hotspot interface and the
        // low-latency WLAN shim are all radios, not cables.
        if name.hasPrefix("awdl") || name.hasPrefix("llw") || name.hasPrefix("ap") { return .wifi }
        if name.hasPrefix("en"), ifiType == iftEther { return .ethernet }
        return .other
    }

    /// BSD name -> localized display name, via SystemConfiguration.
    private static func systemConfigurationNames() -> [String: String] {
        #if canImport(SystemConfiguration)
        guard let all = SCNetworkInterfaceCopyAll() as? [SCNetworkInterface] else { return [:] }
        var result: [String: String] = [:]
        for interface in all {
            guard let bsd = SCNetworkInterfaceGetBSDName(interface) as String? else { continue }
            if let display = SCNetworkInterfaceGetLocalizedDisplayName(interface) as String? {
                result[bsd] = display
            }
        }
        return result
        #else
        return [:]
        #endif
    }

    // MARK: - Address parsing

    private static func parse(_ address: UnsafeMutablePointer<sockaddr>,
                              netmask: UnsafeMutablePointer<sockaddr>?) -> InterfaceAddress? {
        let family = address.pointee.sa_family
        let width = family == UInt8(AF_INET) ? 4 : 16
        guard let octets = rawAddress(address, width: width) else { return nil }
        let mask = netmask.flatMap { rawAddress($0, width: width) }
            ?? [UInt8](repeating: 0xFF, count: width)

        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        let length = socklen_t(family == UInt8(AF_INET)
                               ? MemoryLayout<sockaddr_in>.size : MemoryLayout<sockaddr_in6>.size)
        guard getnameinfo(address, length, &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else {
            return nil
        }
        // Strip the scope suffix ("fe80::1%en0") so text compares cleanly.
        let text = String(cString: host).split(separator: "%").first.map(String.init) ?? ""
        return InterfaceAddress(text: text, isIPv6: family == UInt8(AF_INET6),
                                octets: octets, mask: mask)
    }

    private static func rawAddress(_ address: UnsafeMutablePointer<sockaddr>, width: Int) -> [UInt8]? {
        // sin_addr sits at offset 4 of sockaddr_in; sin6_addr at offset 8 of
        // sockaddr_in6. A netmask sockaddr may be short, so read defensively.
        let offset = width == 4 ? 4 : 8
        let declared = Int(address.pointee.sa_len)
        let available = max(0, declared - offset)
        guard available > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: width)
        let raw = UnsafeRawPointer(address)
        for i in 0..<min(width, available) {
            bytes[i] = raw.loadUnaligned(fromByteOffset: offset + i, as: UInt8.self)
        }
        return bytes
    }

    /// Parses a numeric IPv4/IPv6 literal into a host-route address (all-ones
    /// mask), for subnet membership tests.
    static func parse(literal host: String) -> InterfaceAddress? {
        var v4 = in_addr()
        if inet_pton(AF_INET, host, &v4) == 1 {
            let value = v4.s_addr.bigEndian
            let octets = [UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16),
                          UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]
            return InterfaceAddress(text: host, isIPv6: false, octets: octets,
                                    mask: [UInt8](repeating: 0xFF, count: 4))
        }
        var v6 = in6_addr()
        if inet_pton(AF_INET6, host, &v6) == 1 {
            let octets = withUnsafeBytes(of: &v6) { Array($0.bindMemory(to: UInt8.self)) }
            return InterfaceAddress(text: host, isIPv6: true, octets: octets,
                                    mask: [UInt8](repeating: 0xFF, count: 16))
        }
        return nil
    }

    /// The address other machines should dial to reach this one: an IPv4 on the
    /// fattest real cable, preferring Thunderbolt over Ethernet over Wi-Fi.
    /// Used by `Rendezvous` to advertise a rank's data listener.
    ///
    /// A 169.254/16 self-assigned address counts only on Thunderbolt, where a
    /// point-to-point bridge with no DHCP server is the normal case; on any
    /// other media it means the interface failed to configure.
    public static func preferredLocalAddress(among interfaces: [NetworkInterface]? = nil) -> String? {
        let candidates = (interfaces ?? local()).filter { $0.isUp && !$0.isLoopback }
        let ranking: [InterfaceMedia] = [.thunderbolt, .ethernet, .wifi, .other]
        for media in ranking {
            for interface in candidates where interface.media == media {
                let v4 = interface.addresses.filter { !$0.isIPv6 }
                if let routable = v4.first(where: { !isLinkLocal($0) }) { return routable.text }
                if media == .thunderbolt, let selfAssigned = v4.first { return selfAssigned.text }
            }
        }
        return nil
    }

    static func isLinkLocal(_ address: InterfaceAddress) -> Bool {
        guard !address.isIPv6 else { return address.octets.first == 0xFE }
        return address.octets.count == 4 && address.octets[0] == 169 && address.octets[1] == 254
    }
}
