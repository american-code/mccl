import XCTest
@testable import MCCL

/// Interface-media discovery.
///
/// These run against the real machine, so they assert on what is true of *any*
/// Mac — there is always a loopback, its addresses are always 127.0.0.1 and ::1
/// — and cover the classification rules directly rather than hoping the test
/// host happens to have a Thunderbolt cable attached.
final class NetworkInterfaceTests: XCTestCase {

    func testEveryMacHasAClassifiedLoopback() {
        let interfaces = NetworkInterfaces.local()
        XCTAssertFalse(interfaces.isEmpty, "getifaddrs returned nothing")

        guard let loopback = interfaces.first(where: { $0.isLoopback }) else {
            return XCTFail("no loopback interface")
        }
        XCTAssertEqual(loopback.media, .loopback)
        XCTAssertEqual(loopback.linkKind(measuredBandwidth: 5e9), .loopback,
                       "no measurement can turn loopback into a cable")
        XCTAssertTrue(loopback.isUp)
        XCTAssertTrue(loopback.addresses.contains { $0.text == "127.0.0.1" })
    }

    func testInterfacesReportPlausibleShapes() {
        for interface in NetworkInterfaces.local() {
            XCTAssertFalse(interface.name.isEmpty)
            XCTAssertFalse(interface.addresses.isEmpty,
                           "\(interface.name) was listed with no addresses")
            for address in interface.addresses {
                XCTAssertFalse(address.text.isEmpty)
                XCTAssertFalse(address.text.contains("%"), "scope id should be stripped: \(address.text)")
                XCTAssertEqual(address.octets.count, address.isIPv6 ? 16 : 4)
                XCTAssertEqual(address.mask.count, address.octets.count)
            }
        }
    }

    func testSubnetMatchingFindsTheInterfaceAPeerIsReachedOn() {
        let interfaces = NetworkInterfaces.local()
        let loopback = NetworkInterfaces.interface(
            reaching: PeerAddress(host: "127.0.0.1", port: 7000), among: interfaces)
        XCTAssertEqual(loopback?.isLoopback, true)

        // A hostname the probe cannot resolve without DNS: only the loopback
        // aliases are answered, everything else declines rather than guessing.
        XCTAssertEqual(NetworkInterfaces.interface(
            reaching: PeerAddress(host: "localhost", port: 1), among: interfaces)?.isLoopback, true)
        XCTAssertNil(NetworkInterfaces.interface(
            reaching: PeerAddress(host: "some.host.example", port: 1), among: interfaces))
    }

    func testSubnetMembership() {
        func address(_ text: String, mask: String) -> InterfaceAddress {
            let host = NetworkInterfaces.parse(literal: text)!
            let maskBytes = NetworkInterfaces.parse(literal: mask)!.octets
            return InterfaceAddress(text: text, isIPv6: false, octets: host.octets, mask: maskBytes)
        }
        let lan = address("192.168.1.10", mask: "255.255.255.0")
        XCTAssertTrue(lan.sameSubnet(as: NetworkInterfaces.parse(literal: "192.168.1.99")!))
        XCTAssertFalse(lan.sameSubnet(as: NetworkInterfaces.parse(literal: "192.168.2.99")!))
        XCTAssertFalse(lan.sameSubnet(as: NetworkInterfaces.parse(literal: "fe80::1")!),
                       "v4 and v6 never share a subnet")

        // An all-zero mask matches everything, which is never a useful answer.
        let unmasked = address("10.0.0.1", mask: "0.0.0.0")
        XCTAssertFalse(unmasked.sameSubnet(as: NetworkInterfaces.parse(literal: "10.0.0.2")!))
    }

    func testClassificationRules() {
        // A Thunderbolt bridge is what macOS calls the TB IP fabric.
        XCTAssertEqual(NetworkInterfaces.classify(name: "bridge0", ifiType: 0x06,
                                                  isLoopback: false, displayName: "Thunderbolt Bridge"),
                       .thunderbolt)
        // Even without SystemConfiguration, `bridge*` is the TB fabric.
        XCTAssertEqual(NetworkInterfaces.classify(name: "bridge0", ifiType: 0x06,
                                                  isLoopback: false, displayName: nil),
                       .thunderbolt)
        // A TB port presenting as an ordinary en device is caught by the name.
        XCTAssertEqual(NetworkInterfaces.classify(name: "en2", ifiType: 0x06,
                                                  isLoopback: false, displayName: "Thunderbolt 2"),
                       .thunderbolt)
        XCTAssertEqual(NetworkInterfaces.classify(name: "en0", ifiType: 0x06,
                                                  isLoopback: false, displayName: "Wi-Fi"),
                       .wifi)
        XCTAssertEqual(NetworkInterfaces.classify(name: "en1", ifiType: 0x06,
                                                  isLoopback: false, displayName: "Ethernet"),
                       .ethernet)
        XCTAssertEqual(NetworkInterfaces.classify(name: "en1", ifiType: 0x06,
                                                  isLoopback: false, displayName: nil),
                       .ethernet, "an en device with IFT_ETHER is a cable by default")
        XCTAssertEqual(NetworkInterfaces.classify(name: "awdl0", ifiType: 0x06,
                                                  isLoopback: false, displayName: nil), .wifi)
        XCTAssertEqual(NetworkInterfaces.classify(name: "utun4", ifiType: nil,
                                                  isLoopback: false, displayName: nil), .other)
        XCTAssertEqual(NetworkInterfaces.classify(name: "lo0", ifiType: 0x18,
                                                  isLoopback: true, displayName: nil), .loopback)
    }

    func testMediaPlusMeasurementDecidesTheThunderboltGeneration() {
        func interface(_ media: InterfaceMedia) -> NetworkInterface {
            NetworkInterface(name: "x", media: media, displayName: nil, isUp: true,
                             isLoopback: media == .loopback, linkSpeedBitsPerSecond: 0, addresses: [])
        }
        let tb = interface(.thunderbolt)
        XCTAssertEqual(tb.linkKind(measuredBandwidth: 6.0e9), .thunderbolt5)
        XCTAssertEqual(tb.linkKind(measuredBandwidth: 2.5e9), .thunderbolt4)
        XCTAssertEqual(tb.linkKind(measuredBandwidth: 1.0e9), .usb4)
        XCTAssertEqual(tb.linkKind(measuredBandwidth: nil), .thunderbolt4,
                       "unmeasured, claim the older generation rather than the newer")

        // Media beats throughput for everything the media can actually settle.
        XCTAssertEqual(interface(.ethernet).linkKind(measuredBandwidth: 9e9), .ethernet,
                       "a fast measurement does not turn a cable into Thunderbolt")
        XCTAssertEqual(interface(.wifi).linkKind(measuredBandwidth: 9e9), .wifi)
        XCTAssertNil(interface(.other).linkKind(measuredBandwidth: 9e9),
                     "an unattributable interface must defer to inference")
    }

    func testClassifyKindFallsBackToThroughputForUnroutedPeers() {
        // No local interface owns this subnet on a normal machine, so the
        // throughput heuristic decides — the pre-existing behaviour.
        let far = PeerAddress(host: "203.0.113.7", port: 1)
        XCTAssertEqual(TopologyProbe.classifyKind(address: far, bandwidth: 6e9), .thunderbolt5)
        XCTAssertEqual(TopologyProbe.classifyKind(address: far, bandwidth: 1.1e8), .ethernet)

        // A loopback peer is settled by the interface, not the number.
        XCTAssertEqual(TopologyProbe.classifyKind(
            address: PeerAddress(host: "127.0.0.1", port: 1), bandwidth: 6e9), .loopback)
    }

    func testPreferredLocalAddressIsRoutableWhenOneExists() {
        func interface(_ name: String, _ media: InterfaceMedia, _ addresses: [String]) -> NetworkInterface {
            NetworkInterface(
                name: name, media: media, displayName: nil, isUp: true, isLoopback: false,
                linkSpeedBitsPerSecond: 0,
                addresses: addresses.map { text in
                    let parsed = NetworkInterfaces.parse(literal: text)!
                    return InterfaceAddress(text: text, isIPv6: parsed.isIPv6,
                                            octets: parsed.octets, mask: parsed.mask)
                })
        }
        // Thunderbolt outranks Ethernet outranks Wi-Fi.
        XCTAssertEqual(NetworkInterfaces.preferredLocalAddress(among: [
            interface("en0", .wifi, ["192.168.1.5"]),
            interface("en1", .ethernet, ["10.0.0.5"]),
            interface("bridge0", .thunderbolt, ["169.254.1.2"]),
        ]), "169.254.1.2", "a self-assigned address is normal on a TB point-to-point bridge")

        // …but a self-assigned Ethernet address means the interface failed to
        // configure, so skip it.
        XCTAssertEqual(NetworkInterfaces.preferredLocalAddress(among: [
            interface("en1", .ethernet, ["169.254.9.9"]),
            interface("en0", .wifi, ["192.168.1.5"]),
        ]), "192.168.1.5")

        XCTAssertNil(NetworkInterfaces.preferredLocalAddress(among: []))
    }

    func testKindEnumStillRoundTripsThroughJSON() throws {
        // `wifi` was added for interface media; a persisted topology must
        // survive it.
        for kind in [Topology.Link.Kind.thunderbolt5, .thunderbolt4, .usb4, .ethernet, .wifi, .loopback] {
            let topology = Topology(
                nodes: [Topology.Node(id: 0, hostname: "a", chip: "c", unifiedMemoryBytes: 1),
                        Topology.Node(id: 1, hostname: "b", chip: "c", unifiedMemoryBytes: 1)],
                links: [Topology.Link(from: 0, to: 1, kind: kind,
                                      measuredBandwidth: 1e9, measuredLatency: 1e-4)])
            let restored = try Topology.from(jsonData: topology.jsonData())
            XCTAssertEqual(restored.links[0].kind, kind)
        }
    }
}
