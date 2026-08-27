import Foundation
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Node identity

/// What a node reports about itself when probed.
public struct NodeIdentity: Sendable, Codable, Equatable {
    public var hostname: String
    public var chip: String
    public var unifiedMemoryBytes: UInt64

    public init(hostname: String, chip: String, unifiedMemoryBytes: UInt64) {
        self.hostname = hostname
        self.chip = chip
        self.unifiedMemoryBytes = unifiedMemoryBytes
    }

    /// Reads the local machine's identity out of sysctl.
    public static func local() -> NodeIdentity {
        var memory: UInt64 = 0
        var memorySize = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &memory, &memorySize, nil, 0)

        var brand = [CChar](repeating: 0, count: 256)
        var brandSize = brand.count
        if sysctlbyname("machdep.cpu.brand_string", &brand, &brandSize, nil, 0) != 0 {
            brand[0] = 0
        }

        return NodeIdentity(
            hostname: ProcessInfo.processInfo.hostName,
            chip: String(cString: brand),
            unifiedMemoryBytes: memory
        )
    }

    public func node(id: Int) -> Topology.Node {
        Topology.Node(id: id, hostname: hostname, chip: chip, unifiedMemoryBytes: unifiedMemoryBytes)
    }
}

// MARK: - Protocol

enum ProbeOpcode: UInt32 {
    case hello = 1
    case helloReply = 2
    case ping = 3
    case pong = 4
    case sink = 5
    case sinkAck = 6
    case probePeer = 7
    case probePeerReply = 8
    case failure = 9
    case bye = 10
}

/// One measured edge, as produced by `TopologyProbe`.
public struct PeerMeasurement: Sendable, Codable, Equatable {
    public var address: PeerAddress
    public var identity: NodeIdentity
    /// Bytes per second, from large streaming transfers.
    public var bandwidthBytesPerSecond: Double
    /// Seconds, minimum over a small-payload ping-pong.
    public var roundTripSeconds: Double

    public init(address: PeerAddress, identity: NodeIdentity,
                bandwidthBytesPerSecond: Double, roundTripSeconds: Double) {
        self.address = address
        self.identity = identity
        self.bandwidthBytesPerSecond = bandwidthBytesPerSecond
        self.roundTripSeconds = roundTripSeconds
    }
}

struct ProbeRequest: Codable {
    var address: PeerAddress
    var streamBytes: Int
    var chunkBytes: Int
    var pingIterations: Int
    var warmupIterations: Int
}

// MARK: - Server

/// `mcclprobe serve`. Answers identity, ping-pong and bulk-sink requests, and
/// will itself probe a third node on request so the driver can build a genuine
/// pairwise map rather than a star centred on itself.
public final class ProbeServer: @unchecked Sendable {
    public let address: PeerAddress
    private let listener: Listener
    private let transport: Transport
    private let maxPayloadBytes: Int
    private let lock = NSLock()
    private var running = false

    public init(
        transport: Transport = TCPTransport(),
        host: String = "0.0.0.0",
        port: Int = 0,
        maxPayloadBytes: Int = 256 << 20
    ) throws {
        self.transport = transport
        self.maxPayloadBytes = maxPayloadBytes
        self.listener = try transport.listen(host: host, port: port)
        self.address = listener.address
    }

    public func start() {
        lock.lock()
        guard !running else { lock.unlock(); return }
        running = true
        lock.unlock()

        Thread.detachNewThread { [self] in
            while isRunning {
                do {
                    let channel = try listener.accept(timeout: 0.5)
                    Thread.detachNewThread { [self] in serve(channel) }
                } catch MCCLError.timedOut {
                    continue
                } catch {
                    return
                }
            }
        }
    }

    private var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    public func stop() {
        lock.lock()
        running = false
        lock.unlock()
        listener.close()
    }

    private func serve(_ channel: Channel) {
        defer { channel.close() }
        let scratch = ScratchBuffer(capacity: 1 << 20)
        while isRunning {
            let header: WireHeader
            do {
                header = try channel.receiveFrame(into: scratch, maxPayloadBytes: maxPayloadBytes)
            } catch {
                return
            }
            do {
                guard let opcode = ProbeOpcode(rawValue: header.tag) else {
                    throw MCCLError.protocolViolation("unknown probe opcode \(header.tag)")
                }
                switch opcode {
                case .hello:
                    let payload = try JSONEncoder().encode(NodeIdentity.local())
                    try reply(channel, .helloReply, payload: payload, scratch: scratch)

                case .ping:
                    // Echo verbatim: the caller times the whole round trip.
                    var pong = WireHeader()
                    pong.tag = ProbeOpcode.pong.rawValue
                    pong.dataType = DataType.int8.wireCode
                    pong.elementCount = header.elementCount
                    let bytes = Int(header.payloadBytes)
                    let out = ScratchBuffer(capacity: WireHeader.byteCount + bytes)
                    if bytes > 0 {
                        (out.base + WireHeader.byteCount).copyMemory(from: scratch.base, byteCount: bytes)
                    }
                    try channel.sendPreparedFrame(pong, payloadBytes: bytes, in: out)

                case .sink:
                    var ack = WireHeader()
                    ack.tag = ProbeOpcode.sinkAck.rawValue
                    ack.dataType = DataType.int8.wireCode
                    ack.elementCount = header.elementCount
                    let out = ScratchBuffer(capacity: WireHeader.byteCount)
                    try channel.sendPreparedFrame(ack, payloadBytes: 0, in: out)

                case .probePeer:
                    let data = Data(bytes: scratch.base, count: Int(header.payloadBytes))
                    let request = try JSONDecoder().decode(ProbeRequest.self, from: data)
                    var options = TopologyProbe.Options()
                    options.streamBytes = request.streamBytes
                    options.chunkBytes = request.chunkBytes
                    options.pingIterations = request.pingIterations
                    options.warmupIterations = request.warmupIterations
                    options.pairwise = false
                    let measurement = try TopologyProbe.measure(
                        peer: request.address, transport: transport, options: options)
                    try reply(channel, .probePeerReply,
                              payload: try JSONEncoder().encode(measurement), scratch: scratch)

                case .bye:
                    return

                case .helloReply, .pong, .sinkAck, .probePeerReply, .failure:
                    throw MCCLError.protocolViolation("server received response opcode \(opcode)")
                }
            } catch {
                let text = Data("\(error)".utf8)
                try? reply(channel, .failure, payload: text, scratch: scratch)
                return
            }
        }
    }

    private func reply(_ channel: Channel, _ opcode: ProbeOpcode, payload: Data, scratch: ScratchBuffer) throws {
        let out = ScratchBuffer(capacity: WireHeader.byteCount + payload.count)
        payload.withUnsafeBytes { src in
            if !payload.isEmpty {
                (out.base + WireHeader.byteCount).copyMemory(from: src.baseAddress!, byteCount: payload.count)
            }
        }
        var header = WireHeader()
        header.tag = opcode.rawValue
        header.dataType = DataType.int8.wireCode
        header.elementCount = UInt32(payload.count)
        try channel.sendPreparedFrame(header, payloadBytes: payload.count, in: out)
    }
}

// MARK: - Client

/// `mcclprobe measure`. Drives real transfers against `ProbeServer` instances
/// and turns the timings into a `Topology` the planner can consume.
public enum TopologyProbe {
    public struct Options: Sendable {
        /// Total bytes pushed when measuring bandwidth.
        public var streamBytes: Int = 64 << 20
        /// Size of each streamed frame. Large enough that per-frame round trip
        /// is noise, small enough to stay inside socket buffers.
        public var chunkBytes: Int = 8 << 20
        /// Small-payload round trips used for the latency estimate.
        public var pingIterations: Int = 100
        public var warmupIterations: Int = 10
        /// Also ask each peer to probe every other peer, producing a true
        /// pairwise map instead of a star centred on the driving node.
        public var pairwise: Bool = true
        public var timeout: TimeInterval = 30

        public init() {}
    }

    /// Measures one link: bandwidth from bulk streaming, RTT from ping-pong.
    public static func measure(
        peer address: PeerAddress,
        transport: Transport = TCPTransport(),
        options: Options = Options()
    ) throws -> PeerMeasurement {
        let channel = try transport.connect(to: address, timeout: options.timeout)
        defer { channel.close() }
        let scratch = ScratchBuffer(capacity: max(options.chunkBytes, 1 << 20))

        // Identity.
        try request(channel, .hello, payloadBytes: 0, scratch: scratch)
        let helloHeader = try expect(channel, .helloReply, scratch: scratch, maxPayloadBytes: 1 << 20)
        let identity = try JSONDecoder().decode(
            NodeIdentity.self,
            from: Data(bytes: scratch.base, count: Int(helloHeader.payloadBytes)))

        // Latency: minimum of many tiny round trips. The minimum, not the mean —
        // scheduling noise only ever adds time.
        let pingBytes = 8
        for _ in 0..<max(0, options.warmupIterations) {
            try request(channel, .ping, payloadBytes: pingBytes, scratch: scratch)
            _ = try expect(channel, .pong, scratch: scratch, maxPayloadBytes: pingBytes)
        }
        var bestRoundTrip = Double.infinity
        let iterations = max(1, options.pingIterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            try request(channel, .ping, payloadBytes: pingBytes, scratch: scratch)
            _ = try expect(channel, .pong, scratch: scratch, maxPayloadBytes: pingBytes)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            bestRoundTrip = Swift.min(bestRoundTrip, elapsed)
        }

        // Bandwidth: stream `streamBytes` in chunks, each acknowledged, and
        // divide by the wall clock across the whole run.
        let chunk = max(4096, Swift.min(options.chunkBytes, options.streamBytes))
        let chunks = max(1, options.streamBytes / chunk)
        scratch.ensure(WireHeader.byteCount + chunk)
        // Non-zero, non-uniform bytes: keeps any transport-level compression
        // from flattering the measurement.
        let fill = UnsafeMutableRawBufferPointer(start: scratch.base + WireHeader.byteCount, count: chunk)
        for i in stride(from: 0, to: chunk, by: 64) { fill[i] = UInt8(truncatingIfNeeded: i &* 31 &+ 7) }

        // One warm-up chunk so TCP's window is open before the clock starts.
        try request(channel, .sink, payloadBytes: chunk, scratch: scratch)
        _ = try expect(channel, .sinkAck, scratch: scratch, maxPayloadBytes: 0)

        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<chunks {
            try request(channel, .sink, payloadBytes: chunk, scratch: scratch)
            _ = try expect(channel, .sinkAck, scratch: scratch, maxPayloadBytes: 0)
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let bandwidth = seconds > 0 ? Double(chunk * chunks) / seconds : 0

        return PeerMeasurement(
            address: address, identity: identity,
            bandwidthBytesPerSecond: bandwidth,
            roundTripSeconds: bestRoundTrip.isFinite ? bestRoundTrip : 0)
    }

    /// Asks `driver` (a running `ProbeServer`) to measure its own link to `target`.
    public static func measureRemote(
        from driver: PeerAddress, to target: PeerAddress,
        transport: Transport = TCPTransport(),
        options: Options = Options()
    ) throws -> PeerMeasurement {
        let channel = try transport.connect(to: driver, timeout: options.timeout)
        defer { channel.close() }
        let scratch = ScratchBuffer(capacity: 1 << 20)
        let request = ProbeRequest(
            address: target, streamBytes: options.streamBytes, chunkBytes: options.chunkBytes,
            pingIterations: options.pingIterations, warmupIterations: options.warmupIterations)
        let payload = try JSONEncoder().encode(request)
        scratch.ensure(WireHeader.byteCount + payload.count)
        payload.withUnsafeBytes { src in
            (scratch.base + WireHeader.byteCount).copyMemory(from: src.baseAddress!, byteCount: payload.count)
        }
        var header = WireHeader()
        header.tag = ProbeOpcode.probePeer.rawValue
        header.dataType = DataType.int8.wireCode
        header.elementCount = UInt32(payload.count)
        try channel.sendPreparedFrame(header, payloadBytes: payload.count, in: scratch)

        let reply = try expect(channel, .probePeerReply, scratch: scratch, maxPayloadBytes: 1 << 20)
        return try JSONDecoder().decode(
            PeerMeasurement.self, from: Data(bytes: scratch.base, count: Int(reply.payloadBytes)))
    }

    /// Builds a `Topology`: rank 0 is this machine, ranks 1…k are `peers`.
    ///
    /// Links from rank 0 are measured directly. With `options.pairwise`, each
    /// peer is also asked to measure every peer after it, so the planner sees
    /// the real fabric — that is what makes island detection possible.
    public static func buildTopology(
        peers: [PeerAddress],
        transport: Transport = TCPTransport(),
        options: Options = Options(),
        onProgress: ((String) -> Void)? = nil
    ) throws -> Topology {
        var nodes: [Topology.Node] = [NodeIdentity.local().node(id: 0)]
        var links: [Topology.Link] = []

        // Snapshot the local interfaces once: link kinds for edges leaving this
        // node come from the interface media, not from throughput guesswork.
        let interfaces = NetworkInterfaces.local()

        var measurements: [PeerMeasurement] = []
        for (index, peer) in peers.enumerated() {
            onProgress?("measuring 0 <-> \(index + 1) (\(peer))")
            let measurement = try measure(peer: peer, transport: transport, options: options)
            measurements.append(measurement)
            nodes.append(measurement.identity.node(id: index + 1))
            links.append(link(from: 0, to: index + 1, measurement: measurement, interfaces: interfaces))
        }

        if options.pairwise, peers.count > 1 {
            var peerOptions = options
            peerOptions.pairwise = false
            for i in 0..<peers.count {
                for j in (i + 1)..<peers.count {
                    onProgress?("measuring \(i + 1) <-> \(j + 1) (\(peers[i]) -> \(peers[j]))")
                    do {
                        let measurement = try measureRemote(
                            from: peers[i], to: peers[j], transport: transport, options: peerOptions)
                        links.append(link(from: i + 1, to: j + 1, measurement: measurement))
                    } catch {
                        // A peer that cannot reach another peer is information,
                        // not a failure: leave the edge out of the map.
                        onProgress?("  no route \(peers[i]) -> \(peers[j]): \(error)")
                    }
                }
            }
        }

        return Topology(nodes: nodes, links: links)
    }

    private static func link(
        from: Int, to: Int, measurement: PeerMeasurement, interfaces: [NetworkInterface]? = nil
    ) -> Topology.Link {
        Topology.Link(
            from: from, to: to,
            kind: classifyKind(address: measurement.address,
                               bandwidth: measurement.bandwidthBytesPerSecond,
                               interfaces: interfaces),
            measuredBandwidth: measurement.bandwidthBytesPerSecond,
            measuredLatency: measurement.roundTripSeconds)
    }

    /// Labels a link from the interface it actually leaves on, falling back to
    /// throughput inference when no local interface owns the peer's subnet.
    ///
    /// The interface knows things throughput cannot: a congested Thunderbolt
    /// bridge measuring 900 MB/s is still Thunderbolt, and a fast Wi-Fi link is
    /// never a cable. It cannot tell TB4 from TB5 — both are plain IP links to
    /// every API on the machine — so `NetworkInterface.linkKind` still consults
    /// the measurement for the generation.
    ///
    /// The label remains cosmetic either way: the planner reasons over the
    /// measured numbers and never over `Kind`.
    public static func classifyKind(
        address: PeerAddress, bandwidth: Double, interfaces: [NetworkInterface]? = nil
    ) -> Topology.Link.Kind {
        if let interface = NetworkInterfaces.interface(reaching: address, among: interfaces),
           let kind = interface.linkKind(measuredBandwidth: bandwidth) {
            return kind
        }
        return inferKind(address: address, bandwidth: bandwidth)
    }

    /// Classifies a link from what it actually achieved. Used when interface
    /// media is unavailable — a peer several routed hops away, or a transport
    /// whose `host` is not an IP literal at all.
    public static func inferKind(address: PeerAddress, bandwidth: Double) -> Topology.Link.Kind {
        if address.isLoopback { return .loopback }
        switch bandwidth {
        case 4.0e9...: return .thunderbolt5    // ~32+ Gb/s effective
        case 1.8e9..<4.0e9: return .thunderbolt4
        case 8.0e8..<1.8e9: return .usb4
        default: return .ethernet
        }
    }

    // MARK: framing helpers

    private static func request(
        _ channel: Channel, _ opcode: ProbeOpcode, payloadBytes: Int, scratch: ScratchBuffer
    ) throws {
        scratch.ensure(WireHeader.byteCount + payloadBytes)
        var header = WireHeader()
        header.tag = opcode.rawValue
        header.dataType = DataType.int8.wireCode
        header.elementCount = UInt32(payloadBytes)
        try channel.sendPreparedFrame(header, payloadBytes: payloadBytes, in: scratch)
    }

    @discardableResult
    private static func expect(
        _ channel: Channel, _ opcode: ProbeOpcode, scratch: ScratchBuffer, maxPayloadBytes: Int
    ) throws -> WireHeader {
        let header = try channel.receiveFrame(
            into: scratch, maxPayloadBytes: max(maxPayloadBytes, 1 << 20))
        if header.tag == ProbeOpcode.failure.rawValue {
            let text = String(
                decoding: Data(bytes: scratch.base, count: Int(header.payloadBytes)), as: UTF8.self)
            throw MCCLError.protocolViolation("peer reported: \(text)")
        }
        guard header.tag == opcode.rawValue else {
            throw MCCLError.protocolViolation("expected \(opcode), got opcode \(header.tag)")
        }
        return header
    }
}
