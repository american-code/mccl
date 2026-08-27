import Foundation

/// In-process transport. Same `Transport` surface as TCP, but the bytes never
/// leave the address space — used to test the collectives and the wire codecs
/// without binding sockets.
public final class LoopbackTransport: Transport {
    public let name = "loopback"

    /// Registry of live in-process listeners, keyed by their synthetic address.
    private static let registryLock = NSLock()
    nonisolated(unsafe) private static var registry: [PeerAddress: LoopbackListener] = [:]
    nonisolated(unsafe) private static var nextPort = 1

    public init() {}

    public func listen(host: String, port: Int) throws -> Listener {
        LoopbackTransport.registryLock.lock()
        defer { LoopbackTransport.registryLock.unlock() }
        var resolvedPort = port
        if resolvedPort == 0 {
            resolvedPort = LoopbackTransport.nextPort
            LoopbackTransport.nextPort += 1
        }
        let address = PeerAddress(host: host.isEmpty ? "loopback" : host, port: resolvedPort)
        if LoopbackTransport.registry[address] != nil {
            throw MCCLError.invalidArgument("loopback address \(address) already in use")
        }
        let listener = LoopbackListener(address: address)
        LoopbackTransport.registry[address] = listener
        return listener
    }

    public func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            LoopbackTransport.registryLock.lock()
            let listener = LoopbackTransport.registry[address]
            LoopbackTransport.registryLock.unlock()
            if let listener {
                let a = MemoryPipe()
                let b = MemoryPipe()
                // client writes to `a`, reads from `b`; server the mirror image.
                let client = LoopbackChannel(outbound: a, inbound: b, peer: address.description)
                let server = LoopbackChannel(outbound: b, inbound: a, peer: "client->\(address)")
                listener.enqueue(server)
                return client
            }
            if Date() >= deadline { throw MCCLError.timedOut("loopback connect to \(address)") }
            usleep(2_000)
        }
    }

    static func unregister(_ address: PeerAddress) {
        registryLock.lock()
        registry[address] = nil
        registryLock.unlock()
    }
}

final class LoopbackListener: Listener {
    let address: PeerAddress
    private let condition = NSCondition()
    private var pending: [LoopbackChannel] = []
    private var closed = false

    init(address: PeerAddress) { self.address = address }

    func enqueue(_ channel: LoopbackChannel) {
        condition.lock()
        pending.append(channel)
        condition.signal()
        condition.unlock()
    }

    func accept(timeout: TimeInterval?) throws -> Channel {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout ?? 3600)
        while pending.isEmpty {
            if closed { throw MCCLError.connectionClosed }
            if !condition.wait(until: deadline) {
                throw MCCLError.timedOut("loopback accept on \(address)")
            }
        }
        return pending.removeFirst()
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
        LoopbackTransport.unregister(address)
    }

    deinit { close() }
}

/// Unbounded FIFO byte queue with blocking reads. Unbounded is deliberate: a
/// bounded in-process pipe would need the same full-duplex care as a socket,
/// and here there is no wire to congest.
final class MemoryPipe {
    private let condition = NSCondition()
    private var buffer: [UInt8] = []
    private var readCursor = 0
    private var closed = false

    func write(_ bytes: UnsafeRawBufferPointer) throws {
        condition.lock()
        defer { condition.unlock() }
        if closed { throw MCCLError.connectionClosed }
        buffer.append(contentsOf: bytes)
        condition.broadcast()
    }

    func read(into destination: UnsafeMutableRawBufferPointer) throws {
        guard destination.count > 0 else { return }
        var filled = 0
        while filled < destination.count {
            condition.lock()
            while buffer.count - readCursor == 0 {
                if closed { condition.unlock(); throw MCCLError.connectionClosed }
                condition.wait()
            }
            let available = buffer.count - readCursor
            let take = min(available, destination.count - filled)
            buffer.withUnsafeBytes { src in
                let slice = UnsafeRawBufferPointer(rebasing: src[readCursor..<(readCursor + take)])
                UnsafeMutableRawBufferPointer(rebasing: destination[filled..<(filled + take)]).copyMemory(from: slice)
            }
            readCursor += take
            if readCursor == buffer.count {
                buffer.removeAll(keepingCapacity: true)
                readCursor = 0
            }
            condition.unlock()
            filled += take
        }
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
    }
}

final class LoopbackChannel: Channel {
    private let outbound: MemoryPipe
    private let inbound: MemoryPipe
    let sendQueue: DispatchQueue
    let receiveQueue: DispatchQueue
    let peerDescription: String

    init(outbound: MemoryPipe, inbound: MemoryPipe, peer: String) {
        self.outbound = outbound
        self.inbound = inbound
        self.peerDescription = peer
        let id = UUID().uuidString.prefix(8)
        self.sendQueue = DispatchQueue(label: "mccl.loopback.tx.\(id)")
        self.receiveQueue = DispatchQueue(label: "mccl.loopback.rx.\(id)")
    }

    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws { try outbound.write(buffer) }
    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws { try inbound.read(into: buffer) }
    func close() {
        outbound.close()
        inbound.close()
    }
}
