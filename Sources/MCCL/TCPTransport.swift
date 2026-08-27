import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// TCP over POSIX sockets. No external package dependencies — this is the
/// portable baseline transport (Ethernet, and IP-over-Thunderbolt bridges,
/// which present as ordinary interfaces).
public final class TCPTransport: Transport {
    public let name = "tcp"

    /// Socket buffer size requested for streaming transfers. 4 MiB keeps a
    /// 10GbE / TB link full without the sender stalling between ring steps.
    public var socketBufferBytes: Int

    public init(socketBufferBytes: Int = 4 << 20) {
        self.socketBufferBytes = socketBufferBytes
    }

    public func listen(host: String, port: Int) throws -> Listener {
        try TCPListener(host: host, port: port, socketBufferBytes: socketBufferBytes)
    }

    public func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel {
        let deadline = Date().addingTimeInterval(timeout)
        var lastErrno: Int32 = 0
        repeat {
            let candidates = try SocketSupport.resolve(host: address.host, port: address.port, passive: false)
            for candidate in candidates {
                let fd = socket(candidate.family, SOCK_STREAM, IPPROTO_TCP)
                if fd < 0 { lastErrno = errno; continue }
                SocketSupport.tune(fd: fd, bufferBytes: socketBufferBytes)
                var addr = candidate.storage
                let len = candidate.length
                let rc = withUnsafePointer(to: &addr) { ptr -> Int32 in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                        Darwin.connect(fd, sa, socklen_t(len))
                    }
                }
                if rc == 0 {
                    return TCPChannel(fd: fd, peer: address.description)
                }
                lastErrno = errno
                _ = Darwin.close(fd)
            }
            // The peer's listener may not be up yet (cluster bring-up races).
            usleep(20_000)
        } while Date() < deadline
        throw MCCLError.socketFailure("connect to \(address)", errno: lastErrno)
    }
}

// MARK: - Listener

final class TCPListener: Listener {
    private let fd: Int32
    private let socketBufferBytes: Int
    private var closed = false
    private let lock = NSLock()
    let address: PeerAddress

    init(host: String, port: Int, socketBufferBytes: Int) throws {
        self.socketBufferBytes = socketBufferBytes
        let candidates = try SocketSupport.resolve(host: host, port: port, passive: true)
        guard let candidate = candidates.first else {
            throw MCCLError.invalidArgument("cannot resolve \(host):\(port)")
        }
        let fd = socket(candidate.family, SOCK_STREAM, IPPROTO_TCP)
        guard fd >= 0 else { throw MCCLError.socketFailure("socket()", errno: errno) }

        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = candidate.storage
        let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(fd, sa, socklen_t(candidate.length))
            }
        }
        guard bindResult == 0 else {
            let e = errno
            _ = Darwin.close(fd)
            throw MCCLError.socketFailure("bind(\(host):\(port))", errno: e)
        }
        guard Darwin.listen(fd, 64) == 0 else {
            let e = errno
            _ = Darwin.close(fd)
            throw MCCLError.socketFailure("listen()", errno: e)
        }

        self.fd = fd
        self.address = PeerAddress(host: host, port: try SocketSupport.boundPort(fd: fd))
    }

    func accept(timeout: TimeInterval?) throws -> Channel {
        if let timeout {
            var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ms = Int32(max(0, timeout * 1000))
            let rc = poll(&pfd, 1, ms)
            if rc == 0 { throw MCCLError.timedOut("accept on \(address)") }
            if rc < 0 { throw MCCLError.socketFailure("poll()", errno: errno) }
        }
        while true {
            let client = Darwin.accept(fd, nil, nil)
            if client >= 0 {
                SocketSupport.tune(fd: client, bufferBytes: socketBufferBytes)
                return TCPChannel(fd: client, peer: SocketSupport.peerName(fd: client))
            }
            if errno == EINTR { continue }
            throw MCCLError.socketFailure("accept()", errno: errno)
        }
    }

    func close() {
        lock.lock(); defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        _ = Darwin.close(fd)
    }

    deinit { close() }
}

// MARK: - Channel

final class TCPChannel: Channel {
    private let fd: Int32
    private var closed = false
    private let closeLock = NSLock()

    let sendQueue: DispatchQueue
    let receiveQueue: DispatchQueue
    let peerDescription: String
    let remoteAddress: PeerAddress?

    init(fd: Int32, peer: String) {
        self.fd = fd
        self.peerDescription = peer
        self.remoteAddress = SocketSupport.peerAddress(fd: fd)
        self.sendQueue = DispatchQueue(label: "mccl.tcp.tx.\(fd)")
        self.receiveQueue = DispatchQueue(label: "mccl.tcp.rx.\(fd)")
    }

    func sendBytes(_ buffer: UnsafeRawBufferPointer) throws {
        guard let base = buffer.baseAddress, buffer.count > 0 else { return }
        var written = 0
        while written < buffer.count {
            let n = Darwin.send(fd, base + written, buffer.count - written, 0)
            if n > 0 { written += n; continue }
            if n == 0 { throw MCCLError.connectionClosed }
            if errno == EINTR { continue }
            throw MCCLError.socketFailure("send()", errno: errno)
        }
    }

    func receiveBytes(into buffer: UnsafeMutableRawBufferPointer) throws {
        guard let base = buffer.baseAddress, buffer.count > 0 else { return }
        var read = 0
        while read < buffer.count {
            let n = Darwin.recv(fd, base + read, buffer.count - read, 0)
            if n > 0 { read += n; continue }
            if n == 0 { throw MCCLError.connectionClosed }
            if errno == EINTR { continue }
            throw MCCLError.socketFailure("recv()", errno: errno)
        }
    }

    func close() {
        closeLock.lock(); defer { closeLock.unlock() }
        guard !closed else { return }
        closed = true
        _ = Darwin.shutdown(fd, SHUT_RDWR)
        _ = Darwin.close(fd)
    }

    deinit { close() }
}

// MARK: - POSIX helpers

enum SocketSupport {
    struct Candidate {
        var family: Int32
        var storage: sockaddr_storage
        var length: Int
    }

    static func resolve(host: String, port: Int, passive: Bool) throws -> [Candidate] {
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        hints.ai_protocol = IPPROTO_TCP
        hints.ai_flags = passive ? AI_PASSIVE : 0

        var result: UnsafeMutablePointer<addrinfo>?
        let rc = getaddrinfo(host, String(port), &hints, &result)
        guard rc == 0, let head = result else {
            throw MCCLError.invalidArgument("getaddrinfo(\(host):\(port)): \(String(cString: gai_strerror(rc)))")
        }
        defer { freeaddrinfo(head) }

        var candidates: [Candidate] = []
        var cursor: UnsafeMutablePointer<addrinfo>? = head
        while let node = cursor {
            var storage = sockaddr_storage()
            let len = Int(node.pointee.ai_addrlen)
            _ = withUnsafeMutableBytes(of: &storage) { dst in
                UnsafeRawBufferPointer(start: node.pointee.ai_addr, count: len)
                    .copyBytes(to: UnsafeMutableRawBufferPointer(rebasing: dst[0..<len]))
            }
            candidates.append(Candidate(family: node.pointee.ai_family, storage: storage, length: len))
            cursor = node.pointee.ai_next
        }
        // Prefer IPv4 — a mixed cluster is far likelier to agree on it.
        candidates.sort { a, b in (a.family == AF_INET ? 0 : 1) < (b.family == AF_INET ? 0 : 1) }
        return candidates
    }

    static func boundPort(fd: Int32) throws -> Int {
        var storage = sockaddr_storage()
        var len = socklen_t(MemoryLayout<sockaddr_storage>.size)
        let rc = withUnsafeMutablePointer(to: &storage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                getsockname(fd, sa, &len)
            }
        }
        guard rc == 0 else { throw MCCLError.socketFailure("getsockname()", errno: errno) }
        return withUnsafeBytes(of: &storage) { raw -> Int in
            let family = raw.loadUnaligned(fromByteOffset: 1, as: UInt8.self)
            // sin_port and sin6_port sit at the same offset (2).
            _ = family
            let be = raw.loadUnaligned(fromByteOffset: 2, as: UInt16.self)
            return Int(UInt16(bigEndian: be))
        }
    }

    static func peerName(fd: Int32) -> String {
        var storage = sockaddr_storage()
        var len = socklen_t(MemoryLayout<sockaddr_storage>.size)
        let rc = withUnsafeMutablePointer(to: &storage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                getpeername(fd, sa, &len)
            }
        }
        guard rc == 0 else { return "fd\(fd)" }
        var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        var portBuf = [CChar](repeating: 0, count: Int(NI_MAXSERV))
        let ok = withUnsafePointer(to: &storage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                getnameinfo(sa, len, &hostBuf, socklen_t(hostBuf.count), &portBuf, socklen_t(portBuf.count),
                            NI_NUMERICHOST | NI_NUMERICSERV)
            }
        }
        guard ok == 0 else { return "fd\(fd)" }
        return "\(String(cString: hostBuf)):\(String(cString: portBuf))"
    }

    /// The connected peer's address, numeric. Used to check a bootstrap
    /// announcement against where it actually came from.
    static func peerAddress(fd: Int32) -> PeerAddress? {
        var storage = sockaddr_storage()
        var len = socklen_t(MemoryLayout<sockaddr_storage>.size)
        let rc = withUnsafeMutablePointer(to: &storage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                getpeername(fd, sa, &len)
            }
        }
        guard rc == 0 else { return nil }
        var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        var portBuf = [CChar](repeating: 0, count: Int(NI_MAXSERV))
        let ok = withUnsafePointer(to: &storage) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                getnameinfo(sa, len, &hostBuf, socklen_t(hostBuf.count), &portBuf, socklen_t(portBuf.count),
                            NI_NUMERICHOST | NI_NUMERICSERV)
            }
        }
        guard ok == 0 else { return nil }
        // "fe80::1%en0" -> "fe80::1"; the scope is local to the observer and
        // would never match an advertised literal.
        let host = String(cString: hostBuf).split(separator: "%").first.map(String.init) ?? ""
        guard !host.isEmpty else { return nil }
        return PeerAddress(host: host, port: Int(String(cString: portBuf)) ?? 0)
    }

    static func tune(fd: Int32, bufferBytes: Int) {
        var yes: Int32 = 1
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
        var size = Int32(bufferBytes)
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, socklen_t(MemoryLayout<Int32>.size))
    }
}
