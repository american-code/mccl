import Foundation

/// The frame protocol mccl runs over an Apple RDMA queue pair.
///
/// ---------------------------------------------------------------------------
/// Why the framing looks like this
/// ---------------------------------------------------------------------------
///
/// One line of TN3205 determines the entire design:
///
/// > Thunderbolt devices require a receiver and sender to post messages which
/// > are the same number of frames long. For example, if the sender sends a 16K
/// > message and the receiver posts a 4K or 32K receive buffer the receive
/// > operation will fail.
///
/// A receive buffer has to be posted *before* the message arrives, so the
/// receiver must know the size of a message it has not seen yet. There are only
/// two ways out of that, and only one of them survives contact:
///
///  * **Announce each size before sending it.** Every payload becomes two
///    messages — a fixed-size length announcement, then the payload — which
///    doubles the message count and puts a full round trip in front of every
///    write. On a fabric bought for its latency that is the wrong trade, and it
///    does not even remove the problem: the announcement itself still has to be
///    a size both sides agreed on in advance.
///
///  * **Fix the size for the life of the connection.** Every message is exactly
///    `slotBytes`, so the receiver can keep receives posted permanently and the
///    same-size rule is satisfied by construction rather than by agreement.
///
/// mccl takes the second. `slotBytes` is negotiated once, during the TCP
/// bootstrap that already exchanges queue-pair metadata, and never changes.
///
/// The cost is real and worth naming: a write smaller than a slot still occupies
/// a whole slot on the wire. mccl's traffic makes that cheap — `Channel.sendFrame`
/// emits a 24-byte header and its payload as a *single* write, so the padding is
/// paid once per collective step rather than once per header. A 4 MiB ring step
/// at the default 64 KiB slot is 64 full slots and one partial: under 1% waste.
/// A stream of tiny writes would be far worse, and this transport is not built
/// for one.
///
/// ---------------------------------------------------------------------------
/// Layout
/// ---------------------------------------------------------------------------
///
/// Each slot is `slotBytes` on the wire: a 16-byte header followed by up to
/// `slotBytes - 16` bytes of payload, with the remainder undefined (it is never
/// read, and is deliberately not zeroed — it would be a full memset per slot on
/// the hot path, to hide nothing, since the receiver honours `payloadBytes`).
///
/// ```
///  0..3   magic 'MCR1'
///  4..7   sequence   — per-direction, starts at 0, wraps at 2^32
///  8..11  payloadBytes
/// 12..15  flags
/// ```
struct RDMASlotHeader: Equatable {
    static let magic: UInt32 = 0x3152_434D   // 'M','C','R','1' little-endian
    static let byteCount = 16

    /// Set on the last slot a sender emits, so a clean close is distinguishable
    /// from a cable pulled out mid-stream.
    static let flagClosing: UInt32 = 1 << 0

    var sequence: UInt32 = 0
    var payloadBytes: UInt32 = 0
    var flags: UInt32 = 0

    var isClosing: Bool { flags & RDMASlotHeader.flagClosing != 0 }

    func encode(into p: UnsafeMutableRawPointer) {
        p.storeBytes(of: RDMASlotHeader.magic.littleEndian, toByteOffset: 0, as: UInt32.self)
        p.storeBytes(of: sequence.littleEndian, toByteOffset: 4, as: UInt32.self)
        p.storeBytes(of: payloadBytes.littleEndian, toByteOffset: 8, as: UInt32.self)
        p.storeBytes(of: flags.littleEndian, toByteOffset: 12, as: UInt32.self)
    }

    static func decode(from p: UnsafeRawPointer, capacity: Int) throws -> RDMASlotHeader {
        let magic = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
        guard magic == RDMASlotHeader.magic else {
            // On a UC queue pair this is the shape a lost or reordered slot
            // takes when it lands in a buffer that still holds an older slot.
            throw MCCLError.protocolViolation(
                String(format: "bad RDMA slot magic 0x%08x — the slot was not written by mccl, "
                       + "or a previous slot was lost", magic))
        }
        var header = RDMASlotHeader()
        header.sequence = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        header.payloadBytes = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
        header.flags = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
        guard Int(header.payloadBytes) <= capacity else {
            throw MCCLError.protocolViolation(
                "RDMA slot claims \(header.payloadBytes) payload bytes, capacity is \(capacity)")
        }
        return header
    }
}

/// The one size every message on a connection has, and the rules it obeys.
///
/// Validated at connection setup rather than trusted, because every constraint
/// here is one the hardware enforces by failing a transfer rather than by
/// returning an error anybody can read.
public struct RDMASlotGeometry: Equatable, Sendable {
    /// Bytes on the wire per message, header included.
    public let slotBytes: Int
    /// How many slots the registered ring holds, per direction.
    public let slotCount: Int

    /// Usable payload per slot.
    public var payloadCapacity: Int { slotBytes - RDMASlotHeader.byteCount }

    /// Slots are sized in whole Thunderbolt frames, which is also how TN3205
    /// describes queue depth ("Queues are sized in units of 4 KB frames").
    public var frameCount: Int { slotBytes / RDMASpec.frameBytes }

    public init(slotBytes: Int = RDMASlotGeometry.defaultSlotBytes,
                slotCount: Int = RDMASlotGeometry.defaultSlotCount) throws {
        guard slotBytes > RDMASlotHeader.byteCount else {
            throw MCCLError.invalidArgument("RDMA slot of \(slotBytes) bytes has no room for a header")
        }
        guard slotBytes % RDMASpec.frameBytes == 0 else {
            throw MCCLError.invalidArgument(
                "RDMA slot size \(slotBytes) is not a multiple of the \(RDMASpec.frameBytes)-byte "
                + "Thunderbolt frame; the same-size rule is expressed in whole frames")
        }
        guard slotBytes <= RDMASpec.maxMessageBytes else {
            throw MCCLError.invalidArgument(
                "RDMA slot size \(slotBytes) exceeds the \(RDMASpec.maxMessageBytes)-byte maximum "
                + "message (4095 frames of \(RDMASpec.frameBytes))")
        }
        // Both directions' work requests share one queue-pair depth budget.
        guard slotCount > 0, slotCount * 2 <= RDMASpec.maxWorkRequests else {
            throw MCCLError.invalidArgument(
                "an RDMA ring of \(slotCount) slots per direction needs \(slotCount * 2) work "
                + "requests, above the \(RDMASpec.maxWorkRequests) TN3205 allows")
        }
        self.slotBytes = slotBytes
        self.slotCount = slotCount
    }

    /// 64 KiB — sixteen Thunderbolt frames.
    ///
    /// Chosen to sit well above the 24-byte `WireHeader` that precedes every
    /// mccl payload and well below the 16 MB message ceiling, so a collective's
    /// large writes chunk into a handful of slots and its small ones waste a
    /// bounded amount. Not tuned against hardware, because there is none to tune
    /// against; the first TB5 user should sweep it (see docs/RDMA.md).
    public static let defaultSlotBytes = 64 * 1024

    /// 32 slots per direction — 2 MiB of registered memory per connection, and
    /// 64 of the 4095 available work requests.
    public static let defaultSlotCount = 32

    /// Registered bytes per direction.
    public var ringBytes: Int { slotBytes * slotCount }
}

/// The receive-side gap check.
///
/// TN3205 is explicit that a UC queue pair gives no delivery guarantee:
///
/// > Completion of a send operation indicates that the peer has posted
/// > sufficient receive requests ... and that the Thunderbolt hardware has sent
/// > the buffers to the peer. Completion does not indicate the receiver has
/// > successfully received the sent data or that data was not corrupted in
/// > flight because Thunderbolt does not perform Acks in hardware.
///
/// A Thunderbolt link is reliable in practice — it is a short, point-to-point,
/// CRC-protected cable, which is why Apple considers UC sufficient here — but
/// "reliable in practice" is not a guarantee a collectives library may quietly
/// depend on. A dropped slot in an all-reduce does not announce itself: it
/// becomes a plausible-looking tensor with a hole in it, which then trains a
/// model. That failure is silent, and silent numerical corruption is the worst
/// outcome this library can produce.
///
/// So every slot carries a sequence number and the receiver checks it. A gap is
/// a hard error that fails the collective, and the error says what it means.
/// mccl deliberately does **not** attempt recovery: a retransmit protocol over
/// UC would be a reimplementation of the reliability TCP already provides, and
/// if this fabric turns out to need one then TCP is the right transport for that
/// cluster. Detecting is honest; hiding would not be.
struct RDMASequenceChecker {
    private(set) var expected: UInt32 = 0

    /// Accepts `sequence` if it is the next one, otherwise throws.
    ///
    /// Wrapping at 2^32 is handled by the unchecked increment: after 4.29e9
    /// slots (256 TiB at the default slot size) the counter returns to 0 and the
    /// expectation follows it.
    mutating func accept(_ sequence: UInt32) throws {
        guard sequence == expected else {
            throw MCCLError.rdmaSequenceGap(expected: expected, received: sequence)
        }
        expected = expected &+ 1
    }

    /// The next sequence a sender should stamp, then advances.
    mutating func next() -> UInt32 {
        defer { expected = expected &+ 1 }
        return expected
    }
}

/// How a byte stream is cut into slots.
///
/// Pure arithmetic, split out because it is the part worth testing exhaustively:
/// the boundaries (an exactly-full final slot, a zero-byte write, a write one
/// byte over a slot) are where a framing bug would live.
enum RDMAChunking {
    /// The slot payload sizes a write of `byteCount` bytes becomes.
    static func chunks(of byteCount: Int, capacity: Int) -> [Int] {
        precondition(capacity > 0)
        guard byteCount > 0 else { return [] }
        var remaining = byteCount
        var result: [Int] = []
        result.reserveCapacity((byteCount + capacity - 1) / capacity)
        while remaining > 0 {
            let take = min(capacity, remaining)
            result.append(take)
            remaining -= take
        }
        return result
    }

    /// How many slots a write of `byteCount` bytes occupies.
    static func slotCount(for byteCount: Int, capacity: Int) -> Int {
        precondition(capacity > 0)
        return (byteCount + capacity - 1) / capacity
    }
}
