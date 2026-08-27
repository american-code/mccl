import Foundation

/// Per-operation wire state: the codec plus the scratch buffers a rank needs
/// to have a frame in flight in each direction at once.
final class WireContext: @unchecked Sendable {
    let dataType: DataType
    let codec: WireCodec

    /// Outgoing frame: header at offset 0, encoded payload at `byteCount`.
    let tx = ScratchBuffer(capacity: WireHeader.byteCount + 4096)
    /// Incoming payload (header is read separately).
    let rx = ScratchBuffer(capacity: 4096)
    /// Decoded elements, in the caller's dtype, before reduction.
    let staging = ScratchBuffer(capacity: 4096)

    private(set) var rxCapacity = 4096

    init(dataType: DataType, compression: WireCompression) throws {
        self.dataType = dataType
        self.codec = try WireCodec(dataType: dataType, compression: compression)
    }

    /// Sizes every buffer for the largest single transfer this operation makes.
    func reserve(maxElements: Int) {
        let encoded = codec.encodedByteCount(elementCount: maxElements)
        let raw = maxElements * dataType.byteWidth
        rxCapacity = max(encoded, raw)
        tx.ensure(WireHeader.byteCount + encoded)
        rx.ensure(rxCapacity)
        staging.ensure(raw)
    }

    func makeHeader(elementCount: Int, tag: UInt32) -> WireHeader {
        var header = WireHeader()
        header.codec = codec.codec.rawValue
        header.dataType = dataType.wireCode
        header.elementCount = UInt32(elementCount)
        header.blockSize = UInt32(codec.blockSize)
        header.tag = tag
        return header
    }

    func decode(_ header: WireHeader, elementCount: Int, into destination: UnsafeMutableRawPointer) throws {
        guard Int(header.elementCount) == elementCount else {
            throw MCCLError.protocolViolation(
                "peer sent \(header.elementCount) elements, expected \(elementCount)")
        }
        guard let id = WireCodecID(rawValue: header.codec) else {
            throw MCCLError.protocolViolation("unknown wire codec \(header.codec)")
        }
        let peerType = try DataType.from(wireCode: header.dataType)
        guard peerType == dataType else {
            throw MCCLError.protocolViolation("peer sent \(peerType), expected \(dataType)")
        }
        try WireCodec.decode(
            payload: rx.base, payloadBytes: Int(header.payloadBytes),
            elementCount: elementCount, codec: id, blockSize: Int(header.blockSize),
            dataType: dataType, into: destination)
    }
}

/// Escaping-closure-safe slot for the header produced by a concurrent receive.
private final class HeaderBox: @unchecked Sendable {
    var value: WireHeader?
}

// MARK: - Point-to-point primitives

extension Communicator {
    func sendChunk(
        on channel: Channel, from source: UnsafeRawPointer, count: Int, tag: UInt32, ctx: WireContext
    ) throws {
        let bytes = ctx.codec.encode(from: source, elementCount: count, into: ctx.tx.base + WireHeader.byteCount)
        try channel.sendPreparedFrame(ctx.makeHeader(elementCount: count, tag: tag), payloadBytes: bytes, in: ctx.tx)
    }

    func receiveChunk(
        on channel: Channel, into destination: UnsafeMutableRawPointer, count: Int, ctx: WireContext
    ) throws {
        let header = try channel.receiveFrame(into: ctx.rx, maxPayloadBytes: ctx.rxCapacity)
        try ctx.decode(header, elementCount: count, into: destination)
    }

    /// One ring step: push a chunk to the next rank while the previous rank's
    /// chunk lands. The overlap is the whole point of the ring — a serialised
    /// send-then-receive would halve the achievable bandwidth.
    func exchangeChunk(
        sendOn sendChannel: Channel, from source: UnsafeRawPointer, sendCount: Int,
        receiveOn receiveChannel: Channel, into destination: UnsafeMutableRawPointer, receiveCount: Int,
        tag: UInt32, ctx: WireContext
    ) throws {
        let bytes = ctx.codec.encode(from: source, elementCount: sendCount, into: ctx.tx.base + WireHeader.byteCount)
        let header = ctx.makeHeader(elementCount: sendCount, tag: tag)
        let box = HeaderBox()
        let capacity = ctx.rxCapacity

        try runConcurrently(
            { try sendChannel.sendPreparedFrame(header, payloadBytes: bytes, in: ctx.tx) },
            on: sendChannel.sendQueue,
            { box.value = try receiveChannel.receiveFrame(into: ctx.rx, maxPayloadBytes: capacity) },
            on: receiveChannel.receiveQueue
        )

        guard let received = box.value else {
            throw MCCLError.protocolViolation("ring step \(tag) produced no inbound frame")
        }
        try ctx.decode(received, elementCount: receiveCount, into: destination)
    }
}

// MARK: - Ring

extension Communicator {
    /// Reduce-scatter half of a ring all-reduce, over an arbitrary ordered
    /// subgroup. `chunkShift` relabels the chunk indices so the caller can
    /// choose which chunk each rank ends up owning.
    ///
    /// After `n-1` steps the rank at ring position `p` holds the fully reduced
    /// chunk `(p + 1 + chunkShift) mod n`.
    @discardableResult
    func ringReduceScatterPhase(
        group: [Int], base: UnsafeMutableRawPointer, layout: ChunkLayout,
        dataType: DataType, op: ReduceOp, chunkShift: Int, ctx: WireContext, fabric: Fabric
    ) throws -> Int {
        let n = group.count
        guard n > 1, let position = group.firstIndex(of: rank) else { return 0 }
        let width = dataType.byteWidth
        let next = try fabric.channel(to: group[(position + 1) % n])
        let previous = try fabric.channel(to: group[(position - 1 + n) % n])

        for step in 0..<(n - 1) {
            let sendIndex = (position - step + chunkShift + 2 * n) % n
            let recvIndex = (position - step - 1 + chunkShift + 2 * n) % n
            try exchangeChunk(
                sendOn: next, from: base + layout.offsets[sendIndex] * width, sendCount: layout.counts[sendIndex],
                receiveOn: previous, into: ctx.staging.base, receiveCount: layout.counts[recvIndex],
                tag: UInt32(step), ctx: ctx)
            Kernels.reduce(
                into: base + layout.offsets[recvIndex] * width, from: ctx.staging.base,
                count: layout.counts[recvIndex], dataType: dataType, op: op)
        }
        return (position + 1 + chunkShift + 2 * n) % n
    }

    /// All-gather half of a ring all-reduce: rotate the owned chunks all the
    /// way round so every rank ends with the whole buffer.
    func ringAllGatherPhase(
        group: [Int], base: UnsafeMutableRawPointer, layout: ChunkLayout,
        dataType: DataType, chunkShift: Int, ctx: WireContext, fabric: Fabric
    ) throws {
        let n = group.count
        guard n > 1, let position = group.firstIndex(of: rank) else { return }
        let width = dataType.byteWidth
        let next = try fabric.channel(to: group[(position + 1) % n])
        let previous = try fabric.channel(to: group[(position - 1 + n) % n])

        for step in 0..<(n - 1) {
            let sendIndex = (position - step + 1 + chunkShift + 2 * n) % n
            let recvIndex = (position - step + chunkShift + 2 * n) % n
            try exchangeChunk(
                sendOn: next, from: base + layout.offsets[sendIndex] * width, sendCount: layout.counts[sendIndex],
                receiveOn: previous, into: base + layout.offsets[recvIndex] * width,
                receiveCount: layout.counts[recvIndex],
                tag: UInt32(1000 + step), ctx: ctx)
        }
    }

    func ringAllReduce(
        group: [Int], buffer: UnsafeMutableRawBufferPointer, count: Int,
        dataType: DataType, op: ReduceOp, ctx: WireContext, fabric: Fabric
    ) throws {
        let n = group.count
        guard n > 1, group.contains(rank) else { return }
        let layout = ChunkLayout(count: count, parts: n)
        ctx.reserve(maxElements: layout.maxCount)
        let base = buffer.baseAddress!
        try ringReduceScatterPhase(
            group: group, base: base, layout: layout, dataType: dataType,
            op: op, chunkShift: 0, ctx: ctx, fabric: fabric)
        try ringAllGatherPhase(
            group: group, base: base, layout: layout, dataType: dataType,
            chunkShift: 0, ctx: ctx, fabric: fabric)
    }
}

// MARK: - Tree

extension Communicator {
    /// Reduce up a tree onto `root`. Each rank waits for its children, folds
    /// them in, then hands one buffer to its parent: ⌈log2 n⌉ dependent hops
    /// instead of the ring's 2(n-1).
    func treeReduce(
        children: [Int: [Int]], root: Int, buffer: UnsafeMutableRawBufferPointer, count: Int,
        dataType: DataType, op: ReduceOp, ctx: WireContext, fabric: Fabric
    ) throws {
        let parents = TopologyPlanner.parents(of: children)
        let kids = children[rank] ?? []
        guard rank == root || parents[rank] != nil || !kids.isEmpty else { return }
        ctx.reserve(maxElements: count)

        for child in kids {
            let channel = try fabric.channel(to: child)
            try receiveChunk(on: channel, into: ctx.staging.base, count: count, ctx: ctx)
            Kernels.reduce(into: buffer.baseAddress!, from: ctx.staging.base,
                           count: count, dataType: dataType, op: op)
        }
        if rank != root, let parent = parents[rank] {
            try sendChunk(on: try fabric.channel(to: parent), from: buffer.baseAddress!,
                          count: count, tag: 2, ctx: ctx)
        }
    }

    /// Broadcast down the same tree.
    func treeBroadcast(
        children: [Int: [Int]], root: Int, buffer: UnsafeMutableRawBufferPointer, count: Int,
        dataType: DataType, ctx: WireContext, fabric: Fabric
    ) throws {
        let parents = TopologyPlanner.parents(of: children)
        let kids = children[rank] ?? []
        guard rank == root || parents[rank] != nil || !kids.isEmpty else { return }
        ctx.reserve(maxElements: count)

        if rank != root, let parent = parents[rank] {
            try receiveChunk(on: try fabric.channel(to: parent), into: buffer.baseAddress!, count: count, ctx: ctx)
        }
        for child in kids {
            try sendChunk(on: try fabric.channel(to: child), from: buffer.baseAddress!,
                          count: count, tag: 3, ctx: ctx)
        }
    }

    /// Chain broadcast along a ring order — the bandwidth-shaped alternative to
    /// a tree, used when the planner picked `.ring`.
    func chainBroadcast(
        group: [Int], root: Int, buffer: UnsafeMutableRawBufferPointer, count: Int,
        dataType: DataType, ctx: WireContext, fabric: Fabric
    ) throws {
        let n = group.count
        guard n > 1, let position = group.firstIndex(of: rank),
              let rootPosition = group.firstIndex(of: root) else { return }
        ctx.reserve(maxElements: count)
        let hop = (position - rootPosition + n) % n
        if hop > 0 {
            try receiveChunk(on: try fabric.channel(to: group[(position - 1 + n) % n]),
                             into: buffer.baseAddress!, count: count, ctx: ctx)
        }
        if hop < n - 1 {
            try sendChunk(on: try fabric.channel(to: group[(position + 1) % n]),
                          from: buffer.baseAddress!, count: count, tag: 4, ctx: ctx)
        }
    }
}

// MARK: - Public collective bodies

extension Communicator {
    var defaultRingOrder: [Int] {
        let order = TopologyPlanner.ringOrder(for: topology).filter { $0 >= 0 && $0 < worldSize }
        return Set(order).count == worldSize ? order : Array(0..<worldSize)
    }

    func ringOrder(from plan: CollectivePlan) -> [Int] {
        switch plan {
        case .ring(let order): return order
        case .hierarchical(let islands, _): return islands.flatMap { $0 }
        case .tree: return defaultRingOrder
        }
    }

    func requireFabric() throws -> Fabric {
        guard let fabric else { throw MCCLError.noFabric }
        return fabric
    }

    func checkCapacity(_ bytes: Int, _ have: Int) throws {
        guard have >= bytes else { throw MCCLError.bufferTooSmall(need: bytes, have: have) }
    }

    // MARK: all-reduce

    func allReduceSync(
        _ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType,
        op: ReduceOp, compression: WireCompression, stream: StreamID = .default
    ) throws {
        guard count >= 0 else { throw MCCLError.invalidArgument("count must be >= 0") }
        let width = dataType.byteWidth
        try checkCapacity(count * width, buffer.count)
        guard worldSize > 1 else {
            if op == .avg, count > 0 { Kernels.scale(buffer.baseAddress!, count: count, dataType: dataType, by: 1) }
            return
        }
        guard count > 0 else { return }

        // Top-k is not a per-hop codec — it replaces the algorithm. See
        // `topKAllReduceSync`.
        if case .topK(let fraction) = compression {
            return try topKAllReduceSync(buffer, count: count, dataType: dataType,
                                         op: op, fraction: fraction, stream: stream)
        }

        let fabric = try requireFabric()
        let ctx = try WireContext(dataType: dataType, compression: compression)
        let reduction = op.wireReduction

        switch plan(messageBytes: count * width) {
        case .ring(let order):
            try ringAllReduce(group: order, buffer: buffer, count: count,
                              dataType: dataType, op: reduction, ctx: ctx, fabric: fabric)

        case .tree(let root, let children):
            try treeReduce(children: children, root: root, buffer: buffer, count: count,
                           dataType: dataType, op: reduction, ctx: ctx, fabric: fabric)
            try treeBroadcast(children: children, root: root, buffer: buffer, count: count,
                              dataType: dataType, ctx: ctx, fabric: fabric)

        case .hierarchical(let islands, _):
            let leaders = islands.compactMap(\.first)
            guard let island = islands.first(where: { $0.contains(rank) }) else {
                throw MCCLError.topologyInvalid("rank \(rank) belongs to no island")
            }
            // 1. Reduce inside the fast island — every member ends with the
            //    island's partial sum.
            if island.count > 1 {
                try ringAllReduce(group: island, buffer: buffer, count: count,
                                  dataType: dataType, op: reduction, ctx: ctx, fabric: fabric)
            }
            // 2. Only the island leaders cross the slow bridge.
            if leaders.count > 1 && leaders.contains(rank) {
                try ringAllReduce(group: leaders, buffer: buffer, count: count,
                                  dataType: dataType, op: reduction, ctx: ctx, fabric: fabric)
            }
            // 3. Push the global result back down the fast island.
            if island.count > 1 {
                let children = TopologyPlanner.binomialTree(order: island, root: island[0])
                try treeBroadcast(children: children, root: island[0], buffer: buffer,
                                  count: count, dataType: dataType, ctx: ctx, fabric: fabric)
            }
        }

        if op == .avg {
            Kernels.scale(buffer.baseAddress!, count: count, dataType: dataType, by: 1.0 / Double(worldSize))
        }
    }

    // MARK: all-gather

    func allGatherSync(
        _ send: UnsafeRawBufferPointer, into recv: UnsafeMutableRawBufferPointer,
        count: Int, dataType: DataType, compression: WireCompression
    ) throws {
        guard count >= 0 else { throw MCCLError.invalidArgument("count must be >= 0") }
        let width = dataType.byteWidth
        let chunkBytes = count * width
        try checkCapacity(chunkBytes, send.count)
        try checkCapacity(chunkBytes * worldSize, recv.count)
        guard count > 0 else { return }

        let base = recv.baseAddress!
        (base + rank * chunkBytes).copyMemory(from: send.baseAddress!, byteCount: chunkBytes)
        guard worldSize > 1 else { return }

        let fabric = try requireFabric()
        let ctx = try WireContext(dataType: dataType, compression: compression)
        ctx.reserve(maxElements: count)

        // All-gather is bandwidth-shaped by construction: every rank must
        // receive (n-1) full chunks whatever the algorithm, so ring it is.
        let group = ringOrder(from: plan(messageBytes: chunkBytes * worldSize))
        let n = group.count
        guard let position = group.firstIndex(of: rank) else {
            throw MCCLError.topologyInvalid("rank \(rank) missing from ring order")
        }
        let next = try fabric.channel(to: group[(position + 1) % n])
        let previous = try fabric.channel(to: group[(position - 1 + n) % n])

        for step in 0..<(n - 1) {
            let sendSlot = group[(position - step + n) % n]
            let recvSlot = group[(position - step - 1 + n) % n]
            try exchangeChunk(
                sendOn: next, from: base + sendSlot * chunkBytes, sendCount: count,
                receiveOn: previous, into: base + recvSlot * chunkBytes, receiveCount: count,
                tag: UInt32(step), ctx: ctx)
        }
    }

    // MARK: broadcast

    func broadcastSync(
        _ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType,
        root: Int, compression: WireCompression
    ) throws {
        guard root >= 0, root < worldSize else {
            throw MCCLError.rankOutOfRange(rank: root, worldSize: worldSize)
        }
        let width = dataType.byteWidth
        try checkCapacity(count * width, buffer.count)
        guard worldSize > 1, count > 0 else { return }

        let fabric = try requireFabric()
        let ctx = try WireContext(dataType: dataType, compression: compression)
        ctx.reserve(maxElements: count)

        switch plan(messageBytes: count * width) {
        case .ring(let order):
            try chainBroadcast(group: order, root: root, buffer: buffer, count: count,
                               dataType: dataType, ctx: ctx, fabric: fabric)

        case .tree(_, _):
            // Re-root the tree at the caller's root; the planner's root only
            // says which rank is best connected, not who owns the data.
            let children = TopologyPlanner.binomialTree(order: defaultRingOrder, root: root)
            try treeBroadcast(children: children, root: root, buffer: buffer, count: count,
                              dataType: dataType, ctx: ctx, fabric: fabric)

        case .hierarchical(let islands, _):
            let leaders = islands.compactMap(\.first)
            guard let island = islands.first(where: { $0.contains(rank) }),
                  let rootIsland = islands.first(where: { $0.contains(root) }),
                  let rootLeader = rootIsland.first else {
                throw MCCLError.topologyInvalid("rank \(rank) or root \(root) belongs to no island")
            }
            // Hand the payload to the root's island leader, fan out across the
            // slow bridge once, then fan out inside each fast island.
            if root != rootLeader {
                if rank == root {
                    try sendChunk(on: try fabric.channel(to: rootLeader), from: buffer.baseAddress!,
                                  count: count, tag: 5, ctx: ctx)
                } else if rank == rootLeader {
                    try receiveChunk(on: try fabric.channel(to: root), into: buffer.baseAddress!,
                                     count: count, ctx: ctx)
                }
            }
            if leaders.count > 1 && leaders.contains(rank) {
                let children = TopologyPlanner.binomialTree(order: leaders, root: rootLeader)
                try treeBroadcast(children: children, root: rootLeader, buffer: buffer,
                                  count: count, dataType: dataType, ctx: ctx, fabric: fabric)
            }
            if island.count > 1 {
                let children = TopologyPlanner.binomialTree(order: island, root: island[0])
                try treeBroadcast(children: children, root: island[0], buffer: buffer,
                                  count: count, dataType: dataType, ctx: ctx, fabric: fabric)
            }
        }
    }

    // MARK: reduce

    func reduceSync(
        _ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType,
        op: ReduceOp, root: Int, compression: WireCompression
    ) throws {
        guard root >= 0, root < worldSize else {
            throw MCCLError.rankOutOfRange(rank: root, worldSize: worldSize)
        }
        let width = dataType.byteWidth
        try checkCapacity(count * width, buffer.count)
        guard worldSize > 1, count > 0 else {
            if op == .avg, count > 0 {
                Kernels.scale(buffer.baseAddress!, count: count, dataType: dataType, by: 1)
            }
            return
        }
        let fabric = try requireFabric()
        let ctx = try WireContext(dataType: dataType, compression: compression)
        let children = TopologyPlanner.binomialTree(order: defaultRingOrder, root: root)
        try treeReduce(children: children, root: root, buffer: buffer, count: count,
                       dataType: dataType, op: op.wireReduction, ctx: ctx, fabric: fabric)
        if op == .avg, rank == root {
            Kernels.scale(buffer.baseAddress!, count: count, dataType: dataType, by: 1.0 / Double(worldSize))
        }
    }

    // MARK: reduce-scatter

    func reduceScatterSync(
        _ send: UnsafeRawBufferPointer, into recv: UnsafeMutableRawBufferPointer,
        recvCount: Int, dataType: DataType, op: ReduceOp, compression: WireCompression
    ) throws {
        guard recvCount >= 0 else { throw MCCLError.invalidArgument("recvCount must be >= 0") }
        let width = dataType.byteWidth
        let segmentBytes = recvCount * width
        try checkCapacity(segmentBytes * worldSize, send.count)
        try checkCapacity(segmentBytes, recv.count)
        guard recvCount > 0 else { return }

        guard worldSize > 1 else {
            recv.baseAddress!.copyMemory(from: send.baseAddress!, byteCount: segmentBytes)
            if op == .avg { Kernels.scale(recv.baseAddress!, count: recvCount, dataType: dataType, by: 1) }
            return
        }

        let fabric = try requireFabric()
        let ctx = try WireContext(dataType: dataType, compression: compression)
        let group = ringOrder(from: plan(messageBytes: segmentBytes * worldSize))
        let n = group.count
        guard let position = group.firstIndex(of: rank) else {
            throw MCCLError.topologyInvalid("rank \(rank) missing from ring order")
        }

        // Chunk i of the working buffer is the segment destined for group[i],
        // so the ring's position-indexed arithmetic lands each rank on its own
        // output segment.
        let work = ScratchBuffer(capacity: segmentBytes * n)
        for i in 0..<n {
            (work.base + i * segmentBytes).copyMemory(
                from: send.baseAddress! + group[i] * segmentBytes, byteCount: segmentBytes)
        }

        let layout = ChunkLayout(count: recvCount * n, parts: n)
        ctx.reserve(maxElements: layout.maxCount)
        // chunkShift = -1 makes position p finish owning chunk p.
        let owned = try ringReduceScatterPhase(
            group: group, base: work.base, layout: layout, dataType: dataType,
            op: op.wireReduction, chunkShift: n - 1, ctx: ctx, fabric: fabric)
        precondition(owned == position, "ring reduce-scatter ownership mismatch")

        recv.baseAddress!.copyMemory(from: work.base + position * segmentBytes, byteCount: segmentBytes)
        if op == .avg {
            Kernels.scale(recv.baseAddress!, count: recvCount, dataType: dataType, by: 1.0 / Double(worldSize))
        }
    }
}

// MARK: - Top-k all-reduce with error feedback

extension Communicator {
    /// Sparsified all-reduce: every rank sends only its `k` largest-magnitude
    /// elements and keeps the rest as a residual for the next call.
    ///
    /// Shape of the algorithm, and why it is not the ring:
    ///
    /// 1. `work = buffer + residual`. The residual is everything previous calls
    ///    declined to send, so nothing is ever discarded — only delayed.
    /// 2. Select the `k` largest magnitudes of `work`. Those go on the wire;
    ///    `work` minus them becomes the new residual.
    /// 3. All-gather the fixed-size sparse blocks — `k` entries from every rank —
    ///    and sum them locally.
    ///
    /// Step 3 is an all-gather rather than a ring reduce-scatter on purpose. A
    /// ring would have to re-sparsify each partially reduced chunk on every hop,
    /// and the mass dropped by those hops belongs to no rank's residual, so the
    /// result would be biased in a way error feedback cannot repair. Gathering
    /// the sparse blocks instead makes the reduction *exact* given the
    /// sparsification, which is the property the convergence argument needs.
    ///
    /// Wire cost is `n·(4 + k·(4 + w))` bytes per rank against `2(n-1)/n·count·w`
    /// for a dense ring: a win whenever `k` is a small fraction of `count`.
    func topKAllReduceSync(
        _ buffer: UnsafeMutableRawBufferPointer, count: Int, dataType: DataType,
        op: ReduceOp, fraction: Double, stream: StreamID
    ) throws {
        guard fraction > 0, fraction <= 1, fraction.isFinite else {
            throw MCCLError.invalidArgument("topK fraction must be in (0, 1], got \(fraction)")
        }
        guard dataType.isFloatingPoint else {
            throw MCCLError.unsupportedCompression(
                "topK needs a floating-point dtype; \(dataType) is an accumulator type and magnitude "
                + "sparsification would corrupt it")
        }
        // Selecting a subset and summing it is only meaningful when the missing
        // elements are identities of the operation. They are for sum/avg, and
        // for nothing else: a top-k min would return the max of the selection.
        guard op == .sum || op == .avg else {
            throw MCCLError.unsupportedCompression(
                "topK is defined for .sum and .avg gradient reductions only, not \(op)")
        }
        _ = try requireFabric()

        let width = dataType.byteWidth
        let base = buffer.baseAddress!

        // 1. Fold the residual back in. 2. Select, send the selection, keep the
        //    rest. Both live in `TopKKernels.encodeBlock`, so the benchmark's
        //    codec probe measures exactly the code the collective runs.
        var work = residuals.load(stream, count: count)
        let k = TopK.count(elementCount: count, fraction: fraction)
        let blockBytes = TopKBlock.byteCount(nonZeros: k, valueWidth: width)
        let block = ScratchBuffer(capacity: blockBytes)
        TopKKernels.encodeBlock(source: base, count: count, dataType: dataType,
                                fraction: fraction, residual: &work, into: block.base)
        residuals.store(stream, work)

        // 3. Gather every rank's block and sum them. The blocks are uniform in
        //    size, so this is an ordinary all-gather over opaque bytes.
        let gathered = ScratchBuffer(capacity: blockBytes * worldSize)
        try allGatherSync(
            UnsafeRawBufferPointer(start: block.base, count: blockBytes),
            into: UnsafeMutableRawBufferPointer(start: gathered.base, count: blockBytes * worldSize),
            count: blockBytes, dataType: .int8, compression: .none)

        memset(base, 0, count * width)
        for peer in 0..<worldSize {
            try TopKBlock.accumulate(
                payload: gathered.base + peer * blockBytes, payloadBytes: blockBytes,
                elementCount: count, dataType: dataType, into: base)
        }
        if op == .avg {
            Kernels.scale(base, count: count, dataType: dataType, by: 1.0 / Double(worldSize))
        }
    }

    /// Bytes one rank puts on the wire for a `.topK` all-reduce of `count`
    /// elements — the sparse block, sent `(n-1)` times by the ring all-gather.
    /// Exposed so the benchmark and the tests can check the wire really is as
    /// sparse as the fraction promised.
    public func topKWireBytes(count: Int, dataType: DataType, fraction: Double) -> Int {
        let k = TopK.count(elementCount: count, fraction: fraction)
        let block = TopKBlock.byteCount(nonZeros: k, valueWidth: dataType.byteWidth)
        return (WireHeader.byteCount + block) * max(0, worldSize - 1)
    }
}
