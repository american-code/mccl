import XCTest
@testable import MCCL

final class CollectiveTests: XCTestCase {

    // Value rank `r` contributes at index `i`. Chosen so every sum over up to
    // four ranks is a small integer: exact in fp16, bf16 and int8 alike.
    private static func contribution(rank: Int, index: Int) -> Float {
        Float((rank + 1) * ((index % 7) + 1))
    }

    private static func expectedSum(worldSize: Int, index: Int) -> Float {
        Float(worldSize * (worldSize + 1) / 2 * ((index % 7) + 1))
    }

    // MARK: - all-reduce, every dtype, 2/3/4 ranks

    func testAllReduceSumAcrossDataTypesAndWorldSizes() async throws {
        let count = 1000
        for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
            for worldSize in [2, 3, 4] {
                try await Ranks.run(worldSize: worldSize) { comm in
                    let buffer = Buf.allocate(count: count, dataType: dataType)
                    defer { buffer.deallocate() }
                    Buf.fill(buffer, count: count, dataType: dataType) {
                        Self.contribution(rank: comm.rank, index: $0)
                    }
                    try await comm.allReduce(buffer, count: count, dataType: dataType, op: .sum)

                    let result = Buf.read(buffer, count: count, dataType: dataType)
                    for i in 0..<count {
                        XCTAssertEqual(result[i], Self.expectedSum(worldSize: worldSize, index: i),
                                       "\(dataType) n=\(worldSize) rank=\(comm.rank) index=\(i)")
                    }
                }
            }
        }
    }

    func testAllReduceMinMaxAvgProd() async throws {
        let count = 257
        let worldSize = 4

        try await Ranks.run(worldSize: worldSize) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }

            // max
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .max)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Float(worldSize * ((i % 7) + 1)))
            }

            // min
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .min)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Float((i % 7) + 1))
            }

            // avg
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .avg)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                let expected = Self.expectedSum(worldSize: worldSize, index: i) / Float(worldSize)
                XCTAssertEqual(value, expected, accuracy: 1e-4)
            }

            // prod — keep the factors near 1 so four ranks stay in range.
            Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank) + 1 }
            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .prod)
            for value in Buf.read(buffer, count: count, dataType: .float32) {
                XCTAssertEqual(value, 24, accuracy: 1e-3)   // 1*2*3*4
            }
        }
    }

    func testAllReduceOverLoopbackTransport() async throws {
        let count = 4096
        try await Ranks.runLoopback(worldSize: 4) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i))
            }
        }
    }

    /// Counts that do not divide evenly across ranks exercise the ragged tail
    /// of the chunk layout — including counts smaller than the world size.
    func testAllReduceWithAwkwardCounts() async throws {
        for count in [1, 2, 3, 5, 7, 4097] {
            try await Ranks.run(worldSize: 4) { comm in
                let buffer = Buf.allocate(count: count, dataType: .float32)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
                try await comm.allReduce(buffer, count: count, dataType: .float32)
                for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                    XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i), "count=\(count) index=\(i)")
                }
            }
        }
    }

    /// 8 MiB per rank — 2 MiB ring chunks, well past any socket buffer. If the
    /// send/receive overlap were not genuinely concurrent this would deadlock
    /// rather than fail.
    func testLargeMessageAllReduceDoesNotStall() async throws {
        let count = 2 * 1024 * 1024        // 8 MiB of fp32
        try await Ranks.run(worldSize: 4) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank + 1) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            let result = Buf.read(buffer, count: count, dataType: .float32)
            XCTAssertEqual(result.first, 10)
            XCTAssertEqual(result.last, 10)
            XCTAssertFalse(result.contains { $0 != 10 })
        }
    }

    /// Same size, but forced down the tree path, where each hop moves the whole
    /// buffer in one blocking write.
    func testLargeMessageTreeAllReduceDoesNotStall() async throws {
        let count = 1024 * 1024            // 4 MiB of fp32
        let children = TopologyPlanner.binomialTree(order: [0, 1, 2, 3], root: 0)
        try await Ranks.run(worldSize: 4, planOverride: .tree(root: 0, children: children)) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank + 1) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            XCTAssertFalse(Buf.read(buffer, count: count, dataType: .float32).contains { $0 != 10 })
        }
    }

    func testAllReduceRejectsUndersizedBuffer() async throws {
        try await Ranks.run(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 4, dataType: .float32)
            defer { buffer.deallocate() }
            do {
                try await comm.allReduce(buffer, count: 1024, dataType: .float32)
                XCTFail("expected .bufferTooSmall")
            } catch let error as MCCLError {
                guard case .bufferTooSmall = error else { return XCTFail("got \(error)") }
            }
        }
    }

    // MARK: - plan-specific execution

    func testAllReduceUsingForcedTreePlan() async throws {
        let count = 1024
        let worldSize = 4
        let children = TopologyPlanner.binomialTree(order: [0, 1, 2, 3], root: 0)
        XCTAssertEqual(children[0], [1, 2])
        XCTAssertEqual(children[2], [3])

        try await Ranks.run(worldSize: worldSize, planOverride: .tree(root: 0, children: children)) { comm in
            XCTAssertEqual(comm.plan(messageBytes: count * 4), .tree(root: 0, children: children))
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: worldSize, index: i))
            }
        }
    }

    func testAllReduceUsingForcedHierarchicalPlan() async throws {
        let count = 1024
        let plan = CollectivePlan.hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 0)
        try await Ranks.run(worldSize: 4, planOverride: plan) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i))
            }
        }
    }

    func testAllReduceUsingHierarchicalPlanWithAnUnevenIsland() async throws {
        let count = 512
        let plan = CollectivePlan.hierarchical(islands: [[0, 1, 2], [3]], interIslandRoot: 0)
        try await Ranks.run(worldSize: 4, planOverride: plan) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i))
            }
        }
    }

    /// The planner, not an override, picks hierarchical for this fabric — and
    /// the result must still be right.
    func testPlannerChosenHierarchicalAllReduceIsCorrect() async throws {
        // 1 MiB — comfortably past this fabric's ~200 KiB tree/ring crossover.
        let count = 256 * 1024
        let topology = Fabrics.twoIslandsBridged()
        guard case .hierarchical = TopologyPlanner.plan(for: topology, messageBytes: count * 4) else {
            return XCTFail("expected the bridged fabric to plan hierarchically")
        }
        try await Ranks.run(worldSize: 4, topology: topology) { comm in
            guard case .hierarchical = comm.plan(messageBytes: count * 4) else {
                return XCTFail("communicator did not adopt the hierarchical plan")
            }
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i))
            }
        }
    }

    /// Small messages on a fast uniform fabric plan as a tree; run one for real.
    func testPlannerChosenTreeAllReduceIsCorrect() async throws {
        let count = 64
        let topology = Fabrics.uniformFast(nodeCount: 4)
        guard case .tree = TopologyPlanner.plan(for: topology, messageBytes: count * 4) else {
            return XCTFail("expected a tree plan for a latency-bound message")
        }
        try await Ranks.run(worldSize: 4, topology: topology) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.allReduce(buffer, count: count, dataType: .float32)
            for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: 4, index: i))
            }
        }
    }

    // MARK: - compression through a real collective

    func testAllReduceWithDowncastCompression() async throws {
        let count = 2048
        try await Ranks.run(worldSize: 4) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) {
                Float($0 % 100) * 0.37 + Float(comm.rank)
            }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .downcast)
            let result = Buf.read(buffer, count: count, dataType: .float32)
            for i in 0..<count {
                let expected = (0..<4).map { Float(i % 100) * 0.37 + Float($0) }.reduce(0, +)
                // Every hop re-quantises, so allow a few fp16 ulps.
                XCTAssertEqual(result[i], expected, accuracy: max(abs(expected) * 0.01, 0.05))
            }
        }
    }

    func testAllReduceWithInt8BlockwiseCompression() async throws {
        let count = 2048
        try await Ranks.run(worldSize: 4) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank) + 1 }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .int8Blockwise(blockSize: 128))
            for value in Buf.read(buffer, count: count, dataType: .float32) {
                XCTAssertEqual(value, 10, accuracy: 0.5)
            }
        }
    }

    // MARK: - all-gather

    func testAllGather() async throws {
        let count = 333
        let worldSize = 4
        try await Ranks.run(worldSize: worldSize) { comm in
            let send = Buf.allocate(count: count, dataType: .float32)
            let recv = Buf.allocate(count: count * worldSize, dataType: .float32)
            defer { send.deallocate(); recv.deallocate() }
            Buf.fill(send, count: count, dataType: .float32) { Float(comm.rank * 1000 + $0) }

            try await comm.allGather(UnsafeRawBufferPointer(send), into: recv,
                                     count: count, dataType: .float32)

            let gathered = Buf.read(recv, count: count * worldSize, dataType: .float32)
            for peer in 0..<worldSize {
                for i in 0..<count {
                    XCTAssertEqual(gathered[peer * count + i], Float(peer * 1000 + i),
                                   "rank \(comm.rank) slot \(peer) index \(i)")
                }
            }
        }
    }

    func testAllGatherWithCompression() async throws {
        let count = 128
        let worldSize = 3
        try await Ranks.run(worldSize: worldSize) { comm in
            let send = Buf.allocate(count: count, dataType: .float32)
            let recv = Buf.allocate(count: count * worldSize, dataType: .float32)
            defer { send.deallocate(); recv.deallocate() }
            Buf.fill(send, count: count, dataType: .float32) { Float(comm.rank) + Float($0) * 0.5 }

            try await comm.allGather(UnsafeRawBufferPointer(send), into: recv,
                                     count: count, dataType: .float32, compression: .downcast)

            let gathered = Buf.read(recv, count: count * worldSize, dataType: .float32)
            for peer in 0..<worldSize {
                for i in 0..<count {
                    let expected = Float(peer) + Float(i) * 0.5
                    XCTAssertEqual(gathered[peer * count + i], expected, accuracy: 0.05)
                }
            }
        }
    }

    // MARK: - broadcast

    func testBroadcastFromEveryRoot() async throws {
        let count = 777
        let worldSize = 4
        for root in 0..<worldSize {
            try await Ranks.run(worldSize: worldSize) { comm in
                let buffer = Buf.allocate(count: count, dataType: .float32)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: .float32) {
                    comm.rank == root ? Float($0 * 3 + 1) : -1
                }
                try await comm.broadcast(buffer, count: count, dataType: .float32, root: root)
                for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                    XCTAssertEqual(value, Float(i * 3 + 1), "root=\(root) rank=\(comm.rank) index=\(i)")
                }
            }
        }
    }

    func testBroadcastUsingTreePlan() async throws {
        let count = 64
        let topology = Fabrics.uniformFast(nodeCount: 4)
        try await Ranks.run(worldSize: 4, topology: topology) { comm in
            let buffer = Buf.allocate(count: count, dataType: .int32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .int32) { comm.rank == 2 ? Float($0 + 5) : -1 }
            try await comm.broadcast(buffer, count: count, dataType: .int32, root: 2)
            for (i, value) in Buf.read(buffer, count: count, dataType: .int32).enumerated() {
                XCTAssertEqual(value, Float(i + 5))
            }
        }
    }

    func testBroadcastUsingHierarchicalPlan() async throws {
        let count = 512
        let plan = CollectivePlan.hierarchical(islands: [[0, 1], [2, 3]], interIslandRoot: 0)
        for root in [0, 1, 3] {
            try await Ranks.run(worldSize: 4, planOverride: plan) { comm in
                let buffer = Buf.allocate(count: count, dataType: .float32)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: .float32) {
                    comm.rank == root ? Float($0 % 91) : -1
                }
                try await comm.broadcast(buffer, count: count, dataType: .float32, root: root)
                for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                    XCTAssertEqual(value, Float(i % 91), "root=\(root) rank=\(comm.rank) index=\(i)")
                }
            }
        }
    }

    func testBroadcastRejectsOutOfRangeRoot() async throws {
        try await Ranks.run(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 4, dataType: .float32)
            defer { buffer.deallocate() }
            do {
                try await comm.broadcast(buffer, count: 4, dataType: .float32, root: 9)
                XCTFail("expected .rankOutOfRange")
            } catch let error as MCCLError {
                XCTAssertEqual(error, .rankOutOfRange(rank: 9, worldSize: 2))
            }
        }
    }

    // MARK: - reduce-scatter

    func testReduceScatter() async throws {
        let segment = 250
        let worldSize = 4
        try await Ranks.run(worldSize: worldSize) { comm in
            let send = Buf.allocate(count: segment * worldSize, dataType: .float32)
            let recv = Buf.allocate(count: segment, dataType: .float32)
            defer { send.deallocate(); recv.deallocate() }
            Buf.fill(send, count: segment * worldSize, dataType: .float32) {
                Self.contribution(rank: comm.rank, index: $0)
            }

            try await comm.reduceScatter(UnsafeRawBufferPointer(send), into: recv,
                                         recvCount: segment, dataType: .float32, op: .sum)

            let result = Buf.read(recv, count: segment, dataType: .float32)
            for i in 0..<segment {
                let globalIndex = comm.rank * segment + i
                XCTAssertEqual(result[i], Self.expectedSum(worldSize: worldSize, index: globalIndex),
                               "rank \(comm.rank) index \(i)")
            }
        }
    }

    func testReduceScatterAverage() async throws {
        let segment = 64
        let worldSize = 3
        try await Ranks.run(worldSize: worldSize) { comm in
            let send = Buf.allocate(count: segment * worldSize, dataType: .float32)
            let recv = Buf.allocate(count: segment, dataType: .float32)
            defer { send.deallocate(); recv.deallocate() }
            Buf.fill(send, count: segment * worldSize, dataType: .float32) {
                Self.contribution(rank: comm.rank, index: $0)
            }
            try await comm.reduceScatter(UnsafeRawBufferPointer(send), into: recv,
                                         recvCount: segment, dataType: .float32, op: .avg)
            let result = Buf.read(recv, count: segment, dataType: .float32)
            for i in 0..<segment {
                let globalIndex = comm.rank * segment + i
                let expected = Self.expectedSum(worldSize: worldSize, index: globalIndex) / Float(worldSize)
                XCTAssertEqual(result[i], expected, accuracy: 1e-4)
            }
        }
    }

    /// all-gather(reduce-scatter(x)) must equal all-reduce(x) — the identity the
    /// ring all-reduce is built on, checked end to end through the public API.
    func testReduceScatterThenAllGatherEqualsAllReduce() async throws {
        let segment = 97
        let worldSize = 4
        try await Ranks.run(worldSize: worldSize) { comm in
            let total = segment * worldSize
            let send = Buf.allocate(count: total, dataType: .float32)
            let scattered = Buf.allocate(count: segment, dataType: .float32)
            let gathered = Buf.allocate(count: total, dataType: .float32)
            defer { send.deallocate(); scattered.deallocate(); gathered.deallocate() }

            Buf.fill(send, count: total, dataType: .float32) { Self.contribution(rank: comm.rank, index: $0) }
            try await comm.reduceScatter(UnsafeRawBufferPointer(send), into: scattered,
                                         recvCount: segment, dataType: .float32, op: .sum)
            try await comm.allGather(UnsafeRawBufferPointer(scattered), into: gathered,
                                     count: segment, dataType: .float32)

            for (i, value) in Buf.read(gathered, count: total, dataType: .float32).enumerated() {
                XCTAssertEqual(value, Self.expectedSum(worldSize: worldSize, index: i))
            }
        }
    }

    // MARK: - reduce

    func testReduceOntoRoot() async throws {
        let count = 512
        let worldSize = 4
        for root in [0, 3] {
            try await Ranks.run(worldSize: worldSize) { comm in
                let buffer = Buf.allocate(count: count, dataType: .float32)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: .float32) {
                    Self.contribution(rank: comm.rank, index: $0)
                }
                try await comm.reduce(buffer, count: count, dataType: .float32, op: .sum, root: root)
                guard comm.rank == root else { return }
                for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                    XCTAssertEqual(value, Self.expectedSum(worldSize: worldSize, index: i))
                }
            }
        }
    }

    // MARK: - repeated operations on one communicator

    func testManySequentialCollectivesOnTheSameCommunicator() async throws {
        let count = 300
        try await Ranks.run(worldSize: 4) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            for iteration in 0..<12 {
                Buf.fill(buffer, count: count, dataType: .float32) { _ in Float(comm.rank + iteration) }
                try await comm.allReduce(buffer, count: count, dataType: .float32)
                let expected = Float((0..<4).map { $0 + iteration }.reduce(0, +))
                for value in Buf.read(buffer, count: count, dataType: .float32) {
                    XCTAssertEqual(value, expected, "iteration \(iteration)")
                }
            }
        }
    }
}
