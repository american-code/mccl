import XCTest
@testable import MCCL

/// Top-k sparsification with error feedback.
///
/// The property under test is not "the answer is close" — a single sparsified
/// all-reduce is deliberately, visibly lossy. It is that the *residual* carries
/// the difference forward, so a sequence of calls converges on the uncompressed
/// answer instead of biasing away from it.
final class TopKTests: XCTestCase {

    /// Rank `r`'s gradient. Spread over a couple of orders of magnitude so
    /// selection has something to prefer.
    private static func gradient(rank: Int, index: Int, count: Int) -> Float {
        let phase = Float(index) / Float(count) * 6.2831853
        return Float(rank + 1) * (sin(phase) + 0.25 * cos(3 * phase))
    }

    private static func exactSum(worldSize: Int, index: Int, count: Int) -> Float {
        (0..<worldSize).reduce(Float(0)) { $0 + gradient(rank: $1, index: index, count: count) }
    }

    // MARK: - Selection

    func testSelectIndicesReturnsTheLargestMagnitudesInOrder() {
        let magnitudes: [Float] = [3, 1, 9, 4, 0, 7, 2, 8]
        XCTAssertEqual(TopK.selectIndices(magnitudes: magnitudes, k: 3), [2, 5, 7])
        XCTAssertEqual(TopK.selectIndices(magnitudes: magnitudes, k: 1), [2])
        XCTAssertEqual(TopK.selectIndices(magnitudes: magnitudes, k: 8), Array(0..<8))
        XCTAssertEqual(TopK.selectIndices(magnitudes: magnitudes, k: 0), [])
    }

    func testSelectIndicesAgreesWithASortOnRandomData() {
        var generator = SystemRandomNumberGenerator()
        for _ in 0..<20 {
            let n = Int.random(in: 1...500, using: &generator)
            let magnitudes = (0..<n).map { _ in abs(Float.random(in: -100...100, using: &generator)) }
            let k = Int.random(in: 1...n, using: &generator)
            let selected = TopK.selectIndices(magnitudes: magnitudes, k: k)
            XCTAssertEqual(selected.count, k)
            XCTAssertEqual(selected, selected.sorted(), "indices must come back ascending")

            // Every selected magnitude is >= every rejected one.
            let rejected = Set(0..<n).subtracting(selected)
            let smallestSelected = selected.map { magnitudes[$0] }.min() ?? 0
            let largestRejected = rejected.map { magnitudes[$0] }.max() ?? -Float.infinity
            XCTAssertGreaterThanOrEqual(smallestSelected, largestRejected)
        }
    }

    func testFractionRoundsUpAndAlwaysSendsSomething() {
        XCTAssertEqual(TopK.count(elementCount: 1000, fraction: 0.01), 10)
        XCTAssertEqual(TopK.count(elementCount: 1000, fraction: 0.0105), 11, "must round up")
        XCTAssertEqual(TopK.count(elementCount: 1000, fraction: 1.0), 1000)
        XCTAssertEqual(TopK.count(elementCount: 10, fraction: 1e-9), 1,
                       "a collective that sends nothing never makes progress")
        XCTAssertEqual(TopK.count(elementCount: 0, fraction: 0.5), 0)
    }

    // MARK: - Block layout

    func testSparseBlockRoundTripsThroughAccumulate() throws {
        for dataType in [DataType.float32, .float16, .bfloat16] {
            let count = 40
            let values = (0..<count).map { Float($0 % 9) - 4 }
            let indices = [1, 7, 22, 39]
            let bytes = TopKBlock.byteCount(nonZeros: indices.count, valueWidth: dataType.byteWidth)
            let block = ScratchBuffer(capacity: bytes)
            let written = TopKBlock.write(indices: indices, values: values,
                                          dataType: dataType, into: block.base)
            XCTAssertEqual(written, bytes)
            XCTAssertEqual(try TopKBlock.nonZeroCount(payload: block.base, payloadBytes: bytes), 4)

            let dense = Buf.allocate(count: count, dataType: dataType)
            defer { dense.deallocate() }
            // Accumulate twice: the block is a contribution, not an assignment.
            try TopKBlock.accumulate(payload: block.base, payloadBytes: bytes, elementCount: count,
                                     dataType: dataType, into: dense.baseAddress!)
            try TopKBlock.accumulate(payload: block.base, payloadBytes: bytes, elementCount: count,
                                     dataType: dataType, into: dense.baseAddress!)

            let result = Buf.read(dense, count: count, dataType: dataType)
            for i in 0..<count {
                let expected = indices.contains(i) ? 2 * values[i] : 0
                XCTAssertEqual(result[i], expected, accuracy: 0.01, "\(dataType) index \(i)")
            }
        }
    }

    func testMalformedSparseBlockIsRejected() {
        let block = ScratchBuffer(capacity: 64)
        block.base.storeBytes(of: UInt32(1000).littleEndian, toByteOffset: 0, as: UInt32.self)
        XCTAssertThrowsError(try TopKBlock.accumulate(
            payload: block.base, payloadBytes: 8, elementCount: 16,
            dataType: .float32, into: block.base + 32))

        // Well-formed header, but an index past the end of the destination.
        let bytes = TopKBlock.byteCount(nonZeros: 1, valueWidth: 4)
        _ = TopKBlock.write(indices: [99], values: [Float](repeating: 1, count: 100),
                            dataType: .float32, into: block.base)
        let dense = Buf.allocate(count: 16, dataType: .float32)
        defer { dense.deallocate() }
        XCTAssertThrowsError(try TopKBlock.accumulate(
            payload: block.base, payloadBytes: bytes, elementCount: 16,
            dataType: .float32, into: dense.baseAddress!))
    }

    // MARK: - Exactness at fraction 1

    func testTopKAtFullFractionMatchesAnUncompressedAllReduce() async throws {
        let count = 257
        for worldSize in [2, 4] {
            try await Ranks.runLoopback(worldSize: worldSize) { comm in
                let buffer = Buf.allocate(count: count, dataType: .float32)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: .float32) {
                    Self.gradient(rank: comm.rank, index: $0, count: count)
                }
                try await comm.allReduce(buffer, count: count, dataType: .float32,
                                         op: .sum, compression: .topK(fraction: 1.0))
                for (i, value) in Buf.read(buffer, count: count, dataType: .float32).enumerated() {
                    XCTAssertEqual(value, Self.exactSum(worldSize: worldSize, index: i, count: count),
                                   accuracy: 1e-4, "n=\(worldSize) index \(i)")
                }
                XCTAssertEqual(comm.topKResidual()?.allSatisfy { $0 == 0 }, true,
                               "sending everything leaves nothing behind")
            }
        }
    }

    func testTopKAtFullFractionSupportsAvgAndHalfPrecision() async throws {
        let count = 96
        for dataType in [DataType.float32, .float16, .bfloat16] {
            try await Ranks.runLoopback(worldSize: 4) { comm in
                let buffer = Buf.allocate(count: count, dataType: dataType)
                defer { buffer.deallocate() }
                Buf.fill(buffer, count: count, dataType: dataType) { Float(comm.rank + 1) * Float(($0 % 5) + 1) }
                try await comm.allReduce(buffer, count: count, dataType: dataType,
                                         op: .avg, compression: .topK(fraction: 1.0))
                for (i, value) in Buf.read(buffer, count: count, dataType: dataType).enumerated() {
                    XCTAssertEqual(value, Float(10 * ((i % 5) + 1)) / 4.0, accuracy: 0.1,
                                   "\(dataType) index \(i)")
                }
            }
        }
    }

    // MARK: - Error feedback

    func testASingleTopKCallIsVisiblyLossy() async throws {
        let count = 128
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) {
                Self.gradient(rank: comm.rank, index: $0, count: count)
            }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 0.05))
            let result = Buf.read(buffer, count: count, dataType: .float32)
            let zeros = result.filter { $0 == 0 }.count
            // Two ranks at 5% select at most 2·⌈0.05·128⌉ = 14 distinct indices.
            XCTAssertGreaterThanOrEqual(zeros, count - 14,
                                        "one sparsified call must leave most elements untouched")
        }
    }

    func testResidualPlusTransmittedEqualsTheOriginalGradient() async throws {
        let count = 100
        let worldSize = 4
        let fraction = 0.1
        let residuals = Collector<[Float]>()
        let outputs = Collector<[Float]>()

        try await Ranks.runLoopback(worldSize: worldSize) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) {
                Self.gradient(rank: comm.rank, index: $0, count: count)
            }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: fraction))
            residuals.set(comm.rank, comm.topKResidual() ?? [])
            outputs.set(comm.rank, Buf.read(buffer, count: count, dataType: .float32))
        }

        // Nothing is lost: what came out, plus what every rank is still holding,
        // is exactly the uncompressed all-reduce.
        let kept = (0..<worldSize).map { residuals.get($0) ?? [] }
        for rank in 0..<worldSize {
            XCTAssertEqual(kept[rank].count, count)
            XCTAssertEqual(kept[rank].filter { $0 != 0 }.count, count - TopK.count(elementCount: count, fraction: fraction),
                           "exactly the unsent elements survive in the residual")
        }
        for rank in 0..<worldSize {
            let output = outputs.get(rank)!
            for i in 0..<count {
                let held = kept.reduce(Float(0)) { $0 + $1[i] }
                XCTAssertEqual(output[i] + held,
                               Self.exactSum(worldSize: worldSize, index: i, count: count),
                               accuracy: 1e-4, "index \(i)")
            }
        }
    }

    /// The headline property: repeating the same reduction under error feedback
    /// makes the *running mean* converge on the uncompressed answer, and the
    /// error shrinks as the run gets longer.
    func testRepeatedTopKConvergesTowardTheUncompressedResult() async throws {
        let count = 64
        let worldSize = 2
        let fraction = 0.25
        let iterations = 400
        let checkpoint = 40

        let earlyError = Collector<Float>()
        let lateError = Collector<Float>()

        try await Ranks.runLoopback(worldSize: worldSize) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            var running = [Float](repeating: 0, count: count)
            var earlyMax: Float = 0

            for iteration in 1...iterations {
                Buf.fill(buffer, count: count, dataType: .float32) {
                    Self.gradient(rank: comm.rank, index: $0, count: count)
                }
                try await comm.allReduce(buffer, count: count, dataType: .float32,
                                         op: .sum, compression: .topK(fraction: fraction))
                let result = Buf.read(buffer, count: count, dataType: .float32)
                for i in 0..<count { running[i] += result[i] }

                if iteration == checkpoint {
                    earlyMax = (0..<count).map { i in
                        abs(running[i] / Float(checkpoint)
                            - Self.exactSum(worldSize: worldSize, index: i, count: count))
                    }.max() ?? 0
                }
            }

            let lateMax = (0..<count).map { i in
                abs(running[i] / Float(iterations)
                    - Self.exactSum(worldSize: worldSize, index: i, count: count))
            }.max() ?? 0
            earlyError.set(comm.rank, earlyMax)
            lateError.set(comm.rank, lateMax)
        }

        for rank in 0..<worldSize {
            let early = earlyError.get(rank)!
            let late = lateError.get(rank)!
            XCTAssertLessThan(late, 0.05,
                              "rank \(rank): the mean of \(iterations) sparsified all-reduces should sit "
                              + "on the uncompressed answer, got max error \(late)")
            XCTAssertLessThan(late, early,
                              "rank \(rank): error must shrink with more iterations (\(early) -> \(late))")
        }
    }

    func testErrorFeedbackIsPerStream() async throws {
        let count = 32
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }

            Buf.fill(buffer, count: count, dataType: .float32) { Float($0 + 1) }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 0.25),
                                     stream: StreamID(1))
            Buf.fill(buffer, count: count, dataType: .float32) { _ in 0 }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 0.25),
                                     stream: StreamID(2))

            let one = comm.topKResidual(for: StreamID(1))
            let two = comm.topKResidual(for: StreamID(2))
            XCTAssertNotNil(one)
            XCTAssertNotNil(two)
            XCTAssertTrue(one!.contains { $0 != 0 }, "stream 1 held back part of a real gradient")
            XCTAssertTrue(two!.allSatisfy { $0 == 0 }, "stream 2 only ever saw zeros")
            XCTAssertNil(comm.topKResidual(for: .default), "the default stream was never used")

            comm.clearTopKResiduals(stream: StreamID(1))
            XCTAssertNil(comm.topKResidual(for: StreamID(1)))
            XCTAssertNotNil(comm.topKResidual(for: StreamID(2)))
            comm.clearTopKResiduals()
            XCTAssertNil(comm.topKResidual(for: StreamID(2)))
        }
    }

    func testChangingTheElementCountRestartsTheResidual() async throws {
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 64, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: 64, dataType: .float32) { Float($0 + 1) }
            try await comm.allReduce(buffer, count: 64, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 0.1))
            XCTAssertEqual(comm.topKResidual()?.count, 64)

            Buf.fill(buffer, count: 16, dataType: .float32) { Float($0 + 1) }
            try await comm.allReduce(buffer, count: 16, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: 0.1))
            XCTAssertEqual(comm.topKResidual()?.count, 16,
                           "a different tensor on the same stream starts a fresh history")
        }
    }

    // MARK: - Wire sparsity

    func testWirePayloadIsAsSparseAsTheFractionPromises() async throws {
        let count = 20_000
        let worldSize = 2
        let fraction = 0.01
        let transport = CountingTransport()
        let dense = CountingTransport()

        try await Ranks.run(worldSize: worldSize, transport: transport) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) {
                Self.gradient(rank: comm.rank, index: $0, count: count)
            }
            try await comm.allReduce(buffer, count: count, dataType: .float32,
                                     op: .sum, compression: .topK(fraction: fraction))
            XCTAssertEqual(comm.topKWireBytes(count: count, dataType: .float32, fraction: fraction),
                           (WireHeader.byteCount + 4 + 200 * 8) * (worldSize - 1))
        }

        try await Ranks.run(worldSize: worldSize, transport: dense) { comm in
            let buffer = Buf.allocate(count: count, dataType: .float32)
            defer { buffer.deallocate() }
            Buf.fill(buffer, count: count, dataType: .float32) {
                Self.gradient(rank: comm.rank, index: $0, count: count)
            }
            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)
        }

        // k = 200 of 20 000 elements: 4 + 200·(4 + 4) = 1604 bytes per rank per
        // hop, against 2·(n-1)/n·80 000 = 80 000 for the dense ring.
        let expectedPerHop = WireHeader.byteCount + TopKBlock.byteCount(nonZeros: 200, valueWidth: 4)
        // Mesh bring-up costs one bare header per ordered pair; that is not the
        // collective's traffic.
        let handshake = WireHeader.byteCount * (worldSize * (worldSize - 1) / 2)
        XCTAssertEqual(transport.bytesSent - handshake, expectedPerHop * worldSize * (worldSize - 1),
                       "top-k must put exactly the sparse blocks on the wire")
        XCTAssertLessThan(Double(transport.bytesSent), Double(dense.bytesSent) * 0.05,
                          "1% top-k sent \(transport.bytesSent) bytes vs \(dense.bytesSent) dense")
    }

    // MARK: - Rejections

    func testTopKRejectsIntegerDataTypes() async throws {
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 16, dataType: .int32)
            defer { buffer.deallocate() }
            await XCTAssertThrowsMCCLError({
                try await comm.allReduce(buffer, count: 16, dataType: .int32,
                                         op: .sum, compression: .topK(fraction: 0.5))
            }) { error in
                guard case .unsupportedCompression = error else {
                    return XCTFail("expected .unsupportedCompression, got \(error)")
                }
            }
        }
    }

    func testTopKRejectsNonSummingOperations() async throws {
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 16, dataType: .float32)
            defer { buffer.deallocate() }
            for op in [ReduceOp.min, .max, .prod] {
                await XCTAssertThrowsMCCLError({
                    try await comm.allReduce(buffer, count: 16, dataType: .float32,
                                             op: op, compression: .topK(fraction: 0.5))
                }) { error in
                    guard case .unsupportedCompression = error else {
                        return XCTFail("expected .unsupportedCompression for \(op), got \(error)")
                    }
                }
            }
        }
    }

    func testTopKRejectsFractionsOutsideTheUnitInterval() async throws {
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let buffer = Buf.allocate(count: 16, dataType: .float32)
            defer { buffer.deallocate() }
            for fraction in [0.0, -0.5, 1.5, Double.nan] {
                await XCTAssertThrowsMCCLError({
                    try await comm.allReduce(buffer, count: 16, dataType: .float32,
                                             op: .sum, compression: .topK(fraction: fraction))
                }) { error in
                    guard case .invalidArgument = error else {
                        return XCTFail("expected .invalidArgument for \(fraction), got \(error)")
                    }
                }
            }
        }
    }

    func testTopKIsRejectedByCollectivesThatCannotAccountForTheLoss() async throws {
        let count = 32
        try await Ranks.runLoopback(worldSize: 2) { comm in
            let send = Buf.allocate(count: count, dataType: .float32)
            let recv = Buf.allocate(count: count * 2, dataType: .float32)
            defer { send.deallocate(); recv.deallocate() }

            await XCTAssertThrowsMCCLError({
                try await comm.allGather(UnsafeRawBufferPointer(send), into: recv,
                                         count: count, dataType: .float32,
                                         compression: .topK(fraction: 0.5))
            }) { error in
                guard case .unsupportedCompression = error else {
                    return XCTFail("expected .unsupportedCompression, got \(error)")
                }
            }
            await XCTAssertThrowsMCCLError({
                try await comm.broadcast(send, count: count, dataType: .float32, root: 0,
                                         compression: .topK(fraction: 0.5))
            }) { error in
                guard case .unsupportedCompression = error else {
                    return XCTFail("expected .unsupportedCompression, got \(error)")
                }
            }
        }
    }
}

// MARK: - async throwing assertion

/// `XCTAssertThrowsError` has no async form; this is the same contract.
func XCTAssertThrowsMCCLError(
    _ body: () async throws -> Void,
    _ inspect: (MCCLError) -> Void = { _ in },
    file: StaticString = #filePath, line: UInt = #line
) async {
    do {
        try await body()
        XCTFail("expected an MCCLError, none was thrown", file: file, line: line)
    } catch let error as MCCLError {
        inspect(error)
    } catch {
        XCTFail("expected an MCCLError, got \(error)", file: file, line: line)
    }
}
