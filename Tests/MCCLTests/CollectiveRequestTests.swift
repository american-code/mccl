import XCTest
@testable import MCCL

/// Non-blocking collectives: issue, order, completion, and failure.
///
/// The property that actually matters is ordering. Ranks must run collectives
/// in the same sequence or they deadlock against each other, and with
/// non-blocking issue the only sequence every rank can agree on is the order
/// the calls were made. So these check that several requests outstanding at
/// once still land in issue order, not merely that one of them completes.
final class CollectiveRequestTests: XCTestCase {

    func testAnIssuedRequestCompletesWithTheRightAnswer() throws {
        let comms = try Communicator.loopbackGroup(worldSize: 3)
        defer { comms.forEach { $0.shutdown() } }

        let count = 128
        var buffers: [UnsafeMutableRawBufferPointer] = []
        defer { buffers.forEach { $0.deallocate() } }
        var requests: [CollectiveRequest] = []
        for comm in comms {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
            buffers.append(buffer)
            let floats = buffer.bindMemory(to: Float.self)
            for i in 0..<count { floats[i] = Float(comm.rank + 1) }
            requests.append(comm.allReduceAsync(buffer, count: count, dataType: .float32, op: .sum))
        }
        for request in requests { try request.wait() }
        for buffer in buffers {
            let floats = buffer.bindMemory(to: Float.self)
            for i in 0..<count { XCTAssertEqual(floats[i], 6, accuracy: 1e-5, "element \(i)") }
        }
    }

    /// Four requests per rank, all issued before any is waited, each carrying a
    /// different multiplier. A request completing against the wrong buffer — or
    /// two ranks disagreeing about the order — shows up as a wrong sum.
    func testSeveralOutstandingRequestsCompleteInIssueOrder() throws {
        let worldSize = 3
        let inFlight = 4
        let count = 64
        let comms = try Communicator.loopbackGroup(worldSize: worldSize)
        defer { comms.forEach { $0.shutdown() } }

        var buffers: [[UnsafeMutableRawBufferPointer]] = []
        var requests: [[CollectiveRequest]] = []
        defer { buffers.forEach { $0.forEach { $0.deallocate() } } }

        for comm in comms {
            var mine: [UnsafeMutableRawBufferPointer] = []
            var issued: [CollectiveRequest] = []
            for k in 0..<inFlight {
                let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
                let floats = buffer.bindMemory(to: Float.self)
                for i in 0..<count { floats[i] = Float((comm.rank + 1) * (k + 1)) }
                mine.append(buffer)
                issued.append(comm.allReduceAsync(buffer, count: count, dataType: .float32, op: .sum))
            }
            buffers.append(mine)
            requests.append(issued)
        }

        for rankRequests in requests {
            for request in rankRequests { try request.wait() }
        }
        for rankBuffers in buffers {
            for (k, buffer) in rankBuffers.enumerated() {
                let floats = buffer.bindMemory(to: Float.self)
                let expected = Float(6 * (k + 1))
                for i in 0..<count {
                    XCTAssertEqual(floats[i], expected, accuracy: 1e-4, "slot \(k), element \(i)")
                }
            }
        }
    }

    func testWaitingTwiceIsSafeAndSaysTheSameThing() throws {
        let comms = try Communicator.loopbackGroup(worldSize: 2)
        defer { comms.forEach { $0.shutdown() } }
        let count = 16
        var buffers: [UnsafeMutableRawBufferPointer] = []
        defer { buffers.forEach { $0.deallocate() } }
        var requests: [CollectiveRequest] = []
        for comm in comms {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
            buffer.initializeMemory(as: Float.self, repeating: 1)
            buffers.append(buffer)
            requests.append(comm.allReduceAsync(buffer, count: count, dataType: .float32, op: .sum))
        }
        for request in requests {
            XCTAssertNoThrow(try request.wait())
            XCTAssertNoThrow(try request.wait(), "a completed request stays completed")
            XCTAssertTrue(request.isFinished)
        }
    }

    /// A failure travels with the request rather than being raised at issue:
    /// nothing has run yet when the call returns, so there is nothing to report
    /// until it has.
    func testAFailedCollectiveSurfacesFromWaitAndNotFromIssue() throws {
        // A single-rank communicator has no fabric, so the collective throws
        // .noFabric — but only once it runs.
        let comm = Communicator(rank: 0, worldSize: 2, topology: Topology.uniform(nodeCount: 2))
        let count = 8
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        defer { buffer.deallocate() }
        buffer.initializeMemory(as: Float.self, repeating: 1)

        let request = comm.allReduceAsync(buffer, count: count, dataType: .float32, op: .sum)
        XCTAssertThrowsError(try request.wait()) { error in
            guard case MCCLError.noFabric = error else {
                return XCTFail("expected .noFabric, got \(error)")
            }
        }
        XCTAssertTrue(request.isFinished)
    }

    func testTestingDoesNotBlockAndDoesNotConsumeTheCompletion() throws {
        let comms = try Communicator.loopbackGroup(worldSize: 2)
        defer { comms.forEach { $0.shutdown() } }
        let count = 32
        var buffers: [UnsafeMutableRawBufferPointer] = []
        defer { buffers.forEach { $0.deallocate() } }
        var requests: [CollectiveRequest] = []
        for comm in comms {
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
            buffer.initializeMemory(as: Float.self, repeating: 2)
            buffers.append(buffer)
            requests.append(comm.allReduceAsync(buffer, count: count, dataType: .float32, op: .sum))
        }
        // Whatever it reports, it has to report it immediately — the whole
        // point of the call is that it is not `wait`.
        let start = Date()
        for request in requests { _ = request.isFinished }
        XCTAssertLessThan(Date().timeIntervalSince(start), 1, "test must not block")

        for request in requests { try request.wait() }
        for request in requests {
            XCTAssertTrue(request.isFinished, "a waited request still reports finished")
        }
    }
}
