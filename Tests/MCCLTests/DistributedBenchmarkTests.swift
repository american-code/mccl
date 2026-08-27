import XCTest
@testable import MCCL
@testable import MCCLBenchmarks

/// The multi-process half of `mcclbench`.
///
/// Most of this is arithmetic and argument handling, which is checked in
/// process. The part that cannot be faked — that two real `mcclbench`
/// processes find each other through a token, walk the same grid, and leave
/// exactly one table on rank 0's stdout — is checked by spawning them.
final class DistributedBenchmarkTests: XCTestCase {

    // MARK: - Argument handling

    func testDistributedFlagsBuildAWorld() throws {
        let options = try XCTUnwrap(try BenchArguments.parse([
            "--rank", "1", "--world-size", "4", "--token", "mccl1:00000000000000ff:10.0.0.1:7000",
            "--bind", "169.254.23.203", "--label", "thunderbolt", "--timeout", "45",
        ]))
        let distributed = try XCTUnwrap(options.distributed)
        XCTAssertEqual(distributed.rank, 1)
        XCTAssertEqual(distributed.worldSize, 4)
        XCTAssertEqual(distributed.bindHost, "169.254.23.203")
        XCTAssertEqual(distributed.label, "thunderbolt")
        XCTAssertEqual(distributed.timeout, 45)
        // The distributed world size is the one the table's bus factor uses.
        XCTAssertEqual(options.worldSize, 4)
    }

    func testInProcessModeIsUntouchedWithoutDistributedFlags() throws {
        let options = try XCTUnwrap(try BenchArguments.parse(["--ranks", "4", "--quick"]))
        XCTAssertNil(options.distributed)
        XCTAssertEqual(options.worldSize, 4)
        XCTAssertNil(options.explicitSizes)
    }

    func testJoiningRankNeedsAWayToFindRankZero() {
        XCTAssertThrowsError(try BenchArguments.parse(["--rank", "1", "--world-size", "2"])) { error in
            XCTAssertTrue("\(error)".contains("--token"), "\(error)")
        }
    }

    func testRankZeroCannotBeHandedSomeoneElsesToken() {
        XCTAssertThrowsError(try BenchArguments.parse([
            "--rank", "0", "--world-size", "2", "--token", "mccl1:00000000000000ff:10.0.0.1:7000",
        ]))
    }

    func testRankOutsideTheWorldIsRejected() {
        XCTAssertThrowsError(try BenchArguments.parse([
            "--rank", "4", "--world-size", "4", "--token-file", "/tmp/x",
        ]))
    }

    func testLoopbackTransportIsRejectedForADistributedRun() {
        XCTAssertThrowsError(try BenchArguments.parse([
            "--rank", "0", "--world-size", "2", "--transport", "loopback",
        ]))
    }

    // MARK: - Size lists

    func testExplicitSizesReplaceTheSweep() throws {
        let options = try XCTUnwrap(try BenchArguments.parse(["--sizes", "64K,1M,64M"]))
        XCTAssertEqual(options.sizes, [65536, 1 << 20, 64 << 20])
    }

    func testSizeSuffixesAreAll1024Based() throws {
        let options = try XCTUnwrap(try BenchArguments.parse(["--sizes", "512,4KB,4KiB,2MB,1G,8b"]))
        XCTAssertEqual(options.sizes, [512, 4096, 4096, 2 << 20, 1 << 30, 8])
    }

    func testNonsenseSizesAreRejected() {
        XCTAssertThrowsError(try BenchArguments.parse(["--sizes", "64Q"]))
        XCTAssertThrowsError(try BenchArguments.parse(["--sizes", "0"]))
        XCTAssertThrowsError(try BenchArguments.parse(["--sizes", ""]))
    }

    // MARK: - Cross-rank agreement

    func testFingerprintDependsOnTheShapeOfTheSweepOnly() {
        var base = BenchOptions()
        base.explicitSizes = [1024, 4096]
        let reference = DistributedBenchRunner.fingerprint(base)

        var same = base
        same.csv = true                 // output format is nobody else's business
        same.transport = .loopback
        XCTAssertEqual(DistributedBenchRunner.fingerprint(same), reference)

        for mutate in [
            { (o: inout BenchOptions) in o.explicitSizes = [1024, 8192] },
            { (o: inout BenchOptions) in o.algorithms = [.ring] },
            { (o: inout BenchOptions) in o.codecs = [.none] },
            { (o: inout BenchOptions) in o.collective = .allGather },
            { (o: inout BenchOptions) in o.dataType = .float16 },
            { (o: inout BenchOptions) in o.op = .max },
            { (o: inout BenchOptions) in o.worldSize = 3 },
            { (o: inout BenchOptions) in o.byteBudget = 1 << 10 },
        ] {
            var changed = base
            mutate(&changed)
            XCTAssertNotEqual(DistributedBenchRunner.fingerprint(changed), reference)
        }
    }

    func testAgreementPassesWhenEveryRankWasLaunchedTheSame() async throws {
        var options = BenchOptions()
        options.worldSize = 3
        options.explicitSizes = [1024]
        let comms = try Communicator.loopbackGroup(worldSize: 3)
        defer { comms.forEach { $0.shutdown() } }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for comm in comms {
                group.addTask { try await DistributedBenchRunner.checkAgreement(comm: comm, options: options) }
            }
            try await group.waitForAll()
        }
    }

    func testAgreementFailsLoudlyRatherThanDeadlocking() async throws {
        // Rank 1 was launched with a different grid — the sort of mistake that
        // otherwise hangs the whole job on the first mismatched collective.
        var mine = BenchOptions()
        mine.worldSize = 2
        mine.explicitSizes = [1024]
        var theirs = mine
        theirs.explicitSizes = [4096]

        let comms = try Communicator.loopbackGroup(worldSize: 2)
        defer { comms.forEach { $0.shutdown() } }
        var failures = 0
        await withTaskGroup(of: Bool.self) { group in
            for (index, comm) in comms.enumerated() {
                let options = index == 0 ? mine : theirs
                group.addTask {
                    do {
                        try await DistributedBenchRunner.checkAgreement(comm: comm, options: options)
                        return false
                    } catch { return true }
                }
            }
            for await failed in group where failed { failures += 1 }
        }
        XCTAssertEqual(failures, 2, "a mismatched sweep must fail on every rank, not hang")
    }

    // MARK: - Timing agreement

    func testRetimeRescalesEverythingDerivedFromTheClock() {
        var options = BenchOptions()
        options.worldSize = 2
        options.collective = .allReduce
        let row = BenchRow(
            messageBytes: 1 << 20, algorithm: .ring, codec: "none", iterations: 8,
            seconds: 0.001, algorithmicBytesPerSecond: Double(1 << 20) / 0.001,
            busBytesPerSecond: 0, wireBytesPerRank: nil, failure: nil)

        let slower = DistributedBenchRunner.retime(row, seconds: 0.002, options: options)
        XCTAssertEqual(slower.seconds, 0.002)
        XCTAssertEqual(slower.algorithmicBytesPerSecond, Double(1 << 20) / 0.002, accuracy: 1)
        // Two ranks: the bus factor is 2(n-1)/n = 1.
        XCTAssertEqual(slower.busBytesPerSecond, slower.algorithmicBytesPerSecond, accuracy: 1)
        XCTAssertEqual(slower.iterations, row.iterations)
        XCTAssertEqual(slower.messageBytes, row.messageBytes)
    }

    func testSlowestRankIsTheOneReported() async throws {
        let comms = try Communicator.loopbackGroup(worldSize: 3)
        defer { comms.forEach { $0.shutdown() } }
        let times = [0.004, 0.011, 0.007]
        try await withThrowingTaskGroup(of: Double.self) { group in
            for (index, comm) in comms.enumerated() {
                group.addTask { try await DistributedBenchRunner.agreeMax(comm, times[index]) }
            }
            for try await agreed in group {
                XCTAssertEqual(agreed, 0.011, accuracy: 1e-6)
            }
        }
    }

    // MARK: - Token plumbing

    func testTokenFileIsPolledUntilRankZeroWritesIt() throws {
        let path = NSTemporaryDirectory() + "mccl-token-\(UUID().uuidString).id"
        defer { try? FileManager.default.removeItem(atPath: path) }
        let expected = UniqueID(nonce: 0xfeed, address: PeerAddress(host: "169.254.152.222", port: 7331))
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
            try? Data((expected.text + "\n").utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
        }
        let found = try DistributedBenchRunner.waitForToken(path: path, timeout: 10)
        XCTAssertEqual(found, expected)
    }

    func testMissingTokenFileTimesOutRatherThanBlockingForever() {
        let path = NSTemporaryDirectory() + "mccl-absent-\(UUID().uuidString).id"
        XCTAssertThrowsError(try DistributedBenchRunner.waitForToken(path: path, timeout: 0.3))
    }

    func testRankZeroTokenAdvertisesTheAddressItWasBoundTo() throws {
        // The whole point of --bind on a multi-homed machine: peers must be
        // told the cable under test, not whichever interface the host prefers.
        var distributed = DistributedOptions()
        distributed.rank = 0
        distributed.bindHost = "127.0.0.1"
        let id = try DistributedBenchRunner.obtainToken(distributed)
        defer { Rendezvous.discard(id) }
        XCTAssertEqual(id.address.host, "127.0.0.1")
        XCTAssertGreaterThan(id.address.port, 0)
        XCTAssertEqual(UniqueID(text: id.text), id)
    }

    // MARK: - Real processes

    func testTwoProcessesOverLoopbackProduceOneTableOnRankZero() throws {
        let run = try spawnWorld(
            worldSize: 2,
            arguments: ["--sizes", "16K,64K", "--algorithms", "ring", "--codecs", "none,downcast",
                        "--warmup", "1", "--min-iters", "2", "--max-iters", "4", "--csv"])

        // 2 sizes x 1 algorithm x 2 codecs, plus the CSV header.
        let lines = run.stdout[0].split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 5, "rank 0 stdout:\n\(run.stdout[0])")
        XCTAssertEqual(lines[0], BenchTable.header({ var o = BenchOptions(); o.csv = true; return o }()))
        for line in lines.dropFirst() {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            XCTAssertEqual(fields.count, 8, line)
            XCTAssertEqual(fields[1], "ring", line)
            XCTAssertTrue(["none", "downcast"].contains(fields[2]), line)
            XCTAssertGreaterThan(Double(fields[4]) ?? 0, 0, "wall time in \(line)")
            XCTAssertGreaterThan(Double(fields[5]) ?? 0, 0, "throughput in \(line)")
            // Two ranks: bus == algorithmic, because 2(n-1)/n is 1.
            XCTAssertEqual(Double(fields[5]) ?? 0, Double(fields[6]) ?? -1, accuracy: 1e-3, line)
        }
        XCTAssertEqual(Set(lines.dropFirst().map { $0.split(separator: ",")[0] }),
                       ["16384", "65536"])

        // Rank 1 stays off stdout so a launcher scrapes exactly one table.
        XCTAssertTrue(run.stdout[1].isEmpty, "rank 1 stdout:\n\(run.stdout[1])")
        XCTAssertTrue(run.stderr[1].contains("finished 4 point(s)"), run.stderr[1])
    }

    func testFourProcessesSweepTheSameGridAsTheInProcessRunner() throws {
        let arguments = ["--sizes", "16K", "--algorithms", "ring,tree", "--codecs", "none",
                         "--warmup", "1", "--min-iters", "2", "--max-iters", "3", "--csv"]
        let run = try spawnWorld(worldSize: 4, arguments: arguments)
        let rows = run.stdout[0].split(separator: "\n").dropFirst().map(String.init)
        XCTAssertEqual(rows.count, 2, "1 size x 2 algorithms x 1 codec\n\(run.stdout[0])")
        XCTAssertEqual(rows.map { $0.split(separator: ",")[1] }, ["ring", "tree"])
        for row in rows {
            let fields = row.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            // Four ranks: the bus figure is 1.5x the algorithmic one.
            XCTAssertEqual((Double(fields[6]) ?? 0) / (Double(fields[5]) ?? 1), 1.5, accuracy: 1e-2, row)
        }
        for rank in 1..<4 {
            XCTAssertTrue(run.stdout[rank].isEmpty, "rank \(rank) printed a table")
        }
    }

    func testRankZeroEmitsAScrapableToken() throws {
        let run = try spawnWorld(
            worldSize: 2,
            arguments: ["--sizes", "4K", "--algorithms", "ring", "--codecs", "none",
                        "--warmup", "0", "--min-iters", "1", "--max-iters", "1", "--csv"],
            emitToken: true)
        let first = try XCTUnwrap(run.stdout[0].split(separator: "\n").first.map(String.init))
        XCTAssertTrue(first.hasPrefix("MCCL_TOKEN=mccl1:"), first)
        let id = UniqueID(text: String(first.dropFirst("MCCL_TOKEN=".count)))
        XCTAssertEqual(id?.address.host, "127.0.0.1")
    }

    // MARK: - Spawning

    private struct WorldRun {
        var stdout: [String]
        var stderr: [String]
    }

    /// Runs `worldSize` real `mcclbench` processes against one another over
    /// loopback and returns each one's output. Fails the test on a non-zero
    /// exit or a rank that outlives the deadline.
    private func spawnWorld(
        worldSize: Int, arguments: [String], emitToken: Bool = false,
        timeout: TimeInterval = 120, file: StaticString = #filePath, line: UInt = #line
    ) throws -> WorldRun {
        let executable = try benchExecutable()
        let tokenPath = NSTemporaryDirectory() + "mccl-bench-\(UUID().uuidString).id"
        defer { try? FileManager.default.removeItem(atPath: tokenPath) }

        var processes: [Process] = []
        var outPipes: [Pipe] = []
        var errPipes: [Pipe] = []
        var collected = [Data](repeating: Data(), count: worldSize * 2)
        let lock = NSLock()

        for rank in 0..<worldSize {
            let process = Process()
            process.executableURL = executable
            process.arguments = ["--rank", String(rank), "--world-size", String(worldSize),
                                 "--token-file", tokenPath, "--bind", "127.0.0.1",
                                 "--timeout", String(Int(timeout))]
                + (rank == 0 && emitToken ? ["--emit-token"] : [])
                + arguments
            let out = Pipe(), err = Pipe()
            process.standardOutput = out
            process.standardError = err
            // Drain as the ranks run: a full pipe buffer would wedge a rank
            // mid-sweep and the whole world with it.
            for (index, handle) in [(rank, out.fileHandleForReading),
                                    (worldSize + rank, err.fileHandleForReading)] {
                handle.readabilityHandler = { source in
                    let chunk = source.availableData
                    guard !chunk.isEmpty else { return }
                    lock.lock(); collected[index].append(chunk); lock.unlock()
                }
            }
            try process.run()
            processes.append(process)
            outPipes.append(out)
            errPipes.append(err)
        }

        let deadline = Date().addingTimeInterval(timeout)
        for process in processes {
            while process.isRunning, Date() < deadline { usleep(20_000) }
            if process.isRunning {
                processes.forEach { $0.terminate() }
                XCTFail("a rank was still running after \(Int(timeout))s", file: file, line: line)
                break
            }
        }
        // Let the readability handlers flush the tail before tearing them down.
        usleep(200_000)
        for pipe in outPipes + errPipes { pipe.fileHandleForReading.readabilityHandler = nil }

        lock.lock()
        let texts = collected.map { String(decoding: $0, as: UTF8.self) }
        lock.unlock()
        for (rank, process) in processes.enumerated() {
            XCTAssertEqual(process.terminationStatus, 0,
                           "rank \(rank) exited \(process.terminationStatus): \(texts[worldSize + rank])",
                           file: file, line: line)
        }
        return WorldRun(stdout: Array(texts[0..<worldSize]),
                        stderr: Array(texts[worldSize...]))
    }

    /// The `mcclbench` binary sitting next to this test bundle.
    private func benchExecutable() throws -> URL {
        let directory = Bundle(for: type(of: self)).bundleURL.deletingLastPathComponent()
        let candidate = directory.appendingPathComponent("mcclbench")
        guard FileManager.default.isExecutableFile(atPath: candidate.path) else {
            throw XCTSkip("mcclbench is not built at \(candidate.path); run `swift build` first")
        }
        return candidate
    }
}
