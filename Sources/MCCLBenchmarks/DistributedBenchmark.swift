import Foundation
import MCCL

// The multi-process / multi-machine half of `mcclbench`.
//
// The in-process runner (`BenchRunner`) brings up every rank inside one
// process, which measures the collective but not the interconnect. This runner
// is one rank of a real world: it joins through the same `Rendezvous` token the
// C shim's `mcclCommInitRank` uses, sweeps exactly the same grid of points, and
// leaves rank 0 to print the table.
//
// Two properties matter for the numbers to mean anything:
//
//  * Every rank must walk the same (size x algorithm x codec) sequence in the
//    same order, or the ranks deadlock against each other. The grid is a pure
//    function of the options, so agreement is checked once at bring-up with a
//    fingerprint all-reduce rather than hoped for.
//  * The reported wall time is the *slowest* rank's, agreed after each point
//    with a one-element max all-reduce. Rank 0's own clock would understate the
//    cost whenever rank 0 happens to be the rank that waits.

// MARK: - Options

/// The distributed half of a `BenchOptions`. Present only when `mcclbench` was
/// asked to run as one rank of a real world.
public struct DistributedOptions: Sendable {
    public var rank = 0
    public var worldSize = 2
    /// The token text, for ranks that were handed one on the command line.
    public var token: String?
    /// A file rank 0 writes the token to and other ranks poll for. The robust
    /// channel when the launcher is `ssh` and stdout is being captured.
    public var tokenFile: String?
    /// Print `MCCL_TOKEN=<text>` on stdout before anything else, for a launcher
    /// that scrapes it.
    public var emitToken = false
    /// The local address to bind *and* advertise. Set this on a multi-homed
    /// machine: without it a wildcard bind advertises whichever interface
    /// `NetworkInterfaces.preferredLocalAddress()` likes, which may not be the
    /// cable you meant to measure.
    public var bindHost: String?
    /// Fixed rendezvous port for rank 0. 0 picks an ephemeral one.
    public var rendezvousPort = 0
    public var timeout: TimeInterval = 120
    /// Free-form name for the fabric under test, echoed into the header.
    public var label: String?

    public init() {}

    var isRoot: Bool { rank == 0 }
}

// MARK: - Runner

public enum DistributedBenchRunner {

    /// A point that failed on this rank. Fatal in distributed mode: the ranks
    /// have already diverged, so the next collective would hang rather than
    /// report anything.
    public struct PointFailure: Error, CustomStringConvertible {
        public let message: String
        public var description: String { message }
    }

    /// What rank 0 published, so a caller (or a test) can see the token without
    /// scraping stdout.
    public struct Bringup: Sendable {
        public let uniqueID: UniqueID
        public let rank: Int
        public let worldSize: Int
    }

    /// Joins the world, sweeps the grid, and returns the rows.
    ///
    /// Every rank returns rows; they are identical by construction, because the
    /// timings are agreed across ranks. `onRow` fires on every rank too — the
    /// command-line front end is what decides only rank 0 prints.
    public static func run(
        _ options: BenchOptions,
        distributed: DistributedOptions,
        onBringup: (@Sendable (Bringup) -> Void)? = nil,
        onRow: (@Sendable (BenchRow) -> Void)? = nil
    ) async throws -> [BenchRow] {
        var options = options
        options.worldSize = distributed.worldSize
        guard distributed.worldSize > 1 else {
            throw MCCLError.invalidArgument("distributed benchmark needs at least 2 ranks")
        }
        guard distributed.rank >= 0, distributed.rank < distributed.worldSize else {
            throw MCCLError.rankOutOfRange(rank: distributed.rank, worldSize: distributed.worldSize)
        }

        let id = try obtainToken(distributed)
        onBringup?(Bringup(uniqueID: id, rank: distributed.rank, worldSize: distributed.worldSize))

        let comm = try Communicator.join(
            uniqueID: id, rank: distributed.rank, worldSize: distributed.worldSize,
            bindHost: distributed.bindHost ?? "0.0.0.0",
            timeout: distributed.timeout)
        defer { comm.shutdown() }

        // A mismatched grid deadlocks rather than fails, so rule it out first.
        try await checkAgreement(comm: comm, options: options)

        var rows: [BenchRow] = []
        for size in options.sizes {
            for algorithm in options.algorithms {
                guard let plan = algorithm.plan(worldSize: options.worldSize) else { continue }
                comm.planOverride = plan
                for codec in options.codecs {
                    guard codec.applies(to: options.collective, dataType: options.dataType) else { continue }
                    var row = await BenchRunner.measure(
                        size: size, algorithm: algorithm, codec: codec,
                        comms: [comm], options: options,
                        barrier: { try await barrier(comm) })

                    // Agree the timing before anyone reports it. A failure on
                    // any rank is fatal here: the ranks have already diverged,
                    // and continuing would hang on the next collective.
                    if row.failure != nil {
                        rows.append(row)
                        onRow?(row)
                        throw PointFailure(
                            message: "rank \(distributed.rank) failed at \(BenchTable.bytes(size)) "
                            + "\(algorithm.rawValue)/\(codec.description): \(row.failure!)")
                    }
                    let slowest = try await agreeMax(comm, row.seconds)
                    row = retime(row, seconds: slowest, options: options)
                    rows.append(row)
                    onRow?(row)
                }
            }
        }
        return rows
    }

    // MARK: - Token plumbing

    /// Rank 0 creates the token (binding the rendezvous listener as a side
    /// effect); everyone else is handed one, or waits for the file to appear.
    static func obtainToken(_ distributed: DistributedOptions) throws -> UniqueID {
        if let text = distributed.token {
            guard let id = UniqueID(text: text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                throw MCCLError.invalidArgument("--token: '\(text)' is not an mccl token")
            }
            return id
        }
        if distributed.isRoot {
            // `createUniqueID` must run in rank 0's own process: it binds the
            // listener the token names. Binding the explicit host is what pins
            // the rendezvous — and therefore the advertised address — to one
            // cable on a multi-homed machine.
            let host = distributed.bindHost ?? "0.0.0.0"
            let id = try Rendezvous.createUniqueID(
                host: host, port: distributed.rendezvousPort,
                advertisedHost: distributed.bindHost)
            if let path = distributed.tokenFile {
                try Data((id.text + "\n").utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
            }
            return id
        }
        guard let path = distributed.tokenFile else {
            throw MCCLError.invalidArgument("rank \(distributed.rank) needs --token or --token-file")
        }
        return try waitForToken(path: path, timeout: distributed.timeout)
    }

    /// Polls a token file. Rank 0 may not have been launched yet, and an
    /// `ssh`-started rank 1 routinely wins that race.
    static func waitForToken(path: String, timeout: TimeInterval) throws -> UniqueID {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if let text = try? String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8),
               let id = UniqueID(text: text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return id
            }
            usleep(50_000)
        } while Date() < deadline
        throw MCCLError.timedOut("no usable mccl token at \(path) after \(Int(timeout))s")
    }

    // MARK: - Cross-rank agreement

    /// Fails loudly when two ranks were launched with different sweeps.
    ///
    /// The grid is a pure function of the options, so one number covers it: if
    /// every rank's fingerprint survives both a min and a max all-reduce
    /// unchanged, every rank is about to walk the same sequence of points.
    static func checkAgreement(comm: Communicator, options: BenchOptions) async throws {
        let mine = fingerprint(options)
        let low = try await reduceInt32(comm, mine, op: .min)
        let high = try await reduceInt32(comm, mine, op: .max)
        guard low == mine, high == mine else {
            throw MCCLError.invalidArgument(
                "rank \(comm.rank) was launched with a different sweep than its peers "
                + "(fingerprint \(mine), world spans \(low)...\(high)) — every rank needs "
                + "identical --sizes/--algorithms/--codecs/--collective/--dtype/--op")
        }
    }

    /// A 31-bit digest of everything that changes the shape of the sweep.
    /// Deliberately not the timing knobs' business — iteration counts follow
    /// from the sizes, which are included.
    static func fingerprint(_ options: BenchOptions) -> Int32 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ value: some Hashable) {
            for byte in String(describing: value).utf8 {
                hash = (hash ^ UInt64(byte)) &* 0x1000_0000_01b3
            }
            hash = (hash ^ 0x5f) &* 0x1000_0000_01b3
        }
        mix(options.worldSize)
        mix(options.collective.rawValue)
        mix(options.dataType)
        mix(options.op)
        for size in options.sizes { mix(size) }
        for algorithm in options.algorithms { mix(algorithm.rawValue) }
        for codec in options.codecs { mix(codec.description) }
        mix(options.warmup)
        mix(options.minIterations)
        mix(options.maxIterations)
        mix(options.byteBudget)
        return Int32(truncatingIfNeeded: hash & 0x7fff_ffff)
    }

    /// Lines every rank up. One element, so it costs a round trip and nothing
    /// else; used to keep a rank's setup out of another rank's stopwatch.
    static func barrier(_ comm: Communicator) async throws {
        _ = try await reduceInt32(comm, 1, op: .sum)
    }

    /// The slowest rank's time for the point just measured.
    static func agreeMax(_ comm: Communicator, _ seconds: Double) async throws -> Double {
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: MemoryLayout<Float>.size, alignment: 16)
        defer { buffer.deallocate() }
        buffer.bindMemory(to: Float.self)[0] = Float(seconds)
        try await comm.allReduce(buffer, count: 1, dataType: .float32, op: .max)
        return Double(buffer.bindMemory(to: Float.self)[0])
    }

    private static func reduceInt32(_ comm: Communicator, _ value: Int32, op: ReduceOp) async throws -> Int32 {
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: MemoryLayout<Int32>.size, alignment: 16)
        defer { buffer.deallocate() }
        buffer.bindMemory(to: Int32.self)[0] = value
        try await comm.allReduce(buffer, count: 1, dataType: .int32, op: op)
        return buffer.bindMemory(to: Int32.self)[0]
    }

    /// Rebuilds a row around an agreed wall time. Everything else about the
    /// point — bytes, iterations, codec — is identical on every rank.
    static func retime(_ row: BenchRow, seconds: Double, options: BenchOptions) -> BenchRow {
        guard seconds > 0 else { return row }
        let algorithmic = Double(row.messageBytes) / seconds
        return BenchRow(
            messageBytes: row.messageBytes, algorithm: row.algorithm, codec: row.codec,
            iterations: row.iterations, seconds: seconds,
            algorithmicBytesPerSecond: algorithmic,
            busBytesPerSecond: algorithmic * options.collective.busFactor(worldSize: options.worldSize),
            wireBytesPerRank: row.wireBytesPerRank, failure: nil)
    }
}
