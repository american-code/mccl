import Foundation
import MCCL
import MCCLBenchmarks

// mcclbench — compare ring / tree / hierarchical all-reduce, with and without
// each wire codec, across message sizes. The harness lives in MCCLBenchmarks so
// the test suite can drive it directly; this file is argument handling and
// printing.
//
// Two modes, one table. Without --rank the whole world comes up inside this
// process, which measures the collective and the codecs but not a cable. With
// --rank / --world-size this process is one rank of a real world and the table
// describes whatever interconnect the machines share.

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("mcclbench: \(message)\n".utf8))
    exit(1)
}

func note(_ message: String) {
    FileHandle.standardError.write(Data("mcclbench: \(message)\n".utf8))
}

let options: BenchOptions
do {
    guard let parsed = try BenchArguments.parse(Array(CommandLine.arguments.dropFirst())) else {
        print(BenchArguments.usage)
        exit(0)
    }
    options = parsed
} catch {
    fail("\(error)\n\n\(BenchArguments.usage)")
}

// Only rank 0 owns stdout: the other ranks' numbers are agreed into rank 0's
// table anyway, and a launcher scraping this output wants exactly one copy.
let distributed = options.distributed
let isReporter = distributed?.rank ?? 0 == 0

func emit(_ line: String) {
    guard isReporter else { return }
    print(line)
}

if !options.csv, isReporter {
    let identity = NodeIdentity.local()
    if let distributed {
        print("mccl benchmark — \(options.collective.rawValue), \(distributed.worldSize) ranks over "
              + "\(distributed.label.map { "\($0), " } ?? "")distributed tcp, "
              + "\(options.dataType), op \(options.op)")
        print("rank 0 host: \(identity.hostname), \(identity.chip)"
              + (distributed.bindHost.map { ", bound \($0)" } ?? ""))
        print("wall time is the slowest rank's for each point")
    } else {
        print("mccl benchmark — \(options.collective.rawValue), \(options.worldSize) ranks over "
              + "\(options.transport.rawValue), \(options.dataType), op \(options.op)")
        print("host: \(identity.hostname), \(identity.chip)")
    }
    print("")
}

func summarise(_ rows: [BenchRow]) {
    guard !options.csv, isReporter else { return }
    // A closing summary: for each codec, the best bus bandwidth it reached and
    // where. That is the comparison the table exists to support.
    print("")
    print("best point per codec")
    var seen: [String] = []
    for row in rows where row.failure == nil {
        if !seen.contains(row.codec) { seen.append(row.codec) }
    }
    for codec in seen {
        let candidates = rows.filter { $0.codec == codec && $0.failure == nil }
        guard let best = candidates.max(by: { $0.busBytesPerSecond < $1.busBytesPerSecond }) else { continue }
        print("  \(BenchTable.pad(codec, 14))"
              + "\(BenchTable.pad(String(format: "%.3f GB/s bus", best.busBytesPerSecond / 1e9), 20))"
              + "at \(BenchTable.bytes(best.messageBytes)) on \(best.algorithm.rawValue)")
    }
    let failures = rows.filter { $0.failure != nil }
    if !failures.isEmpty {
        print("")
        print("\(failures.count) point(s) failed; see the rows marked —")
    }
}

if let distributed {
    // Distributed: join a real world and sweep the same grid. The header waits
    // until the world is up, because a token line has to reach the launcher
    // before rank 1 can be started.
    // The table header waits for the first row, so a bring-up that fails
    // leaves an error on stderr rather than a header with nothing under it.
    final class HeaderGate: @unchecked Sendable {
        private let lock = NSLock()
        private var printed = false
        func claim() -> Bool {
            lock.lock(); defer { lock.unlock() }
            if printed { return false }
            printed = true
            return true
        }
    }

    do {
        let header = HeaderGate()
        let rows = try await DistributedBenchRunner.run(
            options, distributed: distributed,
            onBringup: { bringup in
                guard bringup.rank == 0 else {
                    note("rank \(bringup.rank)/\(bringup.worldSize) joining \(bringup.uniqueID.text)")
                    return
                }
                if distributed.emitToken {
                    print("MCCL_TOKEN=\(bringup.uniqueID.text)")
                    fflush(stdout)
                }
                note("rank 0 token \(bringup.uniqueID.text)")
            },
            onRow: { row in
                guard isReporter else { return }
                if header.claim() {
                    print(BenchTable.header(options))
                    if let rule = BenchTable.rule(options) { print(rule) }
                }
                print(BenchTable.render(row, options: options))
                fflush(stdout)
            })
        summarise(rows)
        if !isReporter {
            note("rank \(distributed.rank) finished \(rows.count) point(s)")
        }
    } catch {
        fail("\(error)")
    }
} else {
    print(BenchTable.header(options))
    if let rule = BenchTable.rule(options) { print(rule) }

    do {
        let rows = try await BenchRunner.run(options) { row in
            print(BenchTable.render(row, options: options))
            fflush(stdout)
        }
        guard !options.csv else { exit(0) }
        summarise(rows)
    } catch {
        fail("\(error)")
    }
}
