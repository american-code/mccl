import Foundation
import MCCL
import MCCLMLX
import MLX

// `mccltrain` — data-parallel training over mccl, and the demonstration that it
// is arithmetically identical to training on one machine.
//
// Three modes:
//
//   mccltrain --verify              correctness: one process trains the whole
//                                   batch, two ranks each train half and
//                                   all-reduce, the loss trajectories are
//                                   compared step by step
//   mccltrain --single              one rank, the whole batch — the control, and
//                                   the 1-node number in the throughput table
//   mccltrain --rank N --token ...  one rank of a real world across machines
//
// Deliberately not an XCTest target: the lab nodes have no XCTest, and the
// numbers that matter are measured on them.

let usage = """
mccltrain — data-parallel MLP training over mccl's MLX adapter

MODES
  --verify                 train single-process and 2-rank (TCP loopback) and
                           compare the loss trajectories. The adapter's
                           correctness proof; needs no cluster.
  --single                 one rank on the whole batch (the control / 1-node
                           throughput number)
  --rank <n>               join a real world as rank n (needs --token or
                           --token-file)

WORLD
  --world-size <n>         ranks in the world (default 2)
  --token <text>           join the world named by this token
  --token-file <path>      rank 0 writes the token here; other ranks poll it
  --emit-token             rank 0 prints MCCL_TOKEN=<text> first, for launchers
  --bind <host>            local address to bind *and* advertise. Always pass
                           this on a machine with more than one interface.
  --rendezvous-port <n>    fixed port for rank 0's rendezvous listener
  --timeout <seconds>      bring-up deadline (default 120)

MODEL AND DATA
  --inputs <n>             input features (default 32)
  --outputs <n>            output features (default 4)
  --hidden <a,b,...>       hidden layer widths (default 128,128)
  --rows <n>               synthetic dataset rows (default 4096)
  --batch <n>              GLOBAL batch per step; split evenly across ranks
                           (default 256)
  --steps <n>              training steps (default 200)
  --warmup <n>             untimed leading steps (default 5)
  --lr <f>                 SGD learning rate (default 0.05)
  --seed <n>               seed for weights and data (default 20260826)

WIRE
  --compression <scheme>   none | downcast | int8[:block] | topk:<fraction>
                           applied to the fused gradient buffer

OUTPUT
  --print-losses           print every step's world-mean loss
  --losses <path>          write the loss trajectory, one value per line
  --csv                    summary as CSV instead of a table
  --tolerance <f>          --verify: max |Δloss| accepted (default 1e-4)
  -h, --help               this text

The MLX Metal shader library must sit next to this binary. If it does not:
  Tools/fetch-metallib.sh
"""

// MARK: - Arguments

struct Options {
    var mode: Mode = .single
    var config = TrainConfig()
    var worldSize = 2
    var rank = 0
    var token: String?
    var tokenFile: String?
    var emitToken = false
    var bindHost: String?
    var rendezvousPort = 0
    var timeout: TimeInterval = 120
    var printLosses = false
    var lossesPath: String?
    var csv = false
    var tolerance: Float = 1e-4

    enum Mode { case verify, single, distributed }
}

func parseCompression(_ text: String) throws -> (WireCompression, String) {
    let parts = text.split(separator: ":", maxSplits: 1).map(String.init)
    switch parts[0].lowercased() {
    case "none":
        return (.none, "none")
    case "downcast":
        return (.downcast, "downcast")
    case "int8", "int8blockwise":
        let block = parts.count > 1 ? Int(parts[1]) : 256
        guard let block, block > 0 else {
            throw MCCLError.invalidArgument("--compression int8:<blockSize>: bad block size")
        }
        return (.int8Blockwise(blockSize: block), "int8/\(block)")
    case "topk":
        guard parts.count > 1, let fraction = Double(parts[1]), fraction > 0, fraction <= 1 else {
            throw MCCLError.invalidArgument("--compression topk:<fraction in (0,1]>")
        }
        return (.topK(fraction: fraction), "topk/\(fraction)")
    default:
        throw MCCLError.invalidArgument(
            "--compression: unknown scheme '\(text)' (none|downcast|int8[:block]|topk:<fraction>)")
    }
}

func parseArguments(_ arguments: [String]) throws -> Options {
    var options = Options()
    var sawRank = false
    var index = 0

    func value(_ flag: String) throws -> String {
        index += 1
        guard index < arguments.count else { throw MCCLError.invalidArgument("\(flag) needs a value") }
        return arguments[index]
    }
    func int(_ flag: String) throws -> Int {
        let text = try value(flag)
        guard let n = Int(text) else { throw MCCLError.invalidArgument("\(flag): '\(text)' is not an integer") }
        return n
    }
    func float(_ flag: String) throws -> Float {
        let text = try value(flag)
        guard let f = Float(text) else { throw MCCLError.invalidArgument("\(flag): '\(text)' is not a number") }
        return f
    }

    while index < arguments.count {
        let flag = arguments[index]
        switch flag {
        case "-h", "--help": print(usage); exit(0)
        case "--verify": options.mode = .verify
        case "--single": options.mode = .single
        case "--rank": options.rank = try int(flag); sawRank = true
        case "--world-size": options.worldSize = try int(flag)
        case "--token": options.token = try value(flag)
        case "--token-file": options.tokenFile = try value(flag)
        case "--emit-token": options.emitToken = true
        case "--bind": options.bindHost = try value(flag)
        case "--rendezvous-port": options.rendezvousPort = try int(flag)
        case "--timeout": options.timeout = TimeInterval(try int(flag))
        case "--inputs": options.config.inputs = try int(flag)
        case "--outputs": options.config.outputs = try int(flag)
        case "--hidden":
            let text = try value(flag)
            let widths = text.split(separator: ",").compactMap { Int($0) }
            guard widths.count == text.split(separator: ",").count, !widths.isEmpty else {
                throw MCCLError.invalidArgument("--hidden: expected a comma-separated list of widths")
            }
            options.config.hidden = widths
        case "--rows": options.config.rows = try int(flag)
        case "--batch": options.config.batch = try int(flag)
        case "--steps": options.config.steps = try int(flag)
        case "--warmup": options.config.warmup = try int(flag)
        case "--lr": options.config.learningRate = try float(flag)
        case "--seed": options.config.seed = UInt64(try int(flag))
        case "--compression":
            let (scheme, name) = try parseCompression(try value(flag))
            options.config.compression = scheme
            options.config.compressionName = name
        case "--print-losses": options.printLosses = true
        case "--losses": options.lossesPath = try value(flag)
        case "--csv": options.csv = true
        case "--tolerance": options.tolerance = try float(flag)
        default:
            throw MCCLError.invalidArgument("unknown flag '\(flag)' (try --help)")
        }
        index += 1
    }

    if sawRank || options.token != nil || options.tokenFile != nil {
        if options.mode != .verify { options.mode = .distributed }
    }
    if options.mode == .single { options.worldSize = 1 }

    let ranks = options.mode == .verify ? 2 : options.worldSize
    guard options.config.batch % ranks == 0 else {
        throw MCCLError.invalidArgument(
            "--batch \(options.config.batch) must divide evenly across \(ranks) ranks; "
            + "unequal shards would make the averaged gradient the wrong quantity")
    }
    guard options.config.steps > options.config.warmup else {
        throw MCCLError.invalidArgument("--steps must exceed --warmup")
    }
    return options
}

// MARK: - Runtime checks

/// MLX aborts the process rather than throwing when it cannot find its shader
/// library, so check before touching MLX at all and say what to do about it.
func requireMetallib() {
    var directories: [URL] = []
    if let exe = Bundle.main.executableURL?.deletingLastPathComponent() { directories.append(exe) }
    if let override = ProcessInfo.processInfo.environment["MCCL_MLX_METALLIB_DIR"] {
        directories.insert(URL(fileURLWithPath: override), at: 0)
    }
    let found = directories.contains {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent("mlx.metallib").path)
    }
    guard !found else { return }
    FileHandle.standardError.write(Data("""
        mccltrain: mlx.metallib not found next to this binary.

        mlx-swift compiles MLX's C++ core but does not build the Metal shader
        library — that is Xcode's job, and this binary is built with plain
        SwiftPM. Without it MLX cannot run a single GPU op.

        Fix:  Tools/fetch-metallib.sh
        Or:   Tools/fetch-metallib.sh <mlx-version> \(directories.first?.path ?? ".")

        Searched:
        \(directories.map { "  " + $0.path }.joined(separator: "\n"))

        """.utf8))
    exit(2)
}

// MARK: - Reporting

func describe(_ config: TrainConfig, worldSize: Int) -> String {
    let dimensions = ([config.inputs] + config.hidden + [config.outputs])
        .map(String.init).joined(separator: "-")
    return "MLP \(dimensions)  batch \(config.batch) (\(config.batch / max(1, worldSize))/rank)  "
        + "lr \(config.learningRate)  seed \(config.seed)  codec \(config.compressionName)"
}

func bytes(_ count: Int) -> String {
    if count >= 1 << 20 { return String(format: "%.2f MiB", Double(count) / Double(1 << 20)) }
    if count >= 1 << 10 { return String(format: "%.1f KiB", Double(count) / Double(1 << 10)) }
    return "\(count) B"
}

func summarise(_ result: TrainResult, label: String, options: Options) {
    if options.csv {
        print([
            label, String(options.worldSize), options.config.compressionName,
            String(result.timedSteps),
            String(format: "%.6f", result.seconds),
            String(format: "%.3f", result.stepsPerSecond),
            String(format: "%.4f", result.communicationFraction),
            String(result.parameterCount), String(result.gradientBytes),
            String(format: "%.6f", result.losses.first ?? .nan),
            String(format: "%.6f", result.losses.last ?? .nan),
        ].joined(separator: ","))
        return
    }
    print("")
    print("\(label)")
    print(String(repeating: "-", count: 64))
    print(String(format: "  parameters        %d (%@ per all-reduce)",
                 result.parameterCount, bytes(result.gradientBytes)))
    print(String(format: "  timed steps       %d in %.3f s", result.timedSteps, result.seconds))
    print(String(format: "  steps/sec         %.2f", result.stepsPerSecond))
    print(String(format: "  time in comms     %.1f%% (%.3f s)",
                 result.communicationFraction * 100, result.communicationSeconds))
    print(String(format: "  loss              %.6f -> %.6f",
                 result.losses.first ?? .nan, result.losses.last ?? .nan))
}

func writeLosses(_ losses: [Float], to path: String) throws {
    let text = losses.map { String(format: "%.9g", $0) }.joined(separator: "\n") + "\n"
    try Data(text.utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
}

// MARK: - Token plumbing
//
// Identical in shape to `mcclbench`'s: rank 0 creates the token (which binds the
// listener the token names, so it has to happen in rank 0's own process), and
// the other ranks are handed one or poll a file for it.

func obtainToken(_ options: Options) throws -> UniqueID {
    if let text = options.token {
        guard let id = UniqueID(text: text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw MCCLError.invalidArgument("--token: '\(text)' is not an mccl token")
        }
        return id
    }
    if options.rank == 0 {
        let host = options.bindHost ?? "0.0.0.0"
        let id = try Rendezvous.createUniqueID(
            host: host, port: options.rendezvousPort, advertisedHost: options.bindHost)
        if let path = options.tokenFile {
            try Data((id.text + "\n").utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
        }
        return id
    }
    guard let path = options.tokenFile else {
        throw MCCLError.invalidArgument("rank \(options.rank) needs --token or --token-file")
    }
    let deadline = Date().addingTimeInterval(options.timeout)
    repeat {
        if let text = try? String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8),
           let id = UniqueID(text: text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return id
        }
        usleep(50_000)
    } while Date() < deadline
    throw MCCLError.timedOut("no usable mccl token at \(path) after \(Int(options.timeout))s")
}

// MARK: - Main

do {
    let options = try parseArguments(Array(CommandLine.arguments.dropFirst()))
    requireMetallib()

    switch options.mode {
    case .verify:
        if !options.csv {
            print("mccltrain --verify")
            print("  \(describe(options.config, worldSize: 2))")
            print("  single process on the whole batch vs. 2 ranks over TCP loopback")
        }
        let report = try Verifier.run(config: options.config, tolerance: options.tolerance)
        summarise(report.single, label: "1 rank (control, whole batch)", options: options)
        summarise(report.distributed, label: "2 ranks (loopback, half batch each)", options: options)

        if options.printLosses {
            print("")
            print("  step        1-rank        2-rank          |diff|")
            for (i, pair) in zip(report.single.losses, report.distributed.losses).enumerated() {
                print(String(format: "  %4d  %12.8f  %12.8f  %14.3e",
                             i, pair.0, pair.1, abs(pair.0 - pair.1)))
            }
        }
        if let path = options.lossesPath { try writeLosses(report.distributed.losses, to: path) }

        print("")
        print("loss-trajectory equivalence over \(report.single.losses.count) steps")
        print(String(format: "  max |1-rank - 2-rank|   %.3e  (at step %d)",
                     report.maxAbsoluteDifference, report.worstStep))
        print(String(format: "  max relative difference %.3e", report.maxRelativeDifference))
        print(String(format: "  tolerance               %.3e", options.tolerance))
        if report.maxAbsoluteDifference <= options.tolerance {
            print("  PASS — the 2-rank run is the same computation as the 1-rank run.")
        } else {
            print("  FAIL — the trajectories diverge beyond tolerance.")
            exit(1)
        }

    case .single:
        if !options.csv {
            print("mccltrain --single")
            print("  \(describe(options.config, worldSize: 1))")
        }
        let comm = Trainer.soloCommunicator()
        defer { comm.shutdown() }
        let result = try Trainer.run(comm: comm, config: options.config)
        summarise(result, label: "1 rank (whole batch)", options: options)
        if options.printLosses {
            for (i, loss) in result.losses.enumerated() {
                print(String(format: "  %4d  %12.8f", i, loss))
            }
        }
        if let path = options.lossesPath { try writeLosses(result.losses, to: path) }

    case .distributed:
        guard options.worldSize > 1 else {
            throw MCCLError.invalidArgument("--rank needs --world-size > 1 (use --single for one rank)")
        }
        guard options.rank >= 0, options.rank < options.worldSize else {
            throw MCCLError.rankOutOfRange(rank: options.rank, worldSize: options.worldSize)
        }
        let id = try obtainToken(options)
        if options.emitToken, options.rank == 0 { print("MCCL_TOKEN=\(id.text)") }
        FileHandle.standardError.write(
            Data("mccltrain: rank \(options.rank)/\(options.worldSize) joining \(id.text)\n".utf8))

        let comm = try Communicator.join(
            uniqueID: id, rank: options.rank, worldSize: options.worldSize,
            bindHost: options.bindHost ?? "0.0.0.0", timeout: options.timeout)
        defer { comm.shutdown() }

        if options.rank == 0, !options.csv {
            print("mccltrain --rank 0 --world-size \(options.worldSize)")
            print("  \(describe(options.config, worldSize: options.worldSize))")
        }
        let result = try Trainer.run(comm: comm, config: options.config)

        // Only rank 0 prints, so a launcher scraping the run gets one report.
        if options.rank == 0 {
            summarise(result, label: "\(options.worldSize) ranks", options: options)
            if options.printLosses {
                for (i, loss) in result.losses.enumerated() {
                    print(String(format: "  %4d  %12.8f", i, loss))
                }
            }
            if let path = options.lossesPath { try writeLosses(result.losses, to: path) }
        } else {
            FileHandle.standardError.write(Data(String(
                format: "mccltrain: rank %d done, %.2f steps/s, final loss %.6f\n",
                options.rank, result.stepsPerSecond, result.losses.last ?? .nan).utf8))
        }
    }
} catch {
    FileHandle.standardError.write(Data("mccltrain: \(error)\n".utf8))
    exit(1)
}
