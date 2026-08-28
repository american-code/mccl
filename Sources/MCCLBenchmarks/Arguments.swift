import Foundation
import MCCL

/// Command-line parsing for `mcclbench`, in the library target so it is
/// testable without spawning a process.
public enum BenchArguments {

    public struct ParseError: Error, CustomStringConvertible {
        public let message: String
        public var description: String { message }
    }

    public static let usage = """
    mcclbench — mccl collective benchmark

    USAGE
      mcclbench [options]

    Brings up <ranks> ranks in this process over loopback sockets and sweeps
    every (message size x algorithm x codec) point, reporting the wall time and
    the throughput each one achieved.

    With --rank / --world-size it instead runs as *one rank of a real world*,
    joining the others through a rendezvous token — one process per machine,
    over whatever cable those machines share. Same grid, same table.

    "GB/s alg" is payload bytes over wall time — what the caller sees move.
    "GB/s bus" scales that by the traffic each link actually carries
    (2(n-1)/n for all-reduce), which is the number comparable to NCCL's busbw
    and to MLX's ring.

    OPTIONS
      --ranks <n>            ranks in the world (default 4)
      --min-bytes <n>        smallest message (default 1024)
      --max-bytes <n>        largest message (default 67108864)
      --step <n>             size multiplier between points (default 4)
      --sizes <list>         explicit sizes instead of the sweep, e.g.
                             64K,1M,64M (K/M/G are 1024-based; KB/MB likewise)
      --collective <name>    allreduce | allgather | broadcast | reducescatter
      --algorithms <list>    comma-separated: ring,tree,hierarchical
      --codecs <list>        comma-separated: none,downcast,int8[:block],topk[:fraction]
      --dtype <name>         fp32 | fp16 | bf16 | i32 | i8 (default fp32)
      --op <name>            sum | prod | min | max | avg (default sum)
      --transport <name>     tcp | loopback | rdma (default tcp)
                             rdma needs --distributed and Thunderbolt 5; see docs/RDMA.md
      --warmup <n>           untimed iterations per point (default 2)
      --min-iters <n>        floor on timed iterations (default 3)
      --max-iters <n>        ceiling on timed iterations (default 50)
      --budget <n>           bytes to move per point, sets the iteration count
                             (default 134217728)
      --quick                1 KiB .. 1 MiB, ring and tree only
      --csv                  machine-readable output
      --codec-bench          measure the codec kernels alone — no ranks, no
                             sockets — and print each codec's encode/decode
                             ceiling in payload GB/s. This is the machine half
                             of the rule in docs/ARCHITECTURE.md §Measured: a
                             codec pays only where the fabric's uncompressed
                             all-reduce rate is below the codec's own ceiling.
      --help                 this text

    DISTRIBUTED OPTIONS (one process per rank, one rank per machine)
      --rank <n>             this process's rank; implies distributed mode
      --world-size <n>       ranks in the world (default 2)
      --token <text>         join the world named by this token
      --token-file <path>    rank 0 writes the token here; other ranks poll it
      --emit-token           rank 0 prints MCCL_TOKEN=<text> on stdout first
      --bind <hosts>         local address(es) to advertise, comma-separated;
                             the flag may also be repeated. One address pins
                             the run to that cable (and binds it). Two or more
                             bind the wildcard and advertise all of them, and
                             each *pair* of ranks then dials the best path the
                             two of them share. Omitted: every usable local
                             address is discovered and advertised.
      --rendezvous-port <n>  fixed port for rank 0's rendezvous (default: any)
      --timeout <seconds>    bring-up deadline (default 120)
      --label <name>         name for the fabric, echoed into the header

    Rank 0 creates the token and must be started first; the others join with
    --token or --token-file.

    Pass one --bind address when you mean to *measure* a specific cable: it is
    the only way to be sure the traffic crossed the link named in the table.
    Pass several — or none — when you mean to *use* whatever the machines share,
    which is what a mixed fabric needs.

    Pinned to one cable, two machines over Thunderbolt:

      studio-a$ mcclbench --rank 0 --world-size 2 --bind 169.254.152.222 \\
                          --token-file /tmp/tb.id --sizes 64K,1M,64M
      studio-b$ mcclbench --rank 1 --world-size 2 --bind 169.254.23.203 \\
                          --token <the token rank 0 printed>

    Mixed: two machines on a Thunderbolt cable plus a laptop on Wi-Fi. The
    studios advertise both of their addresses, so they reach each other over
    Thunderbolt and the laptop over the LAN, in one world.

      studio-a$ mcclbench --rank 0 --world-size 3 \\
                          --bind 169.254.152.222,192.168.1.250 --token-file /tmp/mix.id
      studio-b$ mcclbench --rank 1 --world-size 3 \\
                          --bind 169.254.23.203,192.168.1.238 --token <token>
      laptop$   mcclbench --rank 2 --world-size 3 --token <token>

    Every rank must be given the same grid — the sweep is checked against its
    peers at bring-up and a mismatch is an error rather than a deadlock. Only
    rank 0 prints the table. The wall time in it is the *slowest* rank's for
    each point, agreed with a one-element max all-reduce once the point is
    done, so it is the time the collective actually took rather than the time
    rank 0 happened to spend inside it.

    Sizes are per collective call, in bytes. The defaults sweep 1 KiB .. 64 MiB
    over 4 ranks and take well under a minute; --quick is faster still.
    """

    public static func parse(_ arguments: [String]) throws -> BenchOptions? {
        var options = BenchOptions()
        var distributed = DistributedOptions()
        var isDistributed = false
        var index = 0

        func next(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else {
                throw ParseError(message: "\(flag) needs a value")
            }
            return arguments[index]
        }
        func integer(_ flag: String) throws -> Int {
            let raw = try next(flag)
            guard let value = Int(raw) else {
                throw ParseError(message: "\(flag): '\(raw)' is not a number")
            }
            return value
        }

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--help", "-h":
                return nil
            case "--ranks":
                options.worldSize = try integer(argument)
            case "--min-bytes":
                options.minBytes = try integer(argument)
            case "--max-bytes":
                options.maxBytes = try integer(argument)
            case "--step":
                options.sizeStep = try integer(argument)
            case "--sizes":
                options.explicitSizes = try parseSizes(try next(argument))
            case "--warmup":
                options.warmup = try integer(argument)
            case "--min-iters":
                options.minIterations = try integer(argument)
            case "--max-iters":
                options.maxIterations = try integer(argument)
            case "--budget":
                options.byteBudget = try integer(argument)
            case "--collective":
                let raw = try next(argument)
                guard let value = BenchCollective(rawValue: raw.lowercased()) else {
                    throw ParseError(message: "unknown collective '\(raw)'")
                }
                options.collective = value
            case "--transport":
                let raw = try next(argument)
                guard let value = BenchTransport(rawValue: raw.lowercased()) else {
                    throw ParseError(message: "unknown transport '\(raw)'")
                }
                options.transport = value
            case "--algorithms":
                options.algorithms = try parseAlgorithms(try next(argument))
            case "--codecs":
                options.codecs = try parseCodecs(try next(argument))
            case "--dtype":
                options.dataType = try parseDataType(try next(argument))
            case "--op":
                options.op = try parseOp(try next(argument))
            case "--quick":
                options.minBytes = 1 << 10
                options.maxBytes = 1 << 20
                options.algorithms = [.ring, .tree]
            case "--csv":
                options.csv = true
            case "--codec-bench":
                options.codecBench = true

            // Distributed mode. Any one of these switches it on.
            case "--distributed":
                isDistributed = true
            case "--rank":
                distributed.rank = try integer(argument)
                isDistributed = true
            case "--world-size":
                distributed.worldSize = try integer(argument)
                isDistributed = true
            case "--token":
                distributed.token = try next(argument)
                isDistributed = true
            case "--token-file":
                distributed.tokenFile = try next(argument)
                isDistributed = true
            case "--emit-token":
                distributed.emitToken = true
                isDistributed = true
            case "--bind":
                distributed.bindHosts += try parseHosts(try next(argument))
                isDistributed = true
            case "--rendezvous-port":
                distributed.rendezvousPort = try integer(argument)
                isDistributed = true
            case "--timeout":
                let raw = try next(argument)
                guard let value = Double(raw), value > 0 else {
                    throw ParseError(message: "--timeout: '\(raw)' is not a positive number of seconds")
                }
                distributed.timeout = value
                isDistributed = true
            case "--label":
                distributed.label = try next(argument)
                isDistributed = true

            default:
                throw ParseError(message: "unknown option '\(argument)'")
            }
            index += 1
        }

        if isDistributed {
            guard !options.codecBench else {
                throw ParseError(message: "--codec-bench measures this machine's codec kernels; "
                                 + "there is no world to join")
            }
            guard distributed.worldSize > 1 else {
                throw ParseError(message: "--world-size must be at least 2")
            }
            guard distributed.rank >= 0, distributed.rank < distributed.worldSize else {
                throw ParseError(message: "--rank \(distributed.rank) is outside a world of "
                                 + "\(distributed.worldSize)")
            }
            guard distributed.rank == 0 || distributed.token != nil || distributed.tokenFile != nil else {
                throw ParseError(message: "rank \(distributed.rank) needs --token or --token-file "
                                 + "to find rank 0")
            }
            guard distributed.rank != 0 || distributed.token == nil else {
                throw ParseError(message: "--token names a world someone else is hosting; rank 0 "
                                 + "creates its own (the rendezvous listener has to live in rank "
                                 + "0's process)")
            }
            guard options.transport != .loopback else {
                throw ParseError(message: "--transport loopback is in-process only; a distributed "
                                 + "run uses tcp or rdma")
            }
            if options.transport == .rdma, let reason = RDMATransport.unusableReason() {
                throw ParseError(message: "--transport rdma is not usable here: \(reason)")
            }
            // The distributed world size is the one that counts; `--ranks` is
            // the in-process knob and would otherwise silently disagree.
            options.worldSize = distributed.worldSize
            options.distributed = distributed
        } else {
            guard options.worldSize > 1 else {
                throw ParseError(message: "--ranks must be at least 2")
            }
        }
        guard options.minBytes > 0, options.maxBytes >= options.minBytes else {
            throw ParseError(message: "--min-bytes/--max-bytes must be positive and ordered")
        }
        guard !options.algorithms.isEmpty, !options.codecs.isEmpty else {
            throw ParseError(message: "--algorithms and --codecs must name at least one entry")
        }
        return options
    }

    /// `64K,1M,64M` — or plain byte counts. K/M/G are 1024-based, and the
    /// `KB`/`MB`/`KiB`/`MiB` spellings mean the same thing, because a benchmark
    /// arguing about SI prefixes helps nobody.
    static func parseSizes(_ raw: String) throws -> [Int] {
        let sizes = try raw.split(separator: ",").map { piece -> Int in
            let text = piece.trimmingCharacters(in: .whitespaces).lowercased()
            guard !text.isEmpty else { throw ParseError(message: "--sizes: empty entry") }
            var multiplier = 1
            var digits = Substring(text)
            for (suffix, scale) in [("kib", 1 << 10), ("mib", 1 << 20), ("gib", 1 << 30),
                                    ("kb", 1 << 10), ("mb", 1 << 20), ("gb", 1 << 30),
                                    ("k", 1 << 10), ("m", 1 << 20), ("g", 1 << 30), ("b", 1)]
            where text.hasSuffix(suffix) {
                multiplier = scale
                digits = digits.dropLast(suffix.count)
                break
            }
            guard let value = Int(digits), value > 0 else {
                throw ParseError(message: "--sizes: '\(piece)' is not a positive size")
            }
            return value * multiplier
        }
        guard !sizes.isEmpty else { throw ParseError(message: "--sizes needs at least one size") }
        return sizes
    }

    /// `--bind 169.254.152.222,192.168.1.250` — or one host, or the flag
    /// repeated. Every entry is an address this rank advertises; the dialer on
    /// the far side picks whichever of them it can reach best.
    static func parseHosts(_ raw: String) throws -> [String] {
        let hosts = raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !hosts.isEmpty else { throw ParseError(message: "--bind needs at least one address") }
        return hosts
    }

    static func parseAlgorithms(_ raw: String) throws -> [BenchAlgorithm] {
        try raw.split(separator: ",").map { piece in
            guard let value = BenchAlgorithm(rawValue: piece.trimmingCharacters(in: .whitespaces).lowercased())
            else { throw ParseError(message: "unknown algorithm '\(piece)'") }
            return value
        }
    }

    static func parseCodecs(_ raw: String) throws -> [BenchCodec] {
        try raw.split(separator: ",").map { piece in
            let parts = piece.trimmingCharacters(in: .whitespaces).lowercased()
                .split(separator: ":", maxSplits: 1)
            let name = String(parts[0])
            let parameter = parts.count > 1 ? String(parts[1]) : nil
            switch name {
            case "none", "raw":
                return .none
            case "downcast", "fp16":
                return .downcast
            case "int8", "int8blockwise":
                let block = parameter.flatMap(Int.init) ?? 256
                guard block > 0 else { throw ParseError(message: "int8 block size must be > 0") }
                return .int8Blockwise(blockSize: block)
            case "topk":
                let fraction = parameter.flatMap(Double.init) ?? 0.01
                guard fraction > 0, fraction <= 1 else {
                    throw ParseError(message: "topk fraction must be in (0, 1], got \(fraction)")
                }
                return .topK(fraction: fraction)
            default:
                throw ParseError(message: "unknown codec '\(name)'")
            }
        }
    }

    static func parseDataType(_ raw: String) throws -> DataType {
        switch raw.lowercased() {
        case "fp32", "float32", "f32": return .float32
        case "fp16", "float16", "f16", "half": return .float16
        case "bf16", "bfloat16": return .bfloat16
        case "i32", "int32": return .int32
        case "i8", "int8": return .int8
        default: throw ParseError(message: "unknown dtype '\(raw)'")
        }
    }

    static func parseOp(_ raw: String) throws -> ReduceOp {
        switch raw.lowercased() {
        case "sum": return .sum
        case "prod", "product": return .prod
        case "min": return .min
        case "max": return .max
        case "avg", "mean": return .avg
        default: throw ParseError(message: "unknown op '\(raw)'")
        }
    }
}
