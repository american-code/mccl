import Foundation
import MCCL

// mcclprobe — measure real link bandwidth/latency between cluster nodes and
// print the topology plan MCCL would choose.
//
//   mcclprobe serve [--host H] [--port P]     # run on each peer node
//   mcclprobe measure host:port…              # drive the measurement, print the map
//   mcclprobe plan <topology.json> [--bytes N]
//   mcclprobe info

let usage = """
mcclprobe — mccl topology probe

USAGE
  mcclprobe info
      Print this machine's identity.

  mcclprobe serve [--host <addr>] [--port <n>]
      Answer probes from other nodes. Default 0.0.0.0:7777.

  mcclprobe measure <host:port> [<host:port>…] [options]
      Measure bandwidth/RTT to each peer (and, unless --no-pairwise, between
      peers), then print the resulting topology and the plans it implies.

      --bytes <n>      total bytes per bandwidth measurement (default 67108864)
      --chunk <n>      bytes per streamed frame (default 8388608)
      --pings <n>      ping-pong iterations for RTT (default 100)
      --no-pairwise    only measure links from this node
      --out <path>     write the topology JSON here
      --json           print the topology JSON to stdout

  mcclprobe plan <topology.json> [--bytes <n>]
      Replay the planner over a saved topology.
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("mcclprobe: \(message)\n".utf8))
    exit(1)
}

func requireInt(_ value: String, _ name: String) -> Int {
    guard let parsed = Int(value) else { fail("\(name) must be a number") }
    return parsed
}

func requireAddress(_ value: String) -> PeerAddress {
    guard let parsed = PeerAddress(value) else { fail("cannot parse peer address '\(value)'") }
    return parsed
}

func formatBytesPerSecond(_ value: Double) -> String {
    guard value > 0 else { return "—" }
    let units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
    var v = value
    var unit = 0
    while v >= 1000, unit < units.count - 1 { v /= 1000; unit += 1 }
    return String(format: "%.2f %@", v, units[unit])
}

func formatBytes(_ value: Int) -> String {
    guard value > 0 else { return "0 B" }
    let units = ["B", "KiB", "MiB", "GiB"]
    var v = Double(value)
    var unit = 0
    while v >= 1024, unit < units.count - 1 { v /= 1024; unit += 1 }
    return unit == 0 ? "\(value) B" : String(format: "%.1f %@", v, units[unit])
}

func formatSeconds(_ value: Double) -> String {
    guard value > 0 else { return "—" }
    if value < 1e-3 { return String(format: "%.1f µs", value * 1e6) }
    if value < 1 { return String(format: "%.2f ms", value * 1e3) }
    return String(format: "%.2f s", value)
}

func printIdentity() {
    let identity = NodeIdentity.local()
    print("host:   \(identity.hostname)")
    print("chip:   \(identity.chip)")
    print("memory: \(identity.unifiedMemoryBytes / (1 << 30)) GB unified")

    let interfaces = NetworkInterfaces.local()
    guard !interfaces.isEmpty else { return }
    print("")
    print("interfaces")
    for interface in interfaces {
        let speed = interface.linkSpeedBitsPerSecond > 0
            ? formatBytesPerSecond(Double(interface.linkSpeedBitsPerSecond) / 8).replacingOccurrences(of: "B/s", with: "b/s")
            : "—"
        print("  \(pad(interface.name, 10))\(pad(interface.media.rawValue, 13))"
              + "\(pad(interface.isUp ? "up" : "down", 6))\(pad(speed, 12))"
              + "\(pad(interface.displayName ?? "—", 26))"
              + interface.addresses.map(\.text).joined(separator: " "))
    }
    if let preferred = NetworkInterfaces.preferredLocalAddress(among: interfaces) {
        print("")
        print("peers should dial this node at \(preferred)")
    }
}

/// `%@` ignores width flags on Darwin, so pad in Swift.
func pad(_ value: String, _ width: Int, left: Bool = true) -> String {
    let padding = String(repeating: " ", count: max(0, width - value.count))
    return left ? value + padding : padding + value
}

func printTopology(_ topology: Topology) {
    print("")
    print("nodes")
    for node in topology.nodes.sorted(by: { $0.id < $1.id }) {
        print("  \(pad(String(node.id), 3, left: false))  \(pad(node.hostname, 24))"
              + "\(pad(node.chip, 18))\(node.unifiedMemoryBytes / (1 << 30)) GB")
    }
    print("")
    print("links")
    for link in topology.links {
        print("  \(pad(String(link.from), 3, left: false)) <-> \(pad(String(link.to), 3))"
              + " \(pad(link.kind.rawValue, 15))"
              + "\(pad(formatBytesPerSecond(link.measuredBandwidth ?? 0), 12, left: false))"
              + "   rtt \(formatSeconds(link.measuredLatency ?? 0))")
    }
}

func printPlans(_ topology: Topology) {
    let analysis = TopologyPlanner.analyze(topology)
    print("")
    print("analysis")
    print("  ranks:            \(analysis.rankCount)")
    print("  bottleneck:       \(formatBytesPerSecond(analysis.bottleneckBandwidth ?? 0))")
    print("  worst rtt:        \(formatSeconds(analysis.worstLatency ?? 0))")
    if let crossover = analysis.treeRingCrossoverBytes {
        print("  tree/ring switch: \(formatBytes(crossover)) (derived from the measurements above)")
    }
    if let ratio = analysis.heterogeneityRatio {
        print(String(format: "  fast/slow ratio:  %.2fx", ratio))
    }
    if let islands = analysis.islands {
        print("  islands:          " + islands.map { "[\($0.map(String.init).joined(separator: ","))]" }
            .joined(separator: " "))
    } else {
        print("  islands:          uniform fabric")
    }

    print("")
    print("plans by message size")
    for bytes in [1 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20] {
        print("  \(formatBytes(bytes).padding(toLength: 10, withPad: " ", startingAt: 0))"
              + TopologyPlanner.plan(for: topology, messageBytes: bytes).description)
    }
}

// MARK: - Argument parsing

var arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else {
    print(usage)
    print("")
    printIdentity()
    exit(0)
}
arguments.removeFirst()

func takeOption(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name) else { return nil }
    guard index + 1 < arguments.count else { fail("\(name) needs a value") }
    let value = arguments[index + 1]
    arguments.removeSubrange(index...(index + 1))
    return value
}

func takeFlag(_ name: String) -> Bool {
    guard let index = arguments.firstIndex(of: name) else { return false }
    arguments.remove(at: index)
    return true
}

switch command {
case "info":
    printIdentity()

case "serve":
    let host = takeOption("--host") ?? "0.0.0.0"
    let port = requireInt(takeOption("--port") ?? "7777", "--port")
    do {
        let server = try ProbeServer(host: host, port: port)
        server.start()
        print("mcclprobe serve listening on \(server.address)")
        printIdentity()
        print("(ctrl-c to stop)")
        // Park forever; the accept loop runs on its own thread.
        dispatchMain()
    } catch {
        fail("\(error)")
    }

case "measure":
    let outPath = takeOption("--out")
    let emitJSON = takeFlag("--json")
    var options = TopologyProbe.Options()
    if let bytes = takeOption("--bytes") { options.streamBytes = requireInt(bytes, "--bytes") }
    if let chunk = takeOption("--chunk") { options.chunkBytes = requireInt(chunk, "--chunk") }
    if let pings = takeOption("--pings") { options.pingIterations = requireInt(pings, "--pings") }
    if takeFlag("--no-pairwise") { options.pairwise = false }

    let peers = arguments.map { requireAddress($0) }
    guard !peers.isEmpty else { fail("measure needs at least one <host:port>") }

    do {
        let topology = try TopologyProbe.buildTopology(peers: peers, options: options) { message in
            FileHandle.standardError.write(Data("  \(message)\n".utf8))
        }
        if emitJSON {
            print(String(decoding: try topology.jsonData(), as: UTF8.self))
        } else {
            printTopology(topology)
            printPlans(topology)
        }
        if let outPath {
            try topology.write(to: URL(fileURLWithPath: outPath))
            print("")
            print("wrote \(outPath)")
        }
    } catch {
        fail("\(error)")
    }

case "plan":
    guard let path = arguments.first else { fail("plan needs a topology JSON path") }
    arguments.removeFirst()
    let messageBytes = Int(takeOption("--bytes") ?? "") ?? 0
    do {
        let topology = try Topology.read(from: URL(fileURLWithPath: path))
        printTopology(topology)
        printPlans(topology)
        if messageBytes > 0 {
            print("")
            print("plan for \(formatBytes(messageBytes)): "
                  + TopologyPlanner.plan(for: topology, messageBytes: messageBytes).description)
        }
    } catch {
        fail("\(error)")
    }

case "-h", "--help", "help":
    print(usage)

default:
    fail("unknown command '\(command)'\n\n\(usage)")
}
