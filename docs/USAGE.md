# Using mccl

A practical guide: probe a cluster, bring a communicator up, run collectives,
compress the wire, link from C, benchmark, and — at the end — pick up one of the
pieces that is not built yet.

Every code block below was compiled and run before it was written down. The
transcripts are real output from an Apple M1 Pro (16 GB, macOS 26.5.1, Swift
6.2.3), which means every number in them is a *loopback* number: one machine
pretending to be several. That is the honest limit of what a single Mac can
demonstrate, and it is called out again wherever it matters.

- [Build](#build)
- [Quickstart](#quickstart)
  - [1. Probe the cluster](#1-probe-the-cluster)
  - [2. Bring up a communicator](#2-bring-up-a-communicator)
  - [3. Run the collectives](#3-run-the-collectives)
  - [4. Compress the wire](#4-compress-the-wire)
- [Using the C API](#using-the-c-api)
- [Benchmarking with mcclbench](#benchmarking-with-mcclbench)
- [Implementing what is missing](#implementing-what-is-missing)

## Build

```
swift build            # debug:   .build/debug/{mcclprobe,mcclbench,libmccl.dylib}
swift build -c release # release: .build/release/…
swift test
```

There are no external package dependencies and nothing to install. `xcodebuild`
is not used or supported here.

To depend on mccl from another SwiftPM package:

```swift
// Package.swift
dependencies: [.package(path: "/path/to/mccl")],
targets: [
    .executableTarget(name: "Trainer",
                      dependencies: [.product(name: "MCCL", package: "mccl")])
]
```

---

## Quickstart

### 1. Probe the cluster

mccl plans over measured link bandwidth and latency, not over assumptions. Run
`mcclprobe serve` on every node, then drive the measurement from one of them.

```
# on each peer node
mcclprobe serve --host 0.0.0.0 --port 7777

# from the driving node — it becomes rank 0
mcclprobe measure node1:7777 node2:7777 node3:7777 --out cluster.json
```

`measure` measures its own links directly and asks each peer to measure every
peer after it, so the map is a real pairwise mesh rather than a star. Here it is
run against three `serve` processes on one machine, which is why every link is
`loopback`:

```
$ mcclprobe measure 127.0.0.1:7811 127.0.0.1:7812 127.0.0.1:7813 \
      --bytes 33554432 --out cluster.json
  measuring 0 <-> 1 (127.0.0.1:7811)
  measuring 0 <-> 2 (127.0.0.1:7812)
  measuring 0 <-> 3 (127.0.0.1:7813)
  measuring 1 <-> 2 (127.0.0.1:7811 -> 127.0.0.1:7812)
  measuring 1 <-> 3 (127.0.0.1:7811 -> 127.0.0.1:7813)
  measuring 2 <-> 3 (127.0.0.1:7812 -> 127.0.0.1:7813)

nodes
    0  mbp.local               Apple M1 Pro      16 GB
    1  mbp.local               Apple M1 Pro      16 GB
    2  mbp.local               Apple M1 Pro      16 GB
    3  mbp.local               Apple M1 Pro      16 GB

links
    0 <-> 1   loopback          2.35 GB/s   rtt 68.0 µs
    0 <-> 2   loopback          1.32 GB/s   rtt 69.4 µs
    0 <-> 3   loopback          1.87 GB/s   rtt 60.0 µs
    1 <-> 2   loopback          1.83 GB/s   rtt 16.0 µs
    1 <-> 3   loopback          4.93 GB/s   rtt 17.3 µs
    2 <-> 3   loopback          2.68 GB/s   rtt 12.8 µs

analysis
  ranks:            4
  bottleneck:       1.32 GB/s
  worst rtt:        69.4 µs
  tree/ring switch: 71.5 KiB (derived from the measurements above)
  fast/slow ratio:  3.74x
  islands:          uniform fabric

plans by message size
  1.0 KiB   tree(root: 3, 0:1 3:2,0)
  64.0 KiB  tree(root: 3, 0:1 3:2,0)
  1.0 MiB   ring(0->1->3->2)
  16.0 MiB  ring(0->1->3->2)
  256.0 MiB ring(0->1->3->2)

wrote cluster.json
```

Three things to read out of that:

- **The 71.5 KiB switch point is derived, not configured.** It is
  `S* = α·B·(log₂n − (n−1)) / ((n−1)/n − log₂n)` evaluated at the measured
  α = 69.4 µs and B = 1.32 GB/s. Change the fabric and it moves; there is no
  byte constant in the library to tune.
- **The ring order is not `0→1→2→3`.** It is a greedy maximum-bandwidth walk
  over the measured links, which on this map is `0→1→3→2`.
- **A 3.74x spread is not automatically two fabrics.** Islands are found at the
  largest *ratio gap* between adjacent measured bandwidths, and the largest gap
  here is only 1.84x — so mccl correctly reports one uniform, noisy fabric
  rather than inventing a hierarchy out of jitter.

`mcclprobe info` prints the local identity and interfaces (which is how link
kinds get labelled), and `mcclprobe plan cluster.json --bytes 4194304` replays
the planner over a saved map without re-measuring anything.

Useful `measure` flags: `--bytes` (total per bandwidth measurement, default
64 MiB), `--chunk` (bytes per streamed frame, default 8 MiB), `--pings` (RTT
iterations, default 100), `--no-pairwise`, `--out <path>`, `--json`.

#### Driving the probe from Swift

```swift
import Foundation
import MCCL

// The probe, driven from Swift instead of from `mcclprobe`. Useful when a
// launcher wants to re-measure the fabric itself on membership change.

// What `mcclprobe serve` runs on each peer node.
let server = try ProbeServer(host: "127.0.0.1", port: 0)
server.start()
defer { server.stop() }
print("serving on \(server.address)")

// What `mcclprobe measure` runs on the driving node. Shorter than the defaults
// (64 MiB / 100 pings) so this demo finishes quickly; use the defaults for a
// number you intend to plan against.
var options = TopologyProbe.Options()
options.streamBytes = 8 << 20
options.chunkBytes = 2 << 20
options.pingIterations = 50

let one = try TopologyProbe.measure(peer: server.address, options: options)
print(String(format: "%@ is %@ (%@), %.2f GB/s, rtt %.1f µs",
             one.address.description, one.identity.hostname, one.identity.chip,
             one.bandwidthBytesPerSecond / 1e9, one.roundTripSeconds * 1e6))

// buildTopology makes this node rank 0 and each peer rank 1…k, and — unless
// you turn off `pairwise` — asks each peer to measure every peer after it, so
// the map is a real pairwise mesh rather than a star centred here.
let second = try ProbeServer(host: "127.0.0.1", port: 0)
second.start()
defer { second.stop() }

let topology = try TopologyProbe.buildTopology(
    peers: [server.address, second.address], options: options) { progress in
        print("  \(progress)")
    }
print("map: \(topology.nodes.count) nodes, \(topology.links.count) links")
for link in topology.links {
    print(String(format: "  %d <-> %d  %@  %.2f GB/s  rtt %.1f µs",
                 link.from, link.to, link.kind.rawValue,
                 (link.measuredBandwidth ?? 0) / 1e9, (link.measuredLatency ?? 0) * 1e6))
}

// Persist it; `Topology.read` takes it back, and `mcclprobe plan` replays it.
let out = URL(fileURLWithPath: "/tmp/mccl-probe-demo.json")
try topology.write(to: out)
let reloaded = try Topology.read(from: out)
precondition(reloaded.links.count == topology.links.count)
print("wrote \(out.path)")
print("plan at 4 MiB: \(TopologyPlanner.plan(for: topology, messageBytes: 4 << 20))")
```

```
serving on 127.0.0.1:59414
127.0.0.1:59414 is mbp.local (Apple M1 Pro), 3.47 GB/s, rtt 61.8 µs
  measuring 0 <-> 1 (127.0.0.1:59414)
  measuring 0 <-> 2 (127.0.0.1:59416)
  measuring 1 <-> 2 (127.0.0.1:59414 -> 127.0.0.1:59416)
map: 3 nodes, 3 links
  0 <-> 1  loopback  3.28 GB/s  rtt 50.0 µs
  0 <-> 2  loopback  3.95 GB/s  rtt 44.5 µs
  1 <-> 2  loopback  3.51 GB/s  rtt 51.7 µs
wrote /tmp/mccl-probe-demo.json
plan at 4 MiB: ring(0->2->1)
```

#### Reading a map back and asking the planner why

```swift
import Foundation
import MCCL

// Replay the planner over a saved map, the way `mcclprobe plan` does — and see
// why it chose what it chose.

let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/loopback4.json"
let topology = try Topology.read(from: URL(fileURLWithPath: path))

let analysis = TopologyPlanner.analyze(topology)
print("ranks:      \(analysis.rankCount)")
print("bottleneck: \(analysis.bottleneckBandwidth.map { $0 / 1e9 } ?? 0) GB/s")
print("worst rtt:  \((analysis.worstLatency ?? 0) * 1e6) µs")
print("crossover:  \(analysis.treeRingCrossoverBytes ?? 0) bytes")
print("het ratio:  \(analysis.heterogeneityRatio ?? 0)")
print("islands:    \(analysis.islands.map(String.init(describing:)) ?? "uniform fabric")")

// The crossover is a closed form in the measurements, not a tuned constant:
//   S* = α·B·(log2 n − (n−1)) / ((n−1)/n − log2 n)
let derived = TopologyPlanner.crossoverBytes(
    rankCount: analysis.rankCount,
    latency: analysis.worstLatency ?? 0,
    bandwidth: analysis.bottleneckBandwidth ?? 0)
precondition(derived == analysis.treeRingCrossoverBytes)

for bytes in [1 << 10, 64 << 10, 1 << 20, 64 << 20] {
    print("\(bytes) B -> \(TopologyPlanner.plan(for: topology, messageBytes: bytes))")
}

// A two-rank tree *is* a ring, so the crossover is 0 there whatever was measured.
print("2-rank crossover: \(TopologyPlanner.crossoverBytes(rankCount: 2, latency: 5e-5, bandwidth: 3e9)) bytes")
// An unprobed map has no crossover to derive, so the planner falls back to ring.
print("unprobed plan: \(TopologyPlanner.plan(for: Topology.uniform(nodeCount: 8), messageBytes: 1 << 10))")

// A map is JSON: build one by hand for a fabric you have not probed yet.
let synthetic = Topology(
    nodes: (0..<4).map { Topology.Node(id: $0, hostname: "node\($0)", chip: "M4 Max",
                                       unifiedMemoryBytes: 128 << 30) },
    links: [
        .init(from: 0, to: 1, kind: .thunderbolt5, measuredBandwidth: 5.5e9, measuredLatency: 20e-6),
        .init(from: 2, to: 3, kind: .thunderbolt5, measuredBandwidth: 5.4e9, measuredLatency: 21e-6),
        .init(from: 1, to: 2, kind: .ethernet, measuredBandwidth: 1.15e9, measuredLatency: 210e-6),
        .init(from: 0, to: 2, kind: .ethernet, measuredBandwidth: 1.10e9, measuredLatency: 230e-6),
        .init(from: 1, to: 3, kind: .ethernet, measuredBandwidth: 1.12e9, measuredLatency: 225e-6),
        .init(from: 0, to: 3, kind: .ethernet, measuredBandwidth: 1.08e9, measuredLatency: 235e-6),
    ])
let mixed = TopologyPlanner.analyze(synthetic)
print("two TB5 pairs on a 10GbE bridge -> islands \(mixed.islands!), "
      + "ratio \(String(format: "%.2f", mixed.heterogeneityRatio!))x")
print("  at 64 MiB: \(TopologyPlanner.plan(for: synthetic, messageBytes: 64 << 20))")
print("  at 1 KiB:  \(TopologyPlanner.plan(for: synthetic, messageBytes: 1 << 10))")
try synthetic.write(to: URL(fileURLWithPath: "/tmp/mccl-synthetic.json"))
```

```
ranks:      4
bottleneck: 1.3195686355228777 GB/s
worst rtt:  69.375 µs
crossover:  73236 bytes
het ratio:  3.736347625640165
islands:    uniform fabric
1024 B -> tree(root: 3, 0:1 3:2,0)
65536 B -> tree(root: 3, 0:1 3:2,0)
1048576 B -> ring(0->1->3->2)
67108864 B -> ring(0->1->3->2)
2-rank crossover: 0 bytes
unprobed plan: ring(0->1->2->3->4->5->6->7)
two TB5 pairs on a 10GbE bridge -> islands [[0, 1], [2, 3]], ratio 5.09x
  at 64 MiB: hierarchical(islands: [0,1] [2,3], interIslandRoot: 0)
  at 1 KiB:  tree(root: 1, 1:2,3 3:0)
```

The synthetic map at the bottom is the mixed-fabric case a single machine cannot
produce: two Thunderbolt 5 pairs bridged by 10GbE. Its numbers are *chosen*, not
measured, and are there to show what the planner does with such a fabric — a
hierarchical plan once the message is bandwidth-bound, and a plain tree while it
is still latency-bound.

### 2. Bring up a communicator

There are three ways in, and which one you want depends on what your launcher
already knows.

#### Everything in one process (tests, benchmarks, single-machine work)

`Communicator.tcpGroup(worldSize:)` binds one ephemeral port per rank and
bootstraps the whole mesh over real loopback sockets, so it exercises the same
code path a cluster would. `Communicator.loopbackGroup(worldSize:)` does the
same over in-process byte queues, with no sockets at all.

```swift
import Foundation
import MCCL

// Four ranks in one process, over real loopback TCP sockets — one ephemeral
// port and one socket pair per rank, so this exercises exactly the code path a
// four-machine cluster would.
let worldSize = 4
let count = 1024

let world = try Communicator.tcpGroup(worldSize: worldSize)
defer { world.forEach { $0.shutdown() } }

try await withThrowingTaskGroup(of: Void.self) { group in
    for comm in world {
        group.addTask {
            // The collectives take raw memory plus a dtype. Note the explicit
            // allocation: `withUnsafeMutableBytes` cannot wrap an `await`, so a
            // buffer that outlives the call is the pattern to use.
            let bytes = count * MemoryLayout<Float>.size
            let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: bytes, alignment: 16)
            defer { buffer.deallocate() }
            let floats = buffer.bindMemory(to: Float.self)
            for i in 0..<count { floats[i] = Float(comm.rank + 1) * Float(i % 8 + 1) }

            try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)

            // Ranks contribute 1..4, so every element is 10 * (i % 8 + 1).
            precondition(floats[3] == 10 * 4)
            if comm.rank == 0 {
                print("all-reduce over \(comm.worldSize) ranks: element 3 = \(floats[3])")
                print("plan at \(bytes) bytes: \(comm.plan(messageBytes: bytes))")
            }
        }
    }
    try await group.waitForAll()
}
```

```
all-reduce over 4 ranks: element 3 = 40.0
plan at 4096 bytes: ring(0->1->2->3)
```

> **Buffer lifetime.** `withUnsafeMutableBytes` takes a *synchronous* closure, so
> it cannot wrap an `await`. Allocate a buffer that outlives the collective, or
> hold a long-lived allocation per tensor — which is what a training loop wants
> anyway.

#### One process per machine, addresses known up front

If your scheduler already knows every node's address, `Communicator.bootstrap`
needs no rank-0 rendezvous at all. Every listener must be bound before anyone
dials; `connect` retries to its own deadline, so a staggered launch is fine.

```swift
import Foundation
import MCCL

// One process per rank, addresses known up front — the multi-machine shape.
//
//   Bootstrap --rank 0 --world 3 --peers 10.0.0.1:7000,10.0.0.2:7000,10.0.0.3:7000
//
// `bootstrap` has no opinion about how the job was launched: if your scheduler
// already knows every node's address, this needs no rank-0 rendezvous at all.

var arguments = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
    return arguments[i + 1]
}

let rank = Int(option("--rank") ?? "0")!
let worldSize = Int(option("--world") ?? "1")!
let peers = (option("--peers") ?? "").split(separator: ",").compactMap { PeerAddress(String($0)) }
guard peers.count == worldSize else {
    FileHandle.standardError.write(Data("need \(worldSize) --peers\n".utf8))
    exit(2)
}

// Bind this rank's own listener at the address the others were told to dial.
// Every listener must be bound before anyone dials; `connect` retries to its
// deadline, so a staggered launch is fine.
let listener = try TCPTransport().listen(host: "0.0.0.0", port: peers[rank].port)

// A measured map is optional. Without one the planner sees an unprobed
// topology and picks the ring, which is the safe default.
let topology = (option("--topology").map { URL(fileURLWithPath: $0) }).flatMap { try? Topology.read(from: $0) }

let comm = try Communicator.bootstrap(
    rank: rank, worldSize: worldSize, addresses: peers,
    listener: listener, topology: topology, timeout: 30)
defer { comm.shutdown() }

let count = 1024
let bytes = count * MemoryLayout<Float>.size
let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: bytes, alignment: 16)
defer { buffer.deallocate() }
let floats = buffer.bindMemory(to: Float.self)
for i in 0..<count { floats[i] = Float(rank + 1) }

try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)

let expected = Float(worldSize * (worldSize + 1) / 2)
precondition(floats[0] == expected, "rank \(rank) got \(floats[0]), expected \(expected)")
print("rank \(rank)/\(worldSize): all-reduce = \(floats[0]), plan = \(comm.plan(messageBytes: bytes))")
```

Run as three processes (here on one host, but the addresses are the only thing
that would change on three machines):

```
$ for r in 0 1 2; do
    ./Bootstrap --rank $r --world 3 --peers 127.0.0.1:7301,127.0.0.1:7302,127.0.0.1:7303 &
  done; wait
rank 1/3: all-reduce = 6.0, plan = ring(0->1->2)
rank 0/3: all-reduce = 6.0, plan = ring(0->1->2)
rank 2/3: all-reduce = 6.0, plan = ring(0->1->2)
```

Pass `--topology cluster.json` to hand each rank the map from step 1; without it
the planner sees an unprobed topology and picks the ring, which is the safe
default.

#### One process per machine, one token

If your launcher can move one string but not a full address table, use the
rendezvous. `Rendezvous.createUniqueID()` runs *in the process that will host
rank 0* and binds its listener; every rank then calls
`Communicator.join(uniqueID:rank:worldSize:)`, which binds its own data listener,
exchanges addresses through rank 0, and blocks until everyone has arrived. This
is exactly what the C shim's `mcclCommInitRank` does.

```swift
import Foundation
import MCCL

// One process per rank, discovering each other through a single token — the
// `mcclGetUniqueId` / `mcclCommInitRank` shape, for launchers that can move one
// string but not a full address table.
//
//   Token --rank 0 --world 3 --token-file /tmp/mccl.id   # creates the token
//   Token --rank 1 --world 3 --token-file /tmp/mccl.id   # reads it
//   Token --rank 2 --world 3 --token-file /tmp/mccl.id

var arguments = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
    return arguments[i + 1]
}

let rank = Int(option("--rank") ?? "0")!
let worldSize = Int(option("--world") ?? "1")!
let tokenPath = option("--token-file") ?? "/tmp/mccl.id"

let id: UniqueID
if rank == 0 {
    // Must run in the process that will host rank 0: `createUniqueID` binds the
    // rendezvous listener and the token describes it.
    id = try Rendezvous.createUniqueID(host: "127.0.0.1")
    try Data(id.text.utf8).write(to: URL(fileURLWithPath: tokenPath), options: .atomic)
    print("rank 0 published \(id.text)")
} else {
    // Any transport that moves one string will do: a file, an env var, MPI.
    var text: String?
    let deadline = Date().addingTimeInterval(30)
    while text == nil, Date() < deadline {
        text = try? String(contentsOf: URL(fileURLWithPath: tokenPath), encoding: .utf8)
        if text == nil { usleep(50_000) }
    }
    guard let text, let parsed = UniqueID(text: text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
        FileHandle.standardError.write(Data("no usable token at \(tokenPath)\n".utf8))
        exit(2)
    }
    id = parsed
}

// join binds this rank's data listener, exchanges addresses through rank 0, and
// brings the mesh up. It blocks until every rank has done the same.
let comm = try Communicator.join(uniqueID: id, rank: rank, worldSize: worldSize, timeout: 60)
defer { comm.shutdown() }

let count = 512
let bytes = count * MemoryLayout<Float>.size
let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: bytes, alignment: 16)
defer { buffer.deallocate() }
let floats = buffer.bindMemory(to: Float.self)
for i in 0..<count { floats[i] = Float(rank + 1) }

try await comm.allReduce(buffer, count: count, dataType: .float32, op: .sum)

let expected = Float(worldSize * (worldSize + 1) / 2)
precondition(floats[0] == expected, "rank \(rank) got \(floats[0]), expected \(expected)")
print("rank \(rank)/\(worldSize): all-reduce = \(floats[0])")

if rank == 0 { try? FileManager.default.removeItem(atPath: tokenPath) }
```

```
$ ./Token --rank 0 --world 3 --token-file /tmp/mccl-demo.id &
$ sleep 0.3
$ ./Token --rank 1 --world 3 --token-file /tmp/mccl-demo.id &
$ ./Token --rank 2 --world 3 --token-file /tmp/mccl-demo.id &
$ wait
rank 2/3: all-reduce = 6.0
rank 1/3: all-reduce = 6.0
rank 0 published mccl1:45eb4bcb4375ae8f:127.0.0.1:59339
rank 0/3: all-reduce = 6.0
```

The token is `mccl1:<nonce hex>:<host>:<port>` — printable on purpose, so it can
travel through a shell, an env var or a job script. The nonce keeps two
concurrent jobs on one host apart: a rank announcing the wrong nonce is rejected
rather than silently joining the other job's world.

Binding with a wildcard host advertises a routable address instead of `0.0.0.0`;
set `MCCL_HOST` to override what a rank advertises when the machine is behind
NAT or has several candidate interfaces.

**What the rendezvous is not:** a service. Rank 0 must be up, `createUniqueID`
must run in rank 0's own process, and there is no discovery, no membership
change, and no recovery from a rank restarting. See
[Implementing what is missing](#implementing-what-is-missing).

### 3. Run the collectives

All five, for every `DataType` (`.float32`, `.float16`, `.bfloat16`, `.int32`,
`.int8`) and every `ReduceOp` (`.sum`, `.prod`, `.min`, `.max`, `.avg`).

```swift
import Foundation
import MCCL

// Every collective mccl implements, over one four-rank world.
let worldSize = 4
let count = 512                      // elements per rank
let segment = count / worldSize      // reduce-scatter output per rank

func makeBuffer(_ elements: Int) -> UnsafeMutableRawBufferPointer {
    UnsafeMutableRawBufferPointer.allocate(
        byteCount: elements * MemoryLayout<Float>.size, alignment: 16)
}

let world = try Communicator.tcpGroup(worldSize: worldSize)
defer { world.forEach { $0.shutdown() } }

try await withThrowingTaskGroup(of: Void.self) { group in
    for comm in world {
        group.addTask {
            let contribution = { (i: Int) in Float(comm.rank + 1) * Float(i % 7 + 1) }
            let sum = { (i: Int) in Float(10 * (i % 7 + 1)) }   // ranks 1..4 sum to 10

            // ---- all-reduce: every rank ends with the reduction of all ranks.
            let a = makeBuffer(count); defer { a.deallocate() }
            let af = a.bindMemory(to: Float.self)
            for i in 0..<count { af[i] = contribution(i) }
            try await comm.allReduce(a, count: count, dataType: .float32, op: .sum)
            precondition(af[5] == sum(5))

            // `.avg` divides by the world size exactly once, at the end.
            for i in 0..<count { af[i] = contribution(i) }
            try await comm.allReduce(a, count: count, dataType: .float32, op: .avg)
            precondition(af[5] == sum(5) / Float(worldSize))

            // ---- all-gather: recv holds worldSize * count elements,
            //      rank r's contribution at slot r.
            let send = makeBuffer(count); defer { send.deallocate() }
            let recv = makeBuffer(count * worldSize); defer { recv.deallocate() }
            let sf = send.bindMemory(to: Float.self)
            for i in 0..<count { sf[i] = Float(comm.rank * 1000 + i % 10) }
            try await comm.allGather(UnsafeRawBufferPointer(send), into: recv,
                                     count: count, dataType: .float32)
            let rf = recv.bindMemory(to: Float.self)
            for r in 0..<worldSize { precondition(rf[r * count + 3] == Float(r * 1000 + 3)) }

            // ---- broadcast: root's buffer wins; everyone else's is overwritten.
            let b = makeBuffer(count); defer { b.deallocate() }
            let bf = b.bindMemory(to: Float.self)
            for i in 0..<count { bf[i] = comm.rank == 2 ? Float(i % 13) : -1 }
            try await comm.broadcast(b, count: count, dataType: .float32, root: 2)
            precondition(bf[4] == 4)

            // ---- reduce-scatter: send worldSize * segment in, keep segment r.
            let rs = makeBuffer(count); defer { rs.deallocate() }
            let out = makeBuffer(segment); defer { out.deallocate() }
            let rsf = rs.bindMemory(to: Float.self)
            for i in 0..<count { rsf[i] = contribution(i) }
            try await comm.reduceScatter(UnsafeRawBufferPointer(rs), into: out,
                                         recvCount: segment, dataType: .float32, op: .sum)
            let of = out.bindMemory(to: Float.self)
            precondition(of[0] == sum(comm.rank * segment))

            // ---- reduce: the result lands on the root only.
            let d = makeBuffer(count); defer { d.deallocate() }
            let df = d.bindMemory(to: Float.self)
            for i in 0..<count { df[i] = contribution(i) }
            try await comm.reduce(d, count: count, dataType: .float32, op: .sum, root: 1)
            if comm.rank == 1 { precondition(df[5] == sum(5)) }

            if comm.rank == 0 { print("all five collectives agreed on \(worldSize) ranks") }
        }
    }
    try await group.waitForAll()
}

// Every dtype and every reduction works the same way.
for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
    print("  \(dataType): \(dataType.byteWidth) bytes/element, "
          + "floating point: \(dataType.isFloatingPoint)")
}
```

```
all five collectives agreed on 4 ranks
  fp32: 4 bytes/element, floating point: true
  fp16: 2 bytes/element, floating point: true
  bf16: 2 bytes/element, floating point: true
  i32: 4 bytes/element, floating point: false
  i8: 1 bytes/element, floating point: false
```

Buffer shapes, in one place:

| call | input | output |
|---|---|---|
| `allReduce` | `count` elements, in place | same buffer, reduced |
| `allGather` | `count` elements | `worldSize * count`, rank `r` at slot `r` |
| `broadcast` | `count` on the root | `count` everywhere |
| `reduceScatter` | `worldSize * recvCount` | `recvCount`, this rank's segment |
| `reduce` | `count`, in place | `count` on the root; scratch elsewhere |

`.avg` on an integer dtype rounds to nearest rather than truncating, so an
averaged all-reduce does not drift downwards over a training run.

`comm.plan(messageBytes:)` reports what the planner would choose;
`comm.planOverride` pins an algorithm, which is what `mcclbench` uses to compare
them. A plan that does not name exactly this world — a stale map, a rank that
left — degrades to the plain ring rather than failing.

### 4. Compress the wire

On a Mac cluster the interconnect is the bottleneck, not the arithmetic. All
three schemes are per-hop and never change the dtype the caller sees.

| scheme | wire size for `n` fp32 elements | error |
|---|---|---|
| `.none` | `4n` | exact |
| `.downcast` | `2n` | relative ≤ 2⁻¹¹ |
| `.int8Blockwise(blockSize: b)` | `4⌈n/b⌉ + n` | absolute ≤ blockAbsmax/254 |
| `.topK(fraction: f)` | `4 + k(4 + 4)` per rank, `k = ⌈fn⌉` | unbiased over repeated calls |

A scheme that cannot help a given dtype degrades to raw rather than erroring:
`.downcast` on an fp16 buffer is already done, and quantising an int32
accumulator would be wrong.

```swift
import Foundation
import MCCL

// Every wire compression scheme, and what each one costs you in accuracy.
let worldSize = 4
let count = 4096

func buffer(_ elements: Int) -> UnsafeMutableRawBufferPointer {
    UnsafeMutableRawBufferPointer.allocate(
        byteCount: elements * MemoryLayout<Float>.size, alignment: 16)
}

let world = try Communicator.tcpGroup(worldSize: worldSize)
defer { world.forEach { $0.shutdown() } }

// A gradient-shaped signal: a few large entries, a long tail of small ones.
func gradient(rank: Int, _ i: Int) -> Float {
    i % 64 == 0 ? Float(rank + 1) * 10 : Float(rank + 1) * 0.001 * Float(i % 7 + 1)
}
func exactSum(_ i: Int) -> Float {
    (0..<worldSize).reduce(Float(0)) { $0 + gradient(rank: $1, i) }
}

try await withThrowingTaskGroup(of: Void.self) { group in
    for comm in world {
        group.addTask {
            let b = buffer(count); defer { b.deallocate() }
            let f = b.bindMemory(to: Float.self)

            // ---- .none: bit-exact, full bandwidth cost.
            for i in 0..<count { f[i] = gradient(rank: comm.rank, i) }
            try await comm.allReduce(b, count: count, dataType: .float32,
                                     op: .sum, compression: .none)
            precondition(f[0] == exactSum(0))

            // ---- .downcast: fp32 -> fp16 on the wire, half the bytes.
            //      Relative error <= 2^-11, so compare with a tolerance.
            for i in 0..<count { f[i] = gradient(rank: comm.rank, i) }
            try await comm.allReduce(b, count: count, dataType: .float32,
                                     op: .sum, compression: .downcast)
            precondition(abs(f[0] - exactSum(0)) <= abs(exactSum(0)) * 0.01)

            // ---- .int8Blockwise: per-block absmax scale + int8 payload.
            //      Error is bounded by blockAbsmax/254, so a block containing
            //      one large entry quantises its small neighbours coarsely.
            for i in 0..<count { f[i] = gradient(rank: comm.rank, i) }
            try await comm.allReduce(b, count: count, dataType: .float32,
                                     op: .sum, compression: .int8Blockwise(blockSize: 256))
            precondition(abs(f[0] - exactSum(0)) <= abs(exactSum(0)) * 0.05)

            // ---- .topK: not a per-hop codec but a different algorithm.
            //      Each call sends only the k largest magnitudes and carries the
            //      rest forward, so repeated calls converge on the exact answer.
            let stream = StreamID(1)
            var accumulated = [Float](repeating: 0, count: count)
            // Mean absolute error of the running average against the exact
            // all-reduce. If error feedback works this shrinks with the number
            // of steps; a merely lossy codec would leave it flat.
            func meanError(after steps: Int) -> Float {
                var total: Float = 0
                for i in 0..<count { total += abs(accumulated[i] / Float(steps) - exactSum(i)) }
                return total / Float(count)
            }
            var errorAfter5: Float = 0
            for step in 1...80 {
                for i in 0..<count { f[i] = gradient(rank: comm.rank, i) }
                try await comm.allReduce(b, count: count, dataType: .float32, op: .sum,
                                         compression: .topK(fraction: 0.02), stream: stream)
                for i in 0..<count { accumulated[i] += f[i] }
                if step == 5 { errorAfter5 = meanError(after: 5) }
            }
            let errorAfter80 = meanError(after: 80)
            precondition(errorAfter80 < errorAfter5,
                         "error feedback should converge, got \(errorAfter5) -> \(errorAfter80)")
            if comm.rank == 0 {
                print(String(format: "top-k mean error: %.6f after 5 steps, %.6f after 80",
                             errorAfter5, errorAfter80))
            }

            if comm.rank == 0 {
                // What top-k is still holding back. A checkpoint saved without
                // this has thrown away part of the gradient.
                let residual = comm.topKResidual(for: stream)!
                let carried = residual.reduce(0) { $0 + abs($1) }
                print("codecs agreed; top-k residual on \(stream): "
                      + "\(residual.count) elements, L1 \(String(format: "%.4f", carried))")
                print("top-k wire bytes at fraction 0.02: "
                      + "\(comm.topKWireBytes(count: count, dataType: .float32, fraction: 0.02))"
                      + " vs dense \(2 * (worldSize - 1) * count * 4 / worldSize)")
            }
            comm.clearTopKResiduals(stream: stream)

            // ---- What top-k refuses, and why.
            let g = buffer(count); defer { g.deallocate() }
            for restriction in ["op", "dtype", "collective"] {
                do {
                    switch restriction {
                    case "op":
                        // Dropped elements are identities of a sum, not of a min.
                        try await comm.allReduce(g, count: count, dataType: .float32,
                                                 op: .min, compression: .topK(fraction: 0.1))
                    case "dtype":
                        // Magnitude sparsification of an int accumulator is meaningless.
                        try await comm.allReduce(g, count: count, dataType: .int32,
                                                 op: .sum, compression: .topK(fraction: 0.1))
                    default:
                        // Only all-reduce can account for the dropped mass.
                        let out = buffer(count * worldSize); defer { out.deallocate() }
                        try await comm.allGather(UnsafeRawBufferPointer(g), into: out,
                                                 count: count, dataType: .float32,
                                                 compression: .topK(fraction: 0.1))
                    }
                    fatalError("\(restriction): expected .unsupportedCompression")
                } catch MCCLError.unsupportedCompression(let why) {
                    if comm.rank == 0 { print("  topK rejected on \(restriction): \(why.prefix(72))…") }
                }
            }
        }
    }
    try await group.waitForAll()
}
```

```
top-k mean error: 0.038452 after 5 steps, 0.028422 after 80
codecs agreed; top-k residual on stream(1): 4096 elements, L1 931.3215
top-k wire bytes at fraction 0.02: 2052 vs dense 24576
  topK rejected on op: topK is defined for .sum and .avg gradient reductions only, not min…
  topK rejected on dtype: topK needs a floating-point dtype; i32 is an accumulator type and magnit…
  topK rejected on collective: topK(fraction: 0.1) is not a per-hop wire codec: it is an all-reduce-onl…
```

#### Top-k: what you have to know before you use it

Top-k is the one stateful thing in mccl, and the state is the whole point. Each
call adds the residual left over from the previous one, sends the `k` largest
magnitudes, and keeps the rest — so nothing is discarded, only delayed, and a
training run converges on the uncompressed answer rather than drifting away from
it.

That has four consequences you cannot ignore:

1. **One stream per tensor.** The residual is keyed by `(StreamID, elementCount)`.
   Two tensors reduced on one communicator with the default stream would share a
   residual and corrupt each other. Give each its own:

   ```swift
   try await comm.allReduce(gradients, count: n, dataType: .float32, op: .sum,
                            compression: .topK(fraction: 0.01), stream: StreamID(layer))
   ```

   A changed element count on the same stream starts a fresh history rather than
   mixing two shapes.

2. **A checkpoint without the residual is incomplete.** `comm.topKResidual(for:)`
   returns what is still being carried, in element order, as `[Float]` whatever
   the caller's dtype (an fp16 residual would flush the small values it exists to
   preserve straight back to zero). Save it beside the model, restore it on
   resume, or accept that resuming discards part of a gradient.
   `comm.clearTopKResiduals(stream:)` drops one stream's history or all of them.

3. **Choose the fraction with headroom.** In the run above, 64 of the 4096
   elements are permanently large, and `fraction: 0.02` gives `k = 82` — only 18
   slots per step rotate through the remaining 4032. The error does fall
   (0.0385 → 0.0284 over 80 steps) but slowly, and the residual's L1 stays high.
   Pick `k` comfortably larger than the count of persistently-dominant entries,
   or the tail will take a very long time to be delivered.

4. **It refuses what it cannot do honestly.** All-reduce only (nothing else can
   account for the dropped mass); `.sum`/`.avg` only (dropped elements are
   identities of a sum, not of a `min`); floating-point dtypes only. Each refusal
   is `MCCLError.unsupportedCompression` with a message saying what to call
   instead.

`comm.topKWireBytes(count:dataType:fraction:)` reports what a call will actually
put on the wire — 2052 bytes per rank above, against 24576 for a dense ring.

---

## Using the C API

`mccl.h` is hand-written and deliberately shaped like NCCL: a runtime that
already speaks `ncclComm_t` should need renaming and nothing else. This is a
complete program — bring-up from a token, an all-reduce, a compressed
all-reduce, teardown.

```c
/* mccl_hello.c — a complete, minimal C client for libmccl.
 *
 *   swift build -c release
 *   cc -std=c11 -Wall mccl_hello.c \
 *      -I Sources/MCCLShim/include \
 *      -L .build/release -lmccl -Wl,-rpath,$PWD/.build/release \
 *      -o mccl_hello
 *   ./mccl_hello
 */

#include <mccl.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RANKS 3
#define COUNT 1024

typedef struct { int rank; mcclUniqueId id; } rank_args;

static int failures = 0;

static void* rank_main(void* raw) {
    rank_args* args = (rank_args*)raw;
    mcclComm_t comm = NULL;

    /* Every rank calls this with the same token; it blocks until all of them
     * have. NCCL's signature exactly, token by value. */
    mcclResult_t r = mcclCommInitRank(&comm, RANKS, args->id, args->rank);
    if (r != mcclSuccess) {
        fprintf(stderr, "rank %d: init failed: %s (%s)\n", args->rank,
                mcclGetErrorString(r), mcclGetLastError(NULL));
        __sync_fetch_and_add(&failures, 1);
        return NULL;
    }

    float* send = malloc(sizeof(float) * COUNT);
    float* recv = malloc(sizeof(float) * COUNT);
    for (int i = 0; i < COUNT; ++i) send[i] = (float)(args->rank + 1);

    /* Blocking: when this returns, the buffers are safe to touch again. */
    r = mcclAllReduce(send, recv, COUNT, mcclFloat32, mcclSum, comm, NULL);
    if (r != mcclSuccess) {
        fprintf(stderr, "rank %d: allreduce: %s (%s)\n", args->rank,
                mcclGetErrorString(r), mcclGetLastError(comm));
        __sync_fetch_and_add(&failures, 1);
    }
    /* 1 + 2 + 3 */
    if (recv[0] != 6.0f) {
        fprintf(stderr, "rank %d: got %f, expected 6\n", args->rank, recv[0]);
        __sync_fetch_and_add(&failures, 1);
    }

    /* mccl-specific: fp16 on the wire, and ask the planner what it chose. */
    r = mcclAllReduceCompressed(send, recv, COUNT, mcclFloat32, mcclSum,
                                mcclCompressDowncast, 0, 0.0, comm, NULL);
    if (r != mcclSuccess) { fprintf(stderr, "compressed: %s\n", mcclGetErrorString(r)); }

    if (args->rank == 0) {
        char plan[128] = {0};
        mcclCommPlanDescription(comm, COUNT * sizeof(float), plan, sizeof(plan));
        int nranks = 0;
        mcclCommCount(comm, &nranks);
        printf("%d ranks, all-reduce = %.1f, plan = %s\n", nranks, recv[0], plan);
    }

    free(send);
    free(recv);
    mcclCommDestroy(comm);
    return NULL;
}

int main(void) {
    int version = 0;
    mcclGetVersion(&version);
    printf("libmccl %d (header says %d)\n", version, MCCL_VERSION_CODE);

    /* Call once, in the process that will host rank 0, before any rank calls
     * mcclCommInitRank. Distribute the token out of band. */
    rank_args args[RANKS];
    mcclResult_t r = mcclGetUniqueId(&args[0].id);
    if (r != mcclSuccess) {
        fprintf(stderr, "mcclGetUniqueId: %s (%s)\n",
                mcclGetErrorString(r), mcclGetLastError(NULL));
        return 1;
    }

    /* The payload is printable text, so it travels through a shell, an env var
     * or a job script — that is how the other machines would receive it. */
    char text[MCCL_UNIQUE_ID_BYTES] = {0};
    mcclUniqueIdToString(&args[0].id, text, sizeof(text));
    printf("token: %s\n", text);

    for (int i = 0; i < RANKS; ++i) {
        args[i].rank = i;
        /* On another machine this would be mcclUniqueIdFromString(text, &args[i].id). */
        memcpy(&args[i].id, &args[0].id, sizeof(mcclUniqueId));
    }

    pthread_t threads[RANKS];
    for (int i = 0; i < RANKS; ++i) pthread_create(&threads[i], NULL, rank_main, &args[i]);
    for (int i = 0; i < RANKS; ++i) pthread_join(threads[i], NULL);

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("ok\n");
    return 0;
}
```

The build command, run from the repository root:

```
$ swift build -c release
$ cc -std=c11 -Wall mccl_hello.c \
     -I Sources/MCCLShim/include \
     -L .build/release -lmccl -Wl,-rpath,$PWD/.build/release \
     -o mccl_hello
$ ./mccl_hello
libmccl 500 (header says 500)
token: mccl1:dd6bd63dcb38875a:169.254.233.203:59475
3 ranks, all-reduce = 6.0, plan = ring(0->1->2)
ok
```

Swap `release` for `debug` in both `-L` and `-rpath` to link the debug build; it
works identically. The `-rpath` is what lets the binary find `libmccl.dylib` at
run time without `DYLD_LIBRARY_PATH`.

### Where mccl differs from NCCL, and why

- **Every call blocks.** NCCL enqueues onto a CUDA stream and returns; mccl v0
  has no device queue to enqueue onto, so a collective runs to completion and
  returns when the buffers are safe to touch again.
- **`mcclStream_t` is not an execution context.** It names an independent
  *sequence* of collectives, which matters only to top-k's per-stream residual.
  Build one with `mcclStreamFromId(7)`; `NULL` is the default stream. Rank `r`
  and rank `s` must use the same id for the same tensor.
- **`mcclCommInitRank` is a `static inline`.** A by-value 128-byte
  `mcclUniqueId` has a different ABI on arm64 and x86_64, so the exported symbol
  (`mcclCommInitRankFromId`) takes the token by pointer and the header restores
  the idiomatic call site. Source-compatible with NCCL either way.
- **One communicator, one thread.** Different communicators are independent.

### Error handling

`mcclResult_t` maps one-to-one onto Swift's `MCCLError`; the values cross the ABI
boundary and are never renumbered. `mcclGetErrorString(code)` names a code and
never returns `NULL`. `mcclGetLastError(comm)` returns the full Swift error text
for the most recent failure on that communicator — offending values included —
and stays valid until the next failing call on it. Pass `NULL` for failures that
happened before a communicator existed (`mcclGetUniqueId`, `mcclCommInitRank`).

mccl-specific extras beyond NCCL's surface: `mcclAllReduceCompressed` (the
codecs), `mcclCommPlanDescription` (ask the planner what it chose),
`mcclUniqueIdToString` / `mcclUniqueIdFromString` / `mcclUniqueIdDiscard`.

`Tests/MCCLTests/CProgram/mccl_smoke.c` is a larger four-rank client exercising
every exported symbol; `CShimTests` compiles it with `cc` against the built dylib
and runs it as part of `swift test`.

---

## Benchmarking with mcclbench

`mcclbench` sweeps every (message size × algorithm × codec) point. One world is
brought up for the whole sweep and the plan is pinned per point, so the table
compares algorithms and not bootstrap costs.

Two modes, one table. By default it brings up `--ranks` ranks inside this
process, which measures the collectives and the codecs but no cable. With
`--rank` / `--world-size` it runs as one rank of a real multi-machine world —
see [Across machines](#across-machines).

```
mcclbench                        # 1 KiB .. 64 MiB, 4 ranks, every algorithm and codec
mcclbench --quick                # 1 KiB .. 1 MiB, ring and tree only
mcclbench --ranks 8 --collective allgather --codecs none,int8:128 --csv
```

### Flags

| flag | default | meaning |
|---|---|---|
| `--ranks <n>` | 4 | ranks in the world; must be ≥ 2 |
| `--min-bytes <n>` | 1024 | smallest message |
| `--max-bytes <n>` | 67108864 | largest message |
| `--step <n>` | 4 | size multiplier between points |
| `--sizes <list>` | — | explicit sizes instead of the sweep, e.g. `64K,1M,64M` |
| `--collective <name>` | `allreduce` | `allreduce`, `allgather`, `broadcast`, `reducescatter` |
| `--algorithms <list>` | all three | `ring`, `tree`, `hierarchical` |
| `--codecs <list>` | all four | `none`, `downcast`, `int8[:block]`, `topk[:fraction]` |
| `--dtype <name>` | `fp32` | `fp32`, `fp16`, `bf16`, `i32`, `i8` |
| `--op <name>` | `sum` | `sum`, `prod`, `min`, `max`, `avg` |
| `--transport <name>` | `tcp` | `tcp` (real loopback sockets) or `loopback` (in-process queues) |
| `--warmup <n>` | 2 | untimed iterations per point |
| `--min-iters <n>` | 3 | floor on timed iterations |
| `--max-iters <n>` | 50 | ceiling on timed iterations |
| `--budget <n>` | 134217728 | bytes to move per point; sets the iteration count |
| `--quick` | — | 1 KiB .. 1 MiB, ring and tree only |
| `--csv` | — | machine-readable output |
| `--help` | — | usage text |

Codecs that cannot apply to a point are skipped rather than reported as
failures: `downcast` needs fp32, `int8` needs a floating-point dtype, and `topk`
is all-reduce-only on a floating-point dtype. `--transport loopback` isolates
codec and kernel cost from the socket.

### Across machines

`--rank` / `--world-size` turn `mcclbench` into one rank of a real world. It
joins through the same rendezvous token as
[the token example above](#one-process-per-machine-one-token), sweeps the same
grid, and prints the same table — over whatever cable the machines share.

| flag | default | meaning |
|---|---|---|
| `--rank <n>` | — | this process's rank; any distributed flag enables the mode |
| `--world-size <n>` | 2 | ranks in the world |
| `--token <text>` | — | join the world named by this token |
| `--token-file <path>` | — | rank 0 writes the token here; other ranks poll for it |
| `--emit-token` | — | rank 0 prints `MCCL_TOKEN=<text>` on stdout, first line |
| `--bind <host>` | — | local address to bind *and* advertise |
| `--rendezvous-port <n>` | any | fixed port for rank 0's rendezvous listener |
| `--timeout <seconds>` | 120 | bring-up deadline |
| `--label <name>` | — | name for the fabric, echoed into the header |

Rank 0 creates the token and must be started first, because `createUniqueID`
binds the listener the token describes:

```
lab-01$ mcclbench --rank 0 --world-size 2 --bind 169.254.152.222 \
                  --token-file /tmp/tb.id --sizes 64K,1M,64M --algorithms ring
mcclbench: rank 0 token mccl1:f5a44d951e3024b2:169.254.152.222:60278

lab-02$ mcclbench --rank 1 --world-size 2 --bind 169.254.23.203 \
                  --token mccl1:f5a44d951e3024b2:169.254.152.222:60278 \
                  --sizes 64K,1M,64M --algorithms ring
```

Four things are worth knowing before you trust the numbers that come out:

**Pass `--bind` on any machine with more than one interface.** Without it the
listener binds a wildcard and advertises whatever
`NetworkInterfaces.preferredLocalAddress()` prefers, which may not be the cable
you meant to measure. `--bind` pins both. Never use an mDNS `.local` name here —
it resolves over whichever path the resolver likes, and a two-machine setup with
both Wi-Fi and Thunderbolt will silently give you the wrong one. Check the
printed token names the address you expect before starting the other ranks, and
check the interface counters afterwards (`netstat -ib`) if it matters.

**Give every rank the same grid.** The sweep must be identical on every rank or
they walk different points and block on each other. It is fingerprinted and
agreed at bring-up, so a mismatch is an error naming the offending rank rather
than a hang — but it is still your job to launch them the same.

**Only rank 0 prints the table.** The other ranks put a status line on stderr and
leave stdout empty, so a launcher scraping the run gets exactly one table.

**The wall time is the slowest rank's**, agreed with a one-element max all-reduce
after each point, and a barrier separates each point's warm-up from its timed
region. Rank 0's own clock would understate the cost whenever rank 0 is the rank
that waits.

A point that fails aborts the run rather than continuing: the ranks have already
diverged by then, and the next collective would hang instead of reporting
anything.

### Reading the table

```
$ mcclbench --quick
        size  algorithm     codec           iters          wall    GB/s alg    GB/s bus
---------------------------------------------------------------------------------------
      64 KiB  ring          none               50      452.2 µs       0.145       0.217
      64 KiB  tree          none               50      218.8 µs       0.299       0.449
     256 KiB  ring          none               50      560.1 µs       0.468       0.702
     256 KiB  tree          none               50      471.1 µs       0.556       0.835
       1 MiB  ring          none               50      1.167 ms       0.898       1.348
       1 MiB  ring          int8/256           50      3.280 ms       0.320       0.480
       1 MiB  tree          none               50      1.524 ms       0.688       1.032
       1 MiB  tree          topk/0.01          50      1.312 ms       0.799       1.199

best point per codec
  none          1.348 GB/s bus      at 1 MiB on ring
  downcast      1.172 GB/s bus      at 1 MiB on ring
  int8/256      0.480 GB/s bus      at 1 MiB on ring
  topk/0.01     1.199 GB/s bus      at 1 MiB on tree
```

*(rows trimmed for width; the full sweep prints every algorithm × codec pair)*

**"GB/s alg"** is payload bytes over wall time — what the caller sees move. It is
the number you compare between two runs of *your own* application.

**"GB/s bus"** scales that by the traffic each link actually carries, which is
the figure comparable to NCCL's `busbw` and to MLX's ring:

| collective | bus factor |
|---|---|
| all-reduce | `2(n-1)/n` |
| all-gather, reduce-scatter | `(n-1)/n` |
| broadcast | `1` |

Bus bandwidth is the one that should approach the link's line rate as the message
grows, and the one that is comparable across different collectives and different
world sizes. Algorithmic bandwidth is not: an all-reduce moves nearly twice its
payload across every link, so its alg number can never reach line rate.

### Reading these numbers honestly

**Over loopback, the table is a codec *cost* measurement, not a codec verdict.**
There is no wire: the "interconnect" is memory bandwidth, roughly an order of
magnitude faster than a 10GbE or TB5 hop. So every codec pays its full encode
cost and buys back nothing, and `int8` duly comes last. What the table gives you
is each codec's CPU price per byte.

That price turned out to decide the question. Measured across two Mac Studios on
a 1.06 GB/s Thunderbolt cable, every codec *loses*: `downcast` halves the bytes
on the wire and is still 25–32% slower than fp32, because the scalar encoder tops
out around 0.50 GB/s of payload — below what the cable delivers uncompressed.
Over Wi-Fi at 0.05 GB/s the same codecs win by 1.9–5.4×. The rule: **a codec pays
only when the fabric's uncompressed all-reduce rate is below that codec's own
encode/decode ceiling.** Full tables in
[ARCHITECTURE.md § Measured: 2-node Thunderbolt](ARCHITECTURE.md#measured-2-node-thunderbolt).

The same caution applies to the algorithm columns: over loopback the tree's
`O(log n)` hop advantage is worth much less than it would be across four
machines, because the "hops" are memcpys.

`--csv` emits one row per point plus a `wire_bytes_per_rank` column, which is
populated for `topk` (where the wire size is not a function of the message size
alone):

```
$ mcclbench --ranks 4 --codecs topk:0.01,none --algorithms ring \
      --min-bytes 1048576 --max-bytes 1048576 --csv
size_bytes,algorithm,codec,iterations,wall_ms,alg_GB_s,bus_GB_s,wire_bytes_per_rank
1048576,ring,topk/0.01,50,1.4247,0.7360,1.1040,63012
1048576,ring,none,50,1.2766,0.8214,1.2320,
```

The harness lives in `Sources/MCCLBenchmarks`, so a test or another tool can
drive `BenchRunner.run(_:)` directly rather than scraping stdout.

---

## Implementing what is missing

Four things are declared, designed for, and not built. Each has a seam already
cut for it; this section is enough to start any of them.

### MLX adapter

**Where it hooks.** MLX ships its own ring all-reduce over
`mlx.core.distributed`. The adapter is a shim that lets MLX hand mccl an array
and get a reduced one back, so `mcclbench` can be pointed at the same workload
MLX's ring runs and produce head-to-head numbers.

MLX arrays are unified-memory buffers, which is the whole reason this is a shim
and not a copy: mccl's collectives already take
`UnsafeMutableRawBufferPointer` + `DataType` + element count, which is exactly
what an `mlx::array`'s data pointer, dtype and size give you. The adapter needs
to:

1. Map MLX dtypes onto `DataType` (`float32`, `float16`, `bfloat16`, `int32`,
   `int8` are all present; anything else is an error, not a silent cast).
2. Ensure the array is evaluated and contiguous before handing over its pointer —
   a lazily-evaluated or strided array has no single buffer to send.
3. Hold one `Communicator` per MLX distributed group, and one `StreamID` per
   tensor if the caller wants `.topK` (see the residual rules above).
4. Bridge async to sync exactly as the C shim does — see `runBlocking` in
   `Sources/MCCLShim/CABI.swift`. MLX's `all_sum` is synchronous from the
   caller's point of view.

The natural place for it is a new target (`MCCLMLX`) depending on `MCCL`, kept
out of the core library so `MCCL` stays dependency-free. The C shim is the model
to copy: thin, no decisions, every rule stated once in `MCCL` and surfaced here
as an error.

**What "done" looks like:** `mcclbench` numbers and MLX-ring numbers for the same
collective, same message sizes, on the same cluster.

### Thunderbolt transport

**The seam is `Transport`** (`Sources/MCCL/Transport.swift`), three requirements:

```swift
public protocol Transport: AnyObject {
    var name: String { get }
    func listen(host: String, port: Int) throws -> Listener
    func connect(to address: PeerAddress, timeout: TimeInterval) throws -> Channel
}
```

Today Thunderbolt is reached as ordinary IP over the TB bridge, so `TCPTransport`
already works across it — what is missing is bulk DMA rather than a TCP stack in
the path. Everything above the transport is unchanged: `MeshFabric`, the
planner, the collectives and the codecs never learn which transport answered.

What a new transport must provide:

- **`listen` returns a `Listener`** whose `address` reports what was *actually*
  bound (ephemeral port resolution matters — `MeshFabric.makeGroup` binds with
  port 0 and reads the address back), and whose `accept(timeout:)` throws
  `MCCLError.timedOut` on expiry rather than blocking forever. Bring-up
  correctness depends on that timeout being honoured.
- **`connect` returns a `Channel`** and *retries until the deadline*. Cluster
  bring-up races are normal: a peer's listener may not be up when you dial. See
  `TCPTransport.connect` for the loop.
- **`Channel` moves whole buffers.** `sendBytes` writes all of it; `receiveBytes`
  fills all of it; both block. A short read is not a valid outcome.
- **Separate `sendQueue` and `receiveQueue`.** They must be distinct dispatch
  queues, because every ring step is "send my chunk forward while the previous
  rank's chunk arrives" and the full-duplex overlap is where the ring's
  bandwidth comes from.
- **`close` is idempotent** and safe from any thread.

`PeerAddress` is deliberately transport-agnostic: a non-IP transport may
reinterpret `host` as its own identifier. `LoopbackTransport` is the smallest
complete implementation to read (171 lines, including its own byte-queue
plumbing); `Tests/MCCLTests/TCPTransportTests.swift` shows the failure modes a
transport is expected to report.

### Rendezvous service

**What exists** (`Sources/MCCL/Rendezvous.swift`) is one round trip through rank
0: every rank binds its data listener, ranks 1…n-1 announce `(rank, address)` to
rank 0, and rank 0 sends the assembled table back down the same connections. That
is enough for `mcclCommInitRank` and nothing more.

**What a service would add**, roughly in order of usefulness:

1. **Discovery** — a rank finds the job without being told rank 0's address.
   Bonjour/mDNS is the obvious Mac-native answer, and `NetworkInterfaces` already
   enumerates and classifies the local interfaces to advertise on.
2. **A durable coordinator** — the current listener is closed as soon as the
   table is handed out (`defer { listener.close() }` in `serve`). A service keeps
   it, so a rank that restarts can re-announce.
3. **Membership change** — the table is `[PeerAddress]` indexed by rank, built
   once. Growing or shrinking a world means versioning that table and telling
   `MeshFabric` to add or drop channels, which it currently cannot do: `channels`
   is fixed at bootstrap.
4. **Reconnection** — a `Channel` that fails mid-collective is currently fatal to
   the operation. Recovery needs the collectives to be re-drivable from a known
   point, which is a larger change than the rendezvous itself.

Start at (1) and (2): they are additive, and neither requires touching the
collectives. The wire format is already versioned (`WireHeader.version`) and the
frame tags are namespaced, so a richer protocol can be added without breaking the
existing one.

### Non-blocking C calls

**What exists:** every exported symbol runs the collective to completion via
`runBlocking` in `Sources/MCCLShim/CABI.swift`, which parks the calling thread on
a `DispatchSemaphore` while a detached `Task` drives the async Swift API. The
Swift surface underneath is already `async`, and each `Communicator` already runs
its collectives on its own serial `workQueue`, so the machinery for overlap is
present — the C ABI simply does not expose it.

**What genuine non-blocking calls need:**

1. **A real `mcclStream_t`.** Today it is an opaque integer naming a *sequence*
   (which is all top-k's residual needs). An execution context would be a handle
   owning a queue of pending operations and a completion mechanism —
   `mcclStreamCreate` / `mcclStreamSynchronize` / `mcclStreamDestroy`.
2. **`mcclGroupStart` / `mcclGroupEnd`.** NCCL's batching primitive: collectives
   enqueued between them are fused into one launch. Without a device queue there
   is nothing to fuse *into*, which is why it is not there.
3. **Buffer ownership across the return.** The blocking design is what makes
   `Borrowed` safe (see its comment in `CABI.swift`): the C caller cannot free
   the memory underneath a collective that has already returned. Non-blocking
   calls need an explicit contract — the caller must not touch the buffers until
   `mcclStreamSynchronize`.
4. **Error delivery after the fact.** `mcclGetLastError(comm)` is per
   communicator and is written on the failing call. Deferred failures need to be
   reported at synchronise time instead.

This is the item most worth waiting on: it is only clearly worth the complexity
once there is a Metal command queue to enqueue against, and mccl deliberately
has no device queue in v0.

---

See [ARCHITECTURE.md](ARCHITECTURE.md) for why each of these is shaped the way it
is, and [WHITEPAPER.md](WHITEPAPER.md) for the design argument and the measured
evaluation.
