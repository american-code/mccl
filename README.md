# mccl

**Metal Collective Communications Library** — an NCCL-equivalent for Apple Silicon clusters.

MLX ships basic ring all-reduce over Thunderbolt/Ethernet, but there is no dedicated
collectives *library* for Mac clusters. mccl is that missing layer: infrastructure, not a
training framework.

## What it does (design goals)

- **Auto-tuned topology.** Measures actual TB5/USB4/Ethernet link bandwidth and latency
  between nodes (`mcclprobe`), then picks ring vs. tree vs. hierarchical per collective,
  per message size — instead of assuming a uniform ring.
- **In-flight wire compression.** On Mac clusters the interconnect is the bottleneck, not
  compute. mccl compresses/quantizes payloads per-hop on the wire (fp16 downcast, blockwise
  int8, top-k with error feedback) transparently to the caller.
- **Clean C/Swift API.** MLX, llama.cpp, and raw-Metal apps can all link against it
  independently. The Swift surface lives in `Sources/MCCL`; a C shim (`mccl.h`,
  `libmccl.dylib`) mirrors the `ncclComm_t` idiom so non-Swift runtimes can adopt it
  by renaming `nccl` to `mccl`.

## Documentation

- **[docs/USAGE.md](docs/USAGE.md)** — the practical guide. Probe a cluster, bring
  a communicator up three different ways, run every collective, use every codec,
  link from C, read `mcclbench` output. Every example in it was compiled and run
  before it was written down. Ends with concrete starting points for the four
  pieces that are not built yet.
- **[docs/WHITEPAPER.md](docs/WHITEPAPER.md)** — the design argument and the
  measured evaluation: why the interconnect rather than compute is the
  constraint, the closed-form ring/tree crossover, ratio-gap island detection,
  why top-k is an algorithm and not a codec, and what the numbers do and do not
  establish.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — component-level design notes.
- **[docs/COMPARISON.md](docs/COMPARISON.md)** — measured efficiency vs. published
  NCCL numbers, and the codec-crossover rule NCCL has no equivalent of.

## Status

Milestones 1–4 are implemented and tested; milestone 5 is partial — the C shim
and the benchmark harness are in, the MLX adapter is not.

**232 tests, all passing**, at 90.56% region / 94.43% function / 97.05% line
coverage. First real-hardware validation: two Mac Studio M1 Max
machines on a direct Thunderbolt cable negotiated at 20 Gb/s probed at **1.06 GB/s
and 167.8 µs**, with the Thunderbolt and Wi-Fi paths correctly classified from
interface media.

Collective throughput is measured on that cable rather than only over loopback.
`mcclbench` runs distributed (`--rank` / `--world-size`, joining through a rendezvous
token), and a 2-node all-reduce reaches **0.746 GB/s bus bandwidth at 16 MiB — 70.4%
of the probed link**. That sweep produced the rule the codecs are governed by — *a
codec pays only when the fabric's uncompressed all-reduce rate is below the codec's
own encode/decode ceiling* — and, with scalar encoders, a verdict against every
codec on Thunderbolt.

**The codecs are now vectorised, and the Thunderbolt verdict has flipped for two of
the three.** `CodecKernels` produces byte-for-byte the same frames (checked against
the scalar loops in both directions, and re-verified at the interface counters), at
8–12× the throughput for `downcast` and `int8`, 1.9× for top-k. Re-run on the same
cable against a same-session scalar control, `downcast` went from 0.79–1.04× to
**1.07–1.53×** against uncompressed and `int8/256` from 0.51–0.80× to
**1.27–1.71×**, while `topk/0.01` — the one codec whose ceiling is still below what
the cable delivers — stayed a loss, exactly as the rule requires. Over Wi-Fi the
codecs now collect almost their full wire reduction (downcast 1.99× of a possible
2.00×, int8 3.88× of 3.94×). See
[ARCHITECTURE.md § Measured: vectorised codecs](docs/ARCHITECTURE.md#measured-vectorised-codecs)
for the tables, the method, and the ceilings measured per chip.

**Working**

- **Transports.** `Transport` / `Listener` / `Channel` over POSIX sockets (`TCPTransport`)
  and in-process (`LoopbackTransport`). No external package dependencies. `MeshFabric`
  brings up a full rank-addressed mesh.
- **Probe.** `mcclprobe serve` / `mcclprobe measure` measure real pairwise bandwidth
  (large streaming transfers) and RTT (small ping-pong), including peer-to-peer links
  measured by the peers themselves. The resulting `Topology` is JSON-persistable, and
  `mcclprobe plan` replays the planner over a saved map. Link kinds now come from the
  interface a peer is actually reached on (`getifaddrs` + `SCNetworkInterface`), with
  throughput inference as the fallback for routed peers.
- **Collectives.** All-reduce, all-gather, broadcast, reduce and reduce-scatter across N
  ranks, for every `DataType` (fp32/fp16/bf16/int32/int8) and `ReduceOp`
  (sum/prod/min/max/avg).
- **Planner.** `TopologyPlanner.plan` picks tree, ring or hierarchical from the measured
  data. The tree/ring crossover is *solved* from measured latency and bandwidth, not
  tuned — there is no hardcoded byte threshold in the library. Mixed-speed fabrics are
  split into islands at the largest ratio gap in measured bandwidth. All three plans are
  executable.
- **Wire compression.** `.downcast` (fp32→fp16), `.int8Blockwise` (per-block absmax
  scale + int8 payload), and `.topK(fraction:)` with per-stream error feedback —
  each rank sends only its `k` largest-magnitude elements and carries the remainder
  forward, so a training run converges on the uncompressed answer instead of
  drifting away from it. The encoders are vector code (`CodecKernels`) and remain
  byte-compatible with the scalar frames, so a mixed-version world interoperates.
- **C shim.** `libmccl.dylib` plus a hand-written
  [`Sources/MCCLShim/include/mccl.h`](Sources/MCCLShim/include/mccl.h), shaped like
  NCCL: `mcclGetUniqueId` / `mcclCommInitRank` / `mcclAllReduce` / `mcclAllGather` /
  `mcclBroadcast` / `mcclReduceScatter` / `mcclReduce` / `mcclCommDestroy`, with
  `mcclResult_t` mapping one-to-one onto `MCCLError`. The test suite compiles a
  four-rank C client with `cc` and runs it against the built dylib.
- **Benchmarks.** `mcclbench` sweeps ring vs. tree vs. hierarchical, with and without
  each codec, from 1 KiB to 64 MiB — in one process, or as one rank of a real
  multi-machine world discovered through a rendezvous token. `--codec-bench` drops
  the ranks and the sockets and measures the codec kernels alone, which is the
  local half of deciding whether compression is worth switching on.

**Not yet**

- The MLX adapter, and comparative numbers against MLX's built-in ring.
- Thunderbolt-specific transport (bulk DMA rather than IP-over-TB).
- A rendezvous *service*. `Rendezvous` closes the gap the C ABI needs — one round
  trip through rank 0 — but there is no discovery, no membership change, and no
  recovery from a rank restarting.
- Non-blocking C calls.

Each of these has a seam already cut for it; [USAGE.md's "Implementing what is
missing"](docs/USAGE.md#implementing-what-is-missing) is enough to start any of
them.

## Build

```
swift build
swift test
```

No external package dependencies. Use `swift build` / `swift test` only —
`xcodebuild` is not supported here.

## Using the probe

```
# on each peer node
mcclprobe serve --port 7777

# from the driving node
mcclprobe measure node1:7777 node2:7777 node3:7777 --out cluster.json

# replay the planner over a saved map
mcclprobe plan cluster.json --bytes 4194304
```

`measure` prints the measured links, the derived tree/ring crossover, any fast-link
islands it found, and the plan it would choose at a range of message sizes. Worked
transcripts and the Swift equivalents:
[USAGE.md § Probe the cluster](docs/USAGE.md#1-probe-the-cluster).

## Using the library

```swift
import MCCL

// One rank per machine; every listener must be bound before anyone bootstraps.
let listener = try TCPTransport().listen(host: "0.0.0.0", port: 7000 + rank)
let comm = try Communicator.bootstrap(
    rank: rank, worldSize: 4, addresses: addresses, listener: listener,
    topology: try Topology.read(from: mapURL))

try await comm.allReduce(buffer, count: count, dataType: .float32,
                         op: .sum, compression: .downcast)
```

`Communicator.tcpGroup(worldSize:)` brings a whole world up inside one process over
loopback TCP, which is how the test suite runs 4 ranks against real sockets. If you
would rather hand out one token than a full address list, `Rendezvous.createUniqueID()`
on rank 0 and `Communicator.join(uniqueID:rank:worldSize:)` everywhere else does the
discovery for you.

### Gradient compression

`.topK(fraction:)` is stateful, and deliberately so. Each call adds the residual left
over from the previous one, sends the `k` largest-magnitude elements, and keeps the
rest:

```swift
// Two tensors reduced on one communicator need two streams, or they would
// share a residual.
try await comm.allReduce(gradients, count: n, dataType: .float32,
                         op: .sum, compression: .topK(fraction: 0.01),
                         stream: StreamID(layer))

comm.topKResidual(for: StreamID(layer))   // what is still being carried
```

A checkpoint that saves the model without the residual has thrown away part of the
gradient, which is why the residual is readable. The rules you have to know before
using it — one stream per tensor, choosing the fraction, what it refuses and why —
are in [USAGE.md § Compress the wire](docs/USAGE.md#4-compress-the-wire).

## Using the C API

```c
#include <mccl.h>

mcclUniqueId id;
mcclGetUniqueId(&id);            /* on rank 0; distribute the bytes out of band */

mcclComm_t comm;
mcclCommInitRank(&comm, nranks, id, rank);
mcclAllReduce(send, recv, count, mcclFloat32, mcclSum, comm, NULL);
mcclCommDestroy(comm);
```

```
swift build -c release
cc app.c -I Sources/MCCLShim/include \
         -L .build/release -lmccl -Wl,-rpath,$PWD/.build/release
```

Every call blocks until the collective has completed on this rank — NCCL's
synchronous-enqueue semantics, minus a device queue to enqueue onto. `mcclStream_t`
therefore names an independent *sequence* of collectives rather than an execution
context, which is what top-k's per-stream residual needs.
`Tests/MCCLTests/CProgram/mccl_smoke.c` is a complete working client, and
[USAGE.md § Using the C API](docs/USAGE.md#using-the-c-api) has a smaller one with
the exact `cc` line.

## Benchmarking

```
mcclbench                        # 1 KiB .. 64 MiB, 4 ranks, every algorithm and codec
mcclbench --quick                # 1 KiB .. 1 MiB, ring and tree
mcclbench --ranks 8 --collective allgather --codecs none,int8:128 --csv
```

Across machines, one process per rank, discovering each other through a rendezvous
token — the same grid, the same table, over a real cable:

```
lab-01$ mcclbench --rank 0 --world-size 2 --bind 169.254.152.222 \
                  --token-file /tmp/tb.id --sizes 64K,1M,64M
lab-02$ mcclbench --rank 1 --world-size 2 --bind 169.254.23.203 \
                  --token mccl1:f5a44d951e3024b2:169.254.152.222:60278
```

Pass `--bind` on any machine with more than one interface: it pins both the listener
and the address the token advertises, and without it the run may quietly measure a
different path than you meant. Only rank 0 prints, and the wall time it prints is the
slowest rank's for each point.

```
        size  algorithm     codec           iters          wall    GB/s alg    GB/s bus
---------------------------------------------------------------------------------------
       1 MiB  ring          none               50      1.310 ms       0.800       1.201
       1 MiB  ring          downcast           50      1.538 ms       0.682       1.023
       1 MiB  ring          int8/256           50      8.083 ms       0.130       0.195
       1 MiB  ring          topk/0.01          50      2.729 ms       0.384       0.576
       1 MiB  tree          none               50      2.656 ms       0.395       0.592
       1 MiB  tree          int8/256           50     11.002 ms       0.095       0.143
       1 MiB  hierarchical  none               50      2.361 ms       0.444       0.666
       1 MiB  hierarchical  topk/0.01          50      1.587 ms       0.661       0.991
```

"GB/s alg" is payload bytes over wall time; "GB/s bus" scales that by the traffic each
link actually carries (`2(n-1)/n` for all-reduce), which is the figure comparable to
NCCL's busbw and to MLX's ring.

Read the loopback numbers as a codec *cost* measurement, not a codec verdict. Loopback
has no wire: the "interconnect" is memory bandwidth, so every codec pays its encode cost
and buys nothing back, and int8 duly comes last. What that table gives you is each
codec's CPU price per byte.

The cluster numbers are in, and the price was the whole story: a codec pays only when
the fabric's uncompressed all-reduce rate falls below that codec's own encode/decode
ceiling. `mcclbench --codec-bench` measures that ceiling directly, with no ranks and
no sockets:

```
        size  codec            ratio    GB/s enc    GB/s dec    GB/s r/t    ns/elem
-----------------------------------------------------------------------------------
      16 MiB  none             1.00x      48.370      32.114      19.300       0.21
      16 MiB  downcast         2.00x      47.731      56.916      25.960       0.15
      16 MiB  int8/256         3.94x       7.964      32.796       6.408       0.62
      16 MiB  topk/0.01       50.00x       1.649      62.625       1.606       2.49
```

Compare "GB/s r/t" against the fabric's uncompressed rate and the verdict follows.
On a 1.06 GB/s Thunderbolt cable the scalar encoders lost that comparison and the
vectorised `downcast` and `int8` win it; `topk` still does not. See
[ARCHITECTURE.md § Measured: vectorised codecs](docs/ARCHITECTURE.md#measured-vectorised-codecs).

Full flag reference and a longer discussion of alg vs. bus bandwidth:
[USAGE.md § Benchmarking](docs/USAGE.md#benchmarking-with-mcclbench).

## License

Apache-2.0 — see [LICENSE](LICENSE).
