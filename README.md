# mccl

**Metal Collective Communications Library** — an NCCL-equivalent for Apple Silicon clusters.

MLX ships basic ring all-reduce over Thunderbolt/Ethernet, but there is no dedicated
collectives *library* for Mac clusters. mccl is that missing layer: infrastructure, not a
training framework.

**The thesis.** The job NCCL does on NVIDIA hardware had no incumbent on Apple silicon,
so mccl fills it in NCCL's own idiom (`mccl.h`, `ncclComm_t`-shaped) and adds the axis
NCCL has no equivalent of: in-band lossy compression with exactness bookkeeping, switched
on by a measured rule — *a codec pays only when the fabric's uncompressed all-reduce rate
is below that codec's own encode/decode ceiling* — which puts the advantage exactly where
CUDA clusters are weakest, on commodity interconnects. The planner likewise solves the
fabric from probed bandwidth and latency instead of assuming it. Where the CUDA stack is
still ahead — RDMA fabrics, scale beyond the three nodes validated here, ecosystem
maturity — is stated alongside the case, in
[WHITEPAPER.md §3](docs/WHITEPAPER.md#3-value-proposition-competing-with-the-cuda-cluster-stack).

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
  link from C, train data-parallel with MLX, read `mcclbench` output. Every
  example in it was compiled and run before it was written down. Ends with
  concrete starting points for the three pieces that are not built yet.
- **[docs/WHITEPAPER.md](docs/WHITEPAPER.md)** — the design argument and the
  measured evaluation: why the interconnect rather than compute is the
  constraint, the closed-form ring/tree crossover, ratio-gap island detection,
  why top-k is an algorithm and not a codec, and what the numbers do and do not
  establish.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — component-level design notes.
- **[docs/COMPARISON.md](docs/COMPARISON.md)** — measured efficiency vs. published
  NCCL numbers, and the codec-crossover rule NCCL has no equivalent of.

## Status

Milestones 1–4 are implemented and tested; milestone 5 is nearly complete — the C
shim, the benchmark harness and the MLX adapter are all in, and the one piece
still missing is blocked upstream rather than here.

**309 tests, all passing** (275 for the core library, 34 for the MLX adapter, which
lives in its own test target so that `MCCL`'s tests never acquire an MLX
dependency), run in CI on macOS arm64 on every push. First real-hardware
validation: two Mac Studio M1 Max machines on a
direct Thunderbolt cable negotiated at 20 Gb/s probed at **1.06 GB/s and 167.8 µs**,
with the Thunderbolt and Wi-Fi paths correctly classified from interface media.

**Data-parallel training works, and is provably the same computation as training
on one machine.** `MCCLMLX` lets an MLX program hand mccl an `MLXArray`;
`mccltrain` trains an MLP data-parallel across the world, averaging gradients
through one fused all-reduce. Its correctness proof is an equivalence rather than
a tolerance: a 2-rank run on half-batches must follow the same loss trajectory as
a single process on the combined batch. Over 200 steps in one process it does to
**4.8e-07**, one unit in the last place of fp32; across the two lab machines over
the Thunderbolt cable, to **3.6e-05** (9.7e-06 relative) — the residue being two
different physical GPUs reducing matmuls in different orders.

**And it is honestly slower than one machine, which is the useful part.** At batch
256 two nodes run at 0.17–0.43× of one, because the gradient all-reduce dwarfs the
compute; the codecs help where the rule says they should (`downcast` 1.25× once the
payload is bandwidth-bound) and do not rescue it. The fix is not a bigger model —
communication and compute both scale with parameter count — it is a bigger batch,
which scales only compute. Measured on the large model, the crossover falls between
batch 1,024 and 4,096, and by batch 16,384 the second machine earns **1.45×**. See
[USAGE.md § Training with MLX](docs/USAGE.md#training-with-mlx).

**The planner's central claim now has its first hardware evidence, at n = 3.**
Three ranks (2× M1 Max + M1 Pro) over Wi-Fi: the tree wins where the collective is
latency-bound (1.87× at 16 KiB, 1.58× at 64 KiB, 1.17× at 256 KiB) and the ring wins
where it is bandwidth-bound (1.32× at 1 MiB, 1.26× at 16 MiB), with the switch falling
between 256 KiB and 1 MiB. That is the qualitative claim the library is built on, and
it holds. **The constant does not:** the closed-form crossover, fed this fabric's
probed α and B, predicts ~75 KB — low by 4–13×, consistent with the model assuming a
single bandwidth while Wi-Fi's effective bandwidth moves twentyfold across the sweep.
Right shape, wrong constant.

**And island detection now has hardware evidence too, on a genuinely mixed fabric.**
The same three machines, but with the two Studios on their Thunderbolt cable and the
laptop reachable from either only over Wi-Fi — a 20.76× ratio inside one world. The
probe finds the islands unaided (`[0,1] [2]`), and the hierarchical plan wins at every
size from 256 KiB to 16 MiB across two runs, reaching **1.74× the ring at 16 MiB**
(449 ms against 783 ms) and 2.49× the best all-Wi-Fi time at the same size. On this
fabric the closed-form crossover also lands correctly — 139.9 KiB predicted, 64–256 KiB
measured — because a mixed fabric's bottleneck really is one constant. At 64 MiB the
tree takes it back by 8%, on one run; recorded as it came out.

Forming that world at all needed **per-pair addressing**: a fabric spanning a
Thunderbolt island and a Wi-Fi bridge has no single address per rank that all of its
peers can dial, so every rank advertises all of its addresses and every *pair* selects
and then *dials* its own best path. See
[ARCHITECTURE.md § Per-pair addressing](docs/ARCHITECTURE.md#per-pair-addressing).

Still untested on hardware: two islands of two ranks (the lab has three machines), and
anything at all about scale beyond n = 3. Note also that the earlier n = 2 result — ring
beating tree by 1.72× — supports a narrower claim than it looks: at two ranks the ring
degenerates to a single full-duplex point-to-point exchange, its best case, so that
measurement says *the executor exploits full duplex*, not that topology selection works.

Collective throughput is measured on that cable rather than only over loopback.
`mcclbench` runs distributed (`--rank` / `--world-size`, joining through a rendezvous
token), and a 2-node all-reduce reaches **0.746 GB/s bus bandwidth at 16 MiB — 70.4%
of the probed link**.

**The result that generalises is a rule, and it survived being tested from both
sides.** *A codec pays only when the fabric's uncompressed all-reduce rate is below
that codec's own encode/decode ceiling.* Both terms are measurable on the machine in
front of you (`mcclbench` for the fabric, `mcclbench --codec-bench` for the ceiling),
so the rule decides whether to switch compression on rather than deferring it to a
benchmark. It is also falsifiable, and vectorising the codecs was the experiment that
tested it: lift a ceiling from below the fabric's rate to above it and that codec must
flip to a win; leave one below and it must keep losing.

Both halves held. `downcast` went from 0.79–1.04× to **1.07–1.53×** against
uncompressed on the same cable, and `int8/256` from 0.51–0.80× to **1.27–1.71×** —
while `topk/0.01`, whose ceiling rose 1.9× and is *still* below what the cable
delivers, stayed a loss. The kernel speedups behind that (8–12× for `downcast` and
`int8`, 1.9× for top-k, `CodecKernels` producing byte-for-byte the same frames as the
scalar loops, checked in both directions and re-verified at the interface counters)
are the instrument, not the finding. Over Wi-Fi every codec wins either way and now
collects almost its full wire reduction (downcast 1.99× of a possible 2.00×, int8
3.88× of 3.94×). See
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

- **MLX adapter.** `MCCLMLX` — a separate target and product, so that `MCCL` itself
  stays dependency-free — extends `Communicator` with `allReduce`/`allGather`/
  `broadcast`/`reduce`/`reduceScatter` over `MLXArray`, plus `averageGradients`,
  which flattens a model's gradients into one buffer and all-reduces them once.
  Ten separate all-reduces would spend 1.7 ms on round trips alone at MLP scale;
  fusing spends one. `mccltrain` is the demo, and the loss-trajectory equivalence
  above is its proof.
- **Reduction kernels.** `Kernels.reduce` had the same runtime-switch-in-the-loop
  pattern that cost the codecs an order of magnitude, and the same fix applied:
  4.4–5.1× on fp32, 21× on int8. `Kernels.scale` turned out to have been
  auto-vectorising all along and was left alone — measured, and recorded in the
  source so it does not get "optimised" again.

**Not yet**

- **A same-language head-to-head against MLX's built-in ring.** Blocked upstream:
  mlx-swift does not build MLX's distributed backends at all (its `Package.swift`
  excludes `mlx/distributed/{ring,mpi,nccl,jaccl}`) and exposes no distributed
  API. Python MLX does ship the ring, so a cross-language measurement over the
  same cable is possible — it compares two runtimes rather than two schedulers.
- **The hierarchical plan on real hardware**, which needs four machines with two
  fast islands. n = 3 validated ring-versus-tree selection; nothing has yet exercised
  island detection, and nothing has tested the planner on a mixed-speed fabric.
- **A crossover constant that matches measurement.** The closed form has the right
  shape and is 4–13× low on Wi-Fi, because it assumes one bandwidth.
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

`MCCL` has no external package dependencies. One target does, and is kept apart on
purpose: `MCCLMLX` (and the `mccltrain` demo built on it) depends on `mlx-swift`.
Nothing else in the package can see it — depend on `MCCL` and you get the same
dependency-free library.

Use `swift build` / `swift test` only — `xcodebuild` is not supported here.

**Anything touching MLX needs one more step**, including `swift test`:

```
Tools/fetch-metallib.sh
```

mlx-swift compiles MLX's C++ core under SwiftPM but does not build `mlx.metallib`
— that is Xcode's build system, which this project does not use. Without it MLX
aborts on its first GPU operation. The script fetches the version-matched shader
library from the `mlx-metal` pip wheel and installs it beside the built binaries.
Tests that need it skip with this instruction if it is absent; the core library's
275 tests never load MLX at all.

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

## Data-parallel training with MLX

```swift
import MCCL
import MCCLMLX
import MLXNN

let (loss, grads) = valueAndGrad(model: model, lossFunction)(model, x, y)
eval(loss, grads)                                     // mccl reduces bytes, not graphs

let averaged = try averageGradients(grads, comm: comm) // one fused all-reduce, .avg
SGD.step(model: model, gradients: averaged, learningRate: lr)
```

`averageGradients` flattens the whole gradient tree into one contiguous buffer with
a rank-independent layout (sorted parameter paths, fixed dtype order), all-reduces it
once, and slices it back. One all-reduce per tensor would spend a 167.8 µs round trip
each — 1.7 ms for a ten-tensor model, more than the backward pass itself.

Individual collectives are there too, as an extension on `Communicator`:

```swift
let summed = try comm.allReduce(gradient, op: .sum, compression: .downcast)
let all    = try comm.allGather(gradient)     // [worldSize] + gradient.shape
```

`mccltrain` is the working demo — `--verify` proves a 2-rank run is the same
computation as a single-process one, and `--rank`/`--token` runs it across machines.
Full guide, including the copy accounting and the metallib step:
[USAGE.md § Training with MLX](docs/USAGE.md#training-with-mlx).


## Benchmarking

```
mcclbench                        # 1 KiB .. 64 MiB, 4 ranks, every algorithm and codec
mcclbench --quick                # 1 KiB .. 1 MiB, ring and tree
mcclbench --ranks 8 --collective allgather --codecs none,int8:128 --csv
```

Across machines, one process per rank, discovering each other through a rendezvous
token — the same grid, the same table, over a real cable:

```
studio-a$ mcclbench --rank 0 --world-size 2 --bind 169.254.152.222 \
                    --token-file /tmp/tb.id --sizes 64K,1M,64M
studio-b$ mcclbench --rank 1 --world-size 2 --bind 169.254.23.203 \
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
