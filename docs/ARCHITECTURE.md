# mccl architecture

## Layering

```
MLX / llama.cpp / user code
        │  (C API: mccl.h / libmccl.dylib; Swift API: MCCL module)
        ▼
Collectives      all-reduce, all-gather, broadcast, reduce-scatter
        ▼
Planner          ring vs. tree vs. hierarchical, chosen per (topology, message size)
        ▼
Wire layer       framing, in-flight compression/quantization, error feedback state
        ▼
Transports       Thunderbolt (peer-to-peer IP over TB), Ethernet (TCP/QUIC), loopback
```

## Probe

`mcclprobe serve` on each node; `mcclprobe measure` drives pairwise bandwidth and RTT
measurements over every discovered interface, producing a `Topology` with measured
`Link` values. Persist the map; re-probe on membership change or on demand.

Bandwidth comes from large streaming transfers (`sink` frames, each acknowledged, after
a warm-up chunk so TCP's window is already open); RTT is the *minimum* over many
8-byte ping-pongs, since scheduling noise only ever adds time. A serving node will also
probe a third node on request (`probePeer`), so `measure` builds a true pairwise map
rather than a star centred on the driving host.

Link `Kind` comes from the interface the peer is actually reached on.
`NetworkInterfaces.local()` enumerates local interfaces with `getifaddrs` (addresses,
netmasks, `ifi_type`, `ifi_baudrate`, flags) and asks SystemConfiguration for the
user-visible name, which is what separates a Thunderbolt bridge from any other `en` or
`bridge` device. A peer is attributed to the interface whose subnet contains it, longest
mask first.

Media settles ethernet / wifi / loopback outright. It cannot settle TB4 vs. TB5 — both
are ordinary IP links to every API on the machine — so `NetworkInterface.linkKind`
consults the measured bandwidth for the *generation* only. A peer that no local subnet
owns (several routed hops away) falls back to `TopologyProbe.inferKind`, the pure
throughput heuristic.

The label remains cosmetic either way: the planner reasons over the measured numbers and
never over the `Kind`. What the interface buys is honesty — a congested Thunderbolt
bridge measuring 900 MB/s is still Thunderbolt, and fast Wi-Fi is never a cable.

## Planner

- Small messages (latency-bound): tree — O(log n) hops.
- Large messages (bandwidth-bound): ring — each link carries `2(n-1)/n` of the data.
- Mixed fabrics (e.g. TB5 pairs bridged by 10GbE): hierarchical — ring inside each
  fast island, tree across islands, one designated leader per island.
- Threshold between regimes computed from measured latency/bandwidth, not hardcoded.

The crossover is solved, not tuned. With worst-case per-hop latency `α` and bottleneck
bandwidth `B` over `n` ranks:

```
t_ring = 2(n-1)·α + 2·(n-1)/n·S/B
t_tree = 2·log2(n)·α + 2·log2(n)·S/B
S*     = α·B·(log2 n − (n−1)) / ((n−1)/n − log2 n)
```

Both factors are ≤ 0 for n ≥ 2, so `S* ≥ 0`; at n = 2 it is 0, because a two-rank tree
*is* a ring. `TopologyPlanner.crossoverBytes` is exactly this expression, and it is the
only size threshold in the library — there is no hardcoded byte constant anywhere in
algorithm selection.

Islands are found by splitting measured bandwidths at their largest *ratio* gap (the cut
sits at the geometric mean of the straddling pair) and taking connected components of
the fast edges. A fabric only counts as mixed-speed once `max/min ≥ 3x`
(`TopologyPlanner.heterogeneityRatio`), which separates jitter on one fabric from two
genuinely different fabrics. Ratio-based splitting means no link speed is ever named in
the code.

Execution mapping:

| plan | all-reduce | broadcast | all-gather / reduce-scatter |
|---|---|---|---|
| `.ring` | ring reduce-scatter + all-gather | chain along the ring | ring |
| `.tree` | binomial tree reduce + broadcast | binomial tree, re-rooted at the caller's root | ring |
| `.hierarchical` | intra-island ring → leaders-only ring → intra-island tree broadcast | root → island leader → tree across leaders → tree within each island | ring over all ranks |

All-gather and reduce-scatter are bandwidth-shaped by construction — every rank must
move `(n-1)` chunks whatever the algorithm — so they always ring, taking only the ring
*order* from the plan.

## Wire compression

Per-hop, transparent to the caller's dtype. The codec id travels in the frame header, so
a receiver always decodes with the scheme the sender actually used. A scheme that cannot
help a given dtype degrades to raw rather than erroring — `.downcast` on an fp16 buffer
is already done, and quantising an int32 accumulator would be wrong.

- `downcast`: fp32 → fp16 on the wire (2 bytes/element). Relative error ≤ 2⁻¹¹.
- `int8Blockwise(blockSize:)`: per-block absmax scales, int8 payload
  (`4·⌈n/blockSize⌉ + n` bytes). Absolute error ≤ blockAbsmax/254.
- `topK(fraction:)`: **not a per-hop codec** — it replaces the algorithm. See below.

### Top-k with error feedback

Top-k is the one stateful thing in mccl. `Communicator` carries a `ResidualStore`: one
`[Float]` per `StreamID`, holding what the previous call declined to send.

One `.topK` all-reduce is:

1. `work = buffer + residual`. Nothing is ever discarded, only delayed.
2. Select the `k = ⌈fraction·n⌉` largest magnitudes of `work` (quickselect, O(n)
   average — selection is on the hot path and `n` is the whole tensor). Those go on the
   wire; `work` minus them becomes the new residual.
3. All-gather the fixed-size sparse blocks — `k` entries from every rank — and sum them
   locally.

Step 3 is an all-gather rather than a ring reduce-scatter *on purpose*. A ring would
have to re-sparsify each partially reduced chunk on every hop, and the mass those hops
drop belongs to no rank's residual, so the result would be biased in a way error
feedback cannot repair. Gathering the sparse blocks instead makes the reduction exact
given the sparsification, which is the property the convergence argument needs.

Wire cost is `n·(4 + k·(4 + w))` bytes per rank against `2(n-1)/n·count·w` for a dense
ring, so it wins whenever `k` is a small fraction of `count`. Block layout is
`[nnz][indices][values]` — self-describing, so a receiver is never told `k` out of band,
and uniform across ranks, so an ordinary ring all-gather can move it.

Restrictions, each because the alternative would be silently wrong:

- All-reduce only. The other collectives cannot account for the dropped mass, so they
  throw `unsupportedCompression`.
- `.sum` / `.avg` only. Top-k drops elements, and only for a sum are the missing
  elements identities of the operation; a top-k `min` would return the min of the
  selection.
- Floating-point dtypes only. Magnitude sparsification of an int32 accumulator is
  meaningless.
- Residuals are `Float` whatever the caller's dtype: an fp16 residual would flush the
  small values it exists to preserve straight back to zero.
- The residual is keyed by `(StreamID, elementCount)`. Two tensors reduced on one
  communicator need two streams, and a changed element count starts a fresh history.
  `Communicator.topKResidual(for:)` exposes it, because a checkpoint saved without the
  residual has thrown away part of the gradient.

Open research question (ties into the AFP interpretability thread): do lossy
collectives change interpretability guarantees for distributed inference? mccl knows
the exact partition boundaries and compression error per hop, so it can emit ground
truth for that analysis.

## C API

`Sources/MCCLShim/include/mccl.h`, built as `libmccl.dylib` from the `mccl` dynamic
product. A runtime that already speaks `ncclComm_t` should need renaming and nothing
else.

```c
mcclResult_t mcclCommInitRank(mcclComm_t* comm, int nranks, mcclUniqueId id, int rank);
mcclResult_t mcclAllReduce(const void* send, void* recv, size_t count,
                           mcclDataType_t dtype, mcclRedOp_t op,
                           mcclComm_t comm, mcclStream_t stream);
```

Three places where mccl differs from NCCL, and why:

- **Blocking.** NCCL enqueues onto a CUDA stream and returns; mccl v0 has no device
  queue to enqueue onto, so every call runs the collective to completion (dispatch
  semaphore over the async Swift API) and returns when the buffers are safe to touch.
- **`mcclStream_t` is not an execution context.** It names an independent *sequence* of
  collectives, which matters only to top-k's per-stream residual. `mcclStreamFromId`
  builds one from an integer.
- **`mcclCommInitRank` is a `static inline` over `mcclCommInitRankFromId`.** The
  by-value 128-byte `mcclUniqueId` in NCCL's signature has a different ABI on arm64 and
  x86_64; the exported symbol takes the token by pointer, and the header restores the
  idiomatic call site. Source-compatible, and portable.

`mcclResult_t` is `MCCLError` one-to-one — the enum was written flat for exactly this —
and `mcclDataType_t` is `DataType.wireCode`, the discriminator already in every frame
header, so a C caller and a Swift caller cannot disagree about what is in a frame.
`mcclGetErrorString` names a code; `mcclGetLastError(comm)` returns the full Swift error
text, offending values included. mccl-specific extras: `mcclAllReduceCompressed` (the
codecs NCCL has no equivalent for) and `mcclCommPlanDescription` (ask the planner what
it chose).

`Tests/MCCLTests/CProgram/mccl_smoke.c` is a complete four-rank client; `CShimTests`
compiles it with `cc` against the built dylib and runs it, which is the only check that
proves at once that the header is valid C, that its declarations match the exported
symbols, and that a non-Swift process can link the library.

## Rendezvous

`Communicator.bootstrap` takes explicit addresses and has no opinion about how a cluster
is launched. The C ABI cannot: `mcclCommInitRank` gets one token and nothing else. So
`Rendezvous` does one round trip through rank 0.

1. Every rank binds its own data listener first, so no dial can arrive before its target
   is accepting.
2. Ranks 1…n-1 connect to the token's address and announce `(rank, data address)`.
3. Rank 0 waits for all n-1 announcements, then sends the assembled table back down the
   same connections.

The token is printable text — `mccl1:<nonce>:<host>:<port>` — precisely so it can travel
through a shell, a job script or an env var. The nonce keeps two concurrent jobs on one
host apart.

What this is **not** is a rendezvous *service*: no discovery, no membership change, no
recovery from a rank restarting. Rank 0 must be up, and `mcclGetUniqueId` must be called
in the process that will host rank 0.

## Benchmarks

`mcclbench` (harness in `Sources/MCCLBenchmarks`, so the test suite drives it directly
rather than scraping stdout) sweeps message size x algorithm x codec over N ranks in one
process. One world is brought up for the whole sweep and the plan is pinned per point
via `Communicator.planOverride`, so the table compares algorithms and not bootstrap
costs. Iteration counts come from a per-point byte budget, which keeps 1 KiB points
statistically meaningful without letting 64 MiB points own the wall clock.

It reports both algorithmic bandwidth (payload over wall time) and bus bandwidth
(scaled by `2(n-1)/n` for all-reduce), the latter being what compares to NCCL's busbw
and to MLX's ring. Over loopback it is really measuring each codec's CPU price per byte,
since there is no wire to save; that number is what predicts where a codec starts
winning on a link an order of magnitude slower than memory.

`--rank` / `--world-size` switch it to *distributed* mode: the process becomes one rank
of a real world, joining the others through the same `Rendezvous` token the C shim's
`mcclCommInitRank` uses, and the table then describes an actual cable. Rank 0 creates
the token (`--token-file` or `--emit-token`); the others join with `--token`. Two
details matter on a multi-homed machine and neither is optional:

- **`--bind <address>`.** Without it the rendezvous binds a wildcard and advertises
  whatever `NetworkInterfaces.preferredLocalAddress()` likes, and the run silently
  measures a different interface. `--bind` pins the listener *and* what the token
  advertises. Never use an mDNS `.local` name for this — it resolves over whichever
  path the resolver feels like.
- **The grid must match on every rank.** The sweep is a pure function of the options,
  so it is fingerprinted and agreed with a min/max all-reduce at bring-up: a mismatched
  launch is an error rather than a deadlock on the first divergent collective.

The reported wall time is the *slowest* rank's for each point, agreed with a
one-element max all-reduce after the point completes. Rank 0's own clock would
understate the cost whenever rank 0 is the rank that waits. A barrier runs after each
point's warm-up and before the clock starts, so one rank's setup stays out of another
rank's stopwatch.

## Measured: 2-node Thunderbolt

The first collective throughput numbers from real interconnect, and the first verdict
on the wire codecs that is not an extrapolation from loopback.

**Setup.** Two Mac Studio M1 Max (`lab-01`, `lab-02`) joined by a direct Thunderbolt
cable, negotiated at 20 Gb/s, probed by `mcclprobe measure` at **1.06 GB/s and
167.8 µs**. `mcclbench` release builds, rank 0 on `lab-01` bound to `169.254.152.222`,
rank 1 on `lab-02` bound to `169.254.23.203`, all-reduce fp32 sum, ring.

That the traffic really crossed the cable is checked rather than assumed: the token
was verified to advertise `169.254.152.222` before rank 1 was started, and over one
sweep `lab-01`'s `en4` byte counters moved **+1.11 GB in and +1.11 GB out** while `en0`
stayed at zero.

### All-reduce over the Thunderbolt cable

Bus bandwidth, GB/s, best of five sweeps; with n = 2 the bus factor `2(n-1)/n` is 1, so
these are also the algorithmic figures. Parenthesised is the fraction of the
probe-measured 1.06 GB/s link.

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 64 KiB | 0.122 (11.5%) | 0.129 (12.2%) | 0.089 (8.4%) | **0.158 (14.9%)** |
| 256 KiB | **0.299 (28.2%)** | 0.298 (28.1%) | 0.138 (13.0%) | 0.259 (24.4%) |
| 1 MiB | **0.577 (54.4%)** | 0.492 (46.4%) | 0.158 (14.9%) | 0.278 (26.2%) |
| 4 MiB | **0.651 (61.4%)** | 0.504 (47.5%) | 0.162 (15.3%) | 0.285 (26.9%) |
| 16 MiB | **0.746 (70.4%)** | 0.505 (47.6%) | 0.190 (17.9%) | 0.271 (25.6%) |
| 64 MiB | **0.616 (58.1%)** | 0.462 (43.6%) | 0.161 (15.2%) | 0.274 (25.8%) |

**Uncompressed all-reduce reaches 70.4% of the probed link at 16 MiB.** The probe
measures a one-directional stream; an all-reduce additionally reduces every byte it
receives and turns the link around twice, so losing 30% to that is a good result and
says the framing and the ring schedule are not where the time goes. It falls back to
58% at 64 MiB, where the buffers stop fitting comfortably in cache.

Small sizes are latency-bound, exactly as the planner's model says: at 64 KiB the
167.8 µs RTT times two ring steps is already ~340 µs of a 539 µs call, and only 11.5%
of the link is reachable.

### The codecs really do shrink the wire

Measured directly from `lab-01`'s `en4` output counter over five 64 MiB collectives,
so this is bytes on the cable and not an accounting of what the encoder believes:

| codec | bytes sent | vs. uncompressed | payload sent |
|---|---|---|---|
| none | 336.1 MB | 1.00× | 67.1 MB per call |
| downcast | 168.1 MB | **2.00×** | fp32 → fp16 exactly halves it |
| int8/256 | 85.3 MB | **3.94×** | 4× minus the per-block scales |
| topk/0.01 | 6.7 MB | **49.9×** | 1% of elements, as index+value pairs |

### Codec verdict on a ~1 GB/s link: not worth it

Speed-up against `none` at the same size (>1 is a win). Thunderbolt, best of five:

| size | downcast | int8/256 | topk/0.01 |
|---|---|---|---|
| 64 KiB | **1.06×** | 0.73× | **1.30×** |
| 256 KiB | 1.00× | 0.46× | 0.87× |
| 1 MiB | 0.85× | 0.27× | 0.48× |
| 4 MiB | 0.77× | 0.25× | 0.44× |
| 16 MiB | 0.68× | 0.25× | 0.36× |
| 64 MiB | 0.75× | 0.26× | 0.45× |

**On a 1.06 GB/s Thunderbolt link the codecs lose, and the reason is not subtle.**
`downcast` halves the bytes on the wire — verified above — and is still 25–32% slower
everywhere above 256 KiB. The hoped-for result was the opposite: the loopback tables
predicted a codec would start paying once the link was an order of magnitude slower
than memory, and 1 GB/s is that link. It is not enough.

What the table actually shows is each codec's own CPU ceiling. Read down the compressed
columns: `downcast` plateaus at ~0.50 GB/s of payload, `topk/0.01` at ~0.28, `int8/256`
at ~0.16, and none of them improves with size the way `none` does. Those are not link
numbers — they are the rate at which this M1 Max can encode and decode, and they are
flat because the codec, not the cable, is the bottleneck. The rule that falls out is
worth more than the table:

> A codec is worth applying only when the fabric's *uncompressed* all-reduce rate is
> below that codec's own encode/decode ceiling. On this hardware those ceilings are
> ~0.50 GB/s (downcast), ~0.28 GB/s (top-k 1%), ~0.16 GB/s (int8/256).

Thunderbolt runs uncompressed at 0.62–0.75 GB/s, above all three, so every codec loses.
The two exceptions in the table are the same rule seen from the other side: at 64 KiB
the run is latency-bound rather than bandwidth-bound, there is no 0.5 GB/s of encoding
to pay for so little data, and shrinking the message shortens the round trips —
`topk/0.01` wins 1.30× and `downcast` 1.06×.

The ceilings are an artefact of scalar code, not of the schemes. `Kernels` and the
codecs are plain Swift loops; `downcast` at 0.50 GB/s is roughly 3 ns per element for
what should be a few vector instructions. Vectorising them is the single change that
would move this verdict, and this table is the argument for doing it: a 4× faster
`downcast` would clear Thunderbolt's uncompressed rate and start winning.

### Wi-Fi contrast: on a slow link every codec wins

The same sweep between the same two machines over their Wi-Fi interfaces
(`192.168.1.250` ↔ `192.168.1.238`), best of three. Bus bandwidth in GB/s, with the
speed-up against `none` in parentheses:

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 64 KiB | 0.008 | 0.012 (1.42×) | 0.011 (1.27×) | **0.029 (3.41×)** |
| 256 KiB | 0.027 | 0.032 (1.21×) | 0.037 (1.38×) | **0.067 (2.53×)** |
| 1 MiB | 0.044 | 0.071 (1.63×) | 0.075 (1.71×) | **0.132 (3.01×)** |
| 4 MiB | 0.054 | 0.100 (1.84×) | 0.105 (1.94×) | **0.274 (5.05×)** |
| 16 MiB | 0.054 | 0.103 (1.91×) | 0.126 (2.33×) | **0.295 (5.44×)** |
| 64 MiB | 0.052 | 0.099 (1.89×) | 0.110 (2.11×) | **0.256 (4.90×)** |

Every codec wins at every size, and `downcast` wins by almost exactly the 2.00× its
wire reduction predicts — on a 0.05 GB/s path the encode cost is entirely hidden behind
the transfer. This is the same rule: 0.052 GB/s is far below all three ceilings.

It also puts a number on the crossover. Between 0.05 GB/s (all codecs win by ~2–5×) and
0.62 GB/s (all codecs lose by 25–75%) lies each codec's ceiling, and the planner has
everything it needs to choose: it already measures link bandwidth, and the ceilings are
a property of the local machine that a one-off calibration could measure the same way
the probe measures the cable. **Compression should be a planned decision, not a caller's
flag** — that is the design consequence of this table, and the flag is currently the
only way in.

### Two footnotes on method

**Ring is not tree at n = 2.** The expectation going in was that with two ranks the two
plans degenerate to the same thing. They move the same total bytes, but they are not the
same schedule, and the difference is measurable (best of four paired runs):

| size | ring | tree |
|---|---|---|
| 1 MiB | 3.014 ms (0.348 GB/s) | 3.485 ms (0.301 GB/s) |
| 16 MiB | 27.875 ms (0.602 GB/s) | 47.895 ms (0.350 GB/s) |

The ring's reduce-scatter and all-gather each move `N/2` in *both* directions at once,
so it uses the link full-duplex. The binomial tree sends `N` up to the root, reduces,
then sends `N` back down — strictly serialised, half-duplex, and 1.7× slower at 16 MiB.
The planner already prefers the ring for large messages on this topology; this is the
measurement that says it is right to, even at n = 2.

**Noise.** `lab-01` is a shared machine and was running an unrelated inference workload
(~40% of one core) throughout. Worst-case points ran up to 5× the best in the same
sweep, which is why every figure above is the best of several full sweeps rather than a
single run, and why medians are recorded alongside in the raw output. The verdict is not
sensitive to the choice: `none` beats every codec above 256 KiB on Thunderbolt in both
the best-of and the median-of statistic, and loses to every codec on Wi-Fi in both.

## Milestones

- [x] **1. Probe:** pairwise bandwidth/latency measurement + persisted topology map.
      `ProbeServer` / `TopologyProbe`, JSON-persistable `Topology`, driven by
      `mcclprobe serve` / `mcclprobe measure` / `mcclprobe plan`. Interface-media
      discovery via `getifaddrs` + `SCNetworkInterface` is in — see §Probe.
- [x] **2. TCP transport + ring all-reduce.** `Transport`/`Listener`/`Channel` over
      POSIX sockets, plus an in-process `LoopbackTransport`, plus `MeshFabric` bring-up.
      Went past the milestone: all-reduce, all-gather, broadcast, reduce and
      reduce-scatter, for every `DataType` and `ReduceOp`, over N ranks rather than 2.
- [x] **3. Planner with measured thresholds; tree + hierarchical.**
      `TopologyPlanner.plan` / `.analyze` / `.crossoverBytes`, with tree and
      hierarchical plans actually executable.
- [x] **4. Wire compression.** `downcast` and `int8Blockwise` round-trip within their
      error bounds; `topK` carries per-stream error-feedback residuals and converges on
      the uncompressed result over repeated calls.
- [~] **5. C shim + MLX adapter; benchmark against MLX's built-in ring.**
      The C shim ships (`mccl.h`, `libmccl.dylib`, validated by a compiled C client) and
      `mcclbench` compares every algorithm against every codec across message sizes,
      now in-process *or* across machines (`--rank` / `--world-size`) — see
      §Measured for the first real 2-node Thunderbolt numbers. The MLX adapter and the
      head-to-head against MLX's ring are what remain.

### Not yet built

- **MLX adapter**, and head-to-head numbers against MLX's built-in ring. `mcclbench`
  is the harness that will produce them; what is missing is the shim that lets MLX
  hand mccl its arrays, and a cluster to run it on.
- **Thunderbolt-specific transport.** The `Transport` protocol is the seam: a bulk-DMA
  TB transport implements `listen`/`connect` and everything above it is unchanged.
  Today TB is reached as ordinary IP over the Thunderbolt bridge.
- **A real rendezvous service.** `Rendezvous` is one round trip through rank 0 —
  enough for `mcclCommInitRank`, not enough for a long-lived cluster. No discovery, no
  membership change, no recovery from a rank restarting, no reconnection.
- **Non-blocking C calls.** The shim runs every collective to completion. A genuine
  `mcclStream_t` execution context, with `mcclGroupStart`/`mcclGroupEnd` batching,
  waits on there being a device queue worth enqueueing onto.
- **Vectorised reduction and codec kernels.** `Kernels` and the codecs are scalar
  loops. This is no longer a theoretical cost: §Measured shows each codec's encode/
  decode ceiling (~0.50 GB/s for `downcast`, ~0.16 for `int8/256`) sitting *below*
  what a Thunderbolt cable delivers uncompressed, which is the whole reason the
  codecs lose there. Vectorising them is the change that would flip that verdict.
