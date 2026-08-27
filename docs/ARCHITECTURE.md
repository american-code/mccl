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

The kernels behind these live in `CodecKernels` and are vector code for `.float32`,
the dtype the codecs exist for; other dtypes keep the scalar path. Vectorising was
not allowed to move a single byte on the wire, so the fast encoders reproduce the
scalar frames exactly — same fp16 rounding, same round-half-away-from-zero
quantisation, same treatment of NaN and ±infinity, same block scales. A rank running
one build interoperates with a rank running the other, which
`CompressionInteropTests` checks against a copy of the original scalar loops rather
than against itself.

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

## MLX adapter

`MCCLMLX` is a separate target and a separate product, and that separation is the design
decision, not an incidental one. It is the only thing in the package with an external
dependency (`mlx-swift`); `MCCL` must stay linkable by a runtime that has never heard of
MLX, so the arrow points one way and only one way. `MCCLMLX` depends on `MCCL`; nothing
in `MCCL` knows the adapter exists. The C shim is the model in every other respect too:
thin, no decisions, every rule stated once in `MCCL` and surfaced here as an error.

The surface is an extension on `Communicator` taking `MLXArray`s — `allReduce`,
`allGather`, `broadcast`, `reduce`, `reduceScatter` — plus `averageGradients`, which is
the thing a training loop actually wants. All of it is synchronous: MLX's own
`all_sum` is, and an MLX training loop is straight-line code, so the adapter parks on a
semaphore over the async Swift API exactly as `CABI.swift` does.

### The three things the adapter has to arrange

**Laziness.** An `MLXArray` is a graph node until something forces it; mccl reduces
bytes. Every entry point calls `eval()` first. This is also why a data-parallel step
cannot be one compiled MLX graph — the backward pass must be *finished* before the
network call starts, so the step is cut in two around the collective.

**Layout.** A strided or broadcast array has no single contiguous run of bytes to send.
`asData(access: .noCopyIfContiguous)` returns the backing when the layout allows and
materialises a contiguous copy when it does not. In practice the fallback almost never
fires: MLX's own `eval()` has already materialised a strided slice into a fresh
contiguous array by the time the adapter looks.

**Aliasing.** MLX arrays share backings freely, so reducing in place over a caller's
array would corrupt every array that happened to alias it. The adapter returns a new
array and pays a copy for it.

### Zero-copy: what is real and what is not

The property that matters is free. MLX allocates in unified memory, so the pointer
handed to mccl is the address the GPU reads — there is no host/device staging anywhere.
A CUDA equivalent would `cudaMemcpy` every payload to host memory and back before NCCL
could see it. That is a platform advantage, not a clever one, and it is the whole reason
this is an adapter rather than a copy layer.

Writing a result *into* an `MLXArray` is where it gets interesting, and the finding is
worth recording because it is not documented anywhere:

- mlx-swift's only public route to an array's bytes is `asData(access:)`, which returns
  a Foundation `Data`. `Cmlx` — where `mlx_array_data_uint8` lives — is not a public
  product of the package, so there is no supported mutable-pointer accessor at all.
- `Data(bytesNoCopy:count:deallocator:)`, which `.noCopy` builds, **is not no-copy for
  small buffers**. Foundation stores payloads of 14 bytes or fewer inline, copying them;
  `withUnsafeBytes` then returns the address of that temporary and writes through it are
  silently lost. Measured at the boundary: 14 bytes copies, 15 bytes aliases. A bias
  vector of three floats is 12 bytes, so this is not an edge case — it is the case that
  would have made small tensors reduce to a no-op with no error raised anywhere.

`MLXWorkingBuffer` therefore has two backings, chosen by payload size, both correct:
at or above 64 bytes mccl writes the result array's own storage (one copy in, none out);
below it, mccl writes a scratch buffer the adapter owns and `finish()` builds the array
from it (two copies). The threshold sits four times clear of Foundation's 14, and
`MLXCollectiveTests.testAllReduceAcrossTheAdoptionThreshold` sweeps element counts across
it, so it is an implementation detail rather than a behaviour — a Foundation change
surfaces as a failing test, not a wrong answer. `allGather` and `reduceScatter` copy
nothing at either size: mccl gives them separate send and receive buffers.

**Is the remaining copy worth removing?** No. A memcpy runs at ~100 GB/s here — the rate
`Kernels.scale` achieves — against a fabric that delivers 1.06 GB/s, so one copy costs
about 1% of the time that payload spends on the cable. Removing it would buy a rounding
error and cost the guarantee that passing an aliased array is safe.

### Gradient averaging is one collective, not N

`averageGradients` flattens a model's gradient tree into a single contiguous buffer,
all-reduces it once with `.avg`, and slices the result back into the original shapes.

The reason is latency, and the numbers are stark at MLP scale. One all-reduce over the
Thunderbolt cable costs at least one round trip — 167.8 µs measured — before any payload
moves. Ten separate all-reduces for a ten-tensor model spend 1.7 ms on latency alone,
more than the entire backward pass. Fusing gives one round trip and hands the ring a
message large enough to reach its bandwidth plateau. This is what `ncclGroupStart` exists
for, and mccl reaches the same place without a batching primitive because the adapter
knows the whole gradient set at once.

Two invariants make it safe. Parameter paths are **sorted** and dtype groups are taken in
a **fixed canonical order**, so the buffer layout is a function of the model's structure
alone — the one thing every rank is guaranteed to agree on. Getting this wrong would not
raise an error; the byte counts would still match and the ranks would average unrelated
numbers. And `.topK` keeps one residual per dtype group, so a single-dtype model — the
normal case — gets exactly one residual covering the whole flattened gradient and
accumulates it correctly across a run.

### The metallib, and why it is a build step

mlx-swift compiles MLX's C++ core under SwiftPM but does not build `mlx.metallib`; that
is Xcode's build system, which this project does not use and the lab nodes do not have.
Without it MLX links and starts and then *aborts* on the first GPU op, so it cannot be
caught and reported. `Tools/fetch-metallib.sh` takes the version-matched library out of
the `mlx-metal` pip wheel and installs it beside the running binary, where MLX's loader
(`dladdr` on one of its own symbols) looks. The technique is adapted from SwiftSci, which
hit this first.

Two consequences for the test suite. `MCCLMLXTests` is a **separate test target** so that
the core library's 239 tests never acquire an MLX dependency and keep running on a
machine with no metallib. And every test that touches MLX checks for the library first and skips
with the fix, because a fatal error cannot be turned into a test failure.


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

`--codec-bench` drops the ranks and the sockets entirely and measures the codec
kernels alone, in payload GB/s, through `CodecKernelProbe` — the same `WireCodec` and
top-k code a ring step runs, so the ceiling it reports is the one the collective
pays. That number is the machine half of the compression rule (§Measured), and it is
a `mcclbench` mode rather than a test because the machines whose ceilings matter are
lab nodes that run executables, not XCTest.

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

> **This section records two runs.** Everything down to *Wi-Fi contrast* is the
> original measurement of the **scalar** codecs. §*Measured: vectorised codecs* below
> re-runs the same sweep after the encoders were vectorised, against a same-session
> scalar control, and reverses the verdict on Thunderbolt for `downcast` and
> `int8/256`. The scalar tables are kept rather than replaced: the prediction that
> vectorising would flip the verdict was made from them, and a prediction is only
> worth something next to the data that produced it.

**Setup.** Two Mac Studio M1 Max (`studio-a`, `studio-b`) joined by a direct Thunderbolt
cable, negotiated at 20 Gb/s, probed by `mcclprobe measure` at **1.06 GB/s and
167.8 µs**. `mcclbench` release builds, rank 0 on `studio-a` bound to `169.254.152.222`,
rank 1 on `studio-b` bound to `169.254.23.203`, all-reduce fp32 sum, ring.

That the traffic really crossed the cable is checked rather than assumed: the token
was verified to advertise `169.254.152.222` before rank 1 was started, and over one
sweep `studio-a`'s `en4` byte counters moved **+1.11 GB in and +1.11 GB out** while `en0`
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

Measured directly from `studio-a`'s `en4` output counter over five 64 MiB collectives,
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

**What this does and does not establish.** It is a real correction to the naive
expectation that the two plans tie at n = 2, and it says the executor genuinely exploits
full duplex rather than merely claiming to. It says nothing about whether the *planner*
picks well. At n = 2 the ring is not really a ring: it degenerates to a single
full-duplex point-to-point exchange, which is the ring's best case by construction, and
the result is close to definitional once you look at the schedules. The claim it supports
is "the executor exploits full duplex", not "topology selection is validated" — a
distinction that mattered until the n = 3 run below, which is where selection actually
gets tested.

**Noise.** `studio-a` is a shared machine and was running an unrelated inference workload
(~40% of one core) throughout. Worst-case points ran up to 5× the best in the same
sweep, which is why every figure above is the best of several full sweeps rather than a
single run, and why medians are recorded alongside in the raw output. The verdict is not
sensitive to the choice: `none` beats every codec above 256 KiB on Thunderbolt in both
the best-of and the median-of statistic, and loses to every codec on Wi-Fi in both.

## Measured: vectorised codecs

The result of this section is a rule, not a speedup:

> **A codec pays only when the fabric's uncompressed all-reduce rate is below that
> codec's own encode/decode ceiling.**
>
> Both quantities are measurable on the machine in front of you — the fabric's rate with
> `mcclbench`, the codec's ceiling with `mcclbench --codec-bench` — so the rule decides
> the question rather than deferring it to a benchmark of your own workload.

The rule is what makes compression a decision instead of a guess, it is stated in terms
that hold for any fabric and any codec, and it is falsifiable: it predicts that lifting a
codec's ceiling from below the fabric's rate to above it must flip that codec from a loss
to a win, and that a codec whose ceiling stays below must keep losing. Everything below
is the experiment that tested that prediction and the tables it produced. The kernel
speedups are the *instrument*, not the finding.

Same two machines, same cable, same sweep, later the same day. The change under test
is `CodecKernels`: the encoders now run as vector code, byte-compatible with the
scalar frames they replace (`CompressionInteropTests` checks that both ways).

### What the scalar loops were actually doing

`mcclbench --codec-bench` was added first, because the ceilings above were *inferred*
from the plateau of a distributed sweep and had never been measured directly. It runs
the real `WireCodec` and top-k kernels over one buffer with no ranks and no sockets,
and reports payload bytes per second — `enc` for the sending half, `dec` for the
receiving half, and `r/t` for `1/(1/enc + 1/dec)`, which is the rate a rank sustains
when it is encoding what it sends and decoding what it receives. That last number is
the "ceiling" the compression rule is stated against; a ring all-reduce at n = 2
encodes exactly one payload and decodes exactly one per call, so the two are directly
comparable.

Round-trip ceiling, GB/s of payload, best point per codec:

| codec | M1 Pro scalar → vector | M1 Max (`studio-b`, idle) scalar → vector | M1 Max (`studio-a`, shared) scalar → vector |
|---|---|---|---|
| none (memcpy) | 33.63 → 32.21 | 35.37 → 30.68 | 7.46 → 11.91 |
| downcast | 3.55 → **36.89** (10.4×) | 3.67 → **31.80** (8.7×) | 0.89 → **10.79** (12.1×) |
| int8/256 | 0.57 → **5.96** (10.4×) | 0.61 → **6.41** (10.5×) | 0.21 → **1.69** (8.2×) |
| topk/0.01 | 0.76 → **1.47** (1.9×) | 0.93 → **1.62** (1.7×) | 0.19 → **0.36** (1.9×) |

Two things in that table were not expected.

**The scalar codecs were never as slow as ~0.50 GB/s.** On an idle M1 Max the scalar
`downcast` kernel already ran at 3.67 GB/s round-trip — five times the cable. The
0.50/0.28/0.16 figures above are what `studio-a` achieved, and `studio-a` runs EXO.app;
its scalar ceilings measure 0.89/0.19/0.21. Since the reported wall time is the
*slowest* rank's, the collective was gated by the loaded machine's codec rate, and
the plateau landed at ~56% of that machine's kernel ceiling — the rest of a
compressed call being socket, framing and reduction. So the rule was right and the
number attached to it was a property of one busy node, not of the chip.

**Vectorising is not mostly about intrinsics.** The largest single win was deleting a
runtime stride: `WireCodec.encode` walked its buffer with `i * width`, where `width`
comes from the dtype at run time, and `ElementIO` switched on the dtype once per
element, so nothing in the loop was a compile-time constant and the vectoriser did
nothing. With the fp32 stride known, the *plain* fp16 conversion loop auto-vectorises
to 50–80 GB/s and beats every hand-written SIMD spelling tried — `SIMD8<Float16>`
widening lowers to eight scalar converts (4.8 GB/s). Explicit SIMD earns its place
only in the int8 scan and top-k's fold, and there two standard-library conveniences
have to be avoided because neither vectorises: `clamped`/`pointwiseMin`/`pointwiseMax`
(2.2 GB/s against 19 GB/s for the same clamp written with `replace(with:where:)`) and
every float→integer SIMD conversion (0.2 GB/s), which is why the quantiser prepares
floats in SIMD and hands the narrowing to `vDSP_vfix8`.

Top-k gains the least, and for a structural reason: its cost is not arithmetic but
the traffic of selection. Folding the residual, copying the magnitudes, partitioning
them and collecting the winners is five-ish passes over the whole tensor whatever the
instruction set. Partitioning magnitudes instead of an index permutation (one cache
miss per comparison before) is most of the 1.9×.

### All-reduce over the Thunderbolt cable, vectorised vs scalar

Both binaries were run **alternately in the same session** — `mccl-scalar` is the
previous commit built from `git archive HEAD` — because `studio-a`'s load moved by more
than the effect under test while the sweeps ran (load average 4 → 29 → 15 over the
hour; worst/best within a size reached 10×). 32 scalar and 28 vectorised complete
sweeps. Bus bandwidth in GB/s, best of all sweeps, speed-up against `none` in
parentheses:

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 64 KiB | 0.135 | 0.143 (1.06×) | 0.144 (1.07×) | **0.229 (1.69×)** |
| 256 KiB | 0.316 | **0.405 (1.28×)** | 0.403 (1.28×) | 0.352 (1.12×) |
| 1 MiB | 0.460 | **0.570 (1.24×)** | 0.553 (1.20×) | 0.312 (0.68×) |
| 4 MiB | 0.618 | 0.788 (1.27×) | **0.807 (1.31×)** | 0.336 (0.54×) |
| 16 MiB | 0.662 | 0.625 (0.94×) | **0.834 (1.26×)** | 0.428 (0.65×) |
| 64 MiB | 0.534 | **0.802 (1.50×)** | 0.596 (1.12×) | 0.355 (0.67×) |

The honest statistic on a machine this noisy is not the best-of but the *paired*
one: each sweep measures `none` and every codec within a few seconds of itself, so
the ratio inside a sweep is immune to drift that the absolute column is not. Median
within-sweep speed-up against `none`, over every sweep:

| size | scalar downcast | vector downcast | scalar int8/256 | vector int8/256 | scalar topk | vector topk |
|---|---|---|---|---|---|---|
| 64 KiB | 1.01× | **1.17×** | 0.74× | **1.27×** | 1.25× | **1.92×** |
| 256 KiB | 0.86× | **1.26×** | 0.51× | **1.34×** | 0.76× | 1.11× |
| 1 MiB | 0.79× | **1.41×** | 0.55× | **1.40×** | 0.69× | 0.88× |
| 4 MiB | 1.01× | **1.07×** | 0.71× | **1.50×** | 1.06× | 0.72× |
| 16 MiB | 1.04× | **1.53×** | 0.80× | **1.70×** | 0.82× | 1.15× |
| 64 MiB | 0.97× | **1.52×** | 0.76× | **1.71×** | 0.79× | 1.04× |

### Verdict: the flip happened, for two codecs out of three

> **`downcast` and `int8/256` now win on Thunderbolt at every size measured.**
> `topk/0.01` does not, and the reason is the same rule seen from the other side.

The prediction being tested was: *if vectorising lifts `downcast`'s ceiling well above
the ~0.75 GB/s the cable delivers uncompressed, `downcast` should flip to a win at
bandwidth-bound sizes.* It did. On the rank that gates the collective, `downcast`'s
ceiling went from 0.89 GB/s — below the link — to 10.79 GB/s, fourteen times the
link, and the codec went from 0.79–1.04× to 1.07–1.53× against uncompressed.

`int8/256` flipped harder and now wins by more than `downcast` despite costing six
times as much CPU, which is exactly what the rule predicts once *both* codecs are
comfortably above the fabric: past that point the codec's rate stops mattering and
only its compression ratio does, and int8 puts 3.94× fewer bytes on the wire against
downcast's 2.00×. That is the regime the Wi-Fi table has always been in.

`topk/0.01` is the control that keeps the rule honest. Its ceiling on `studio-a` rose
from 0.19 to 0.36 GB/s — a real 1.9× and still *below* the 0.53–0.66 GB/s the cable
delivers uncompressed. The rule says it should lose, and it does (0.54–1.15× at
bandwidth-bound sizes), while continuing to win at 64 KiB where the run is
latency-bound and the 49.9× smaller message shortens the round trips. Three codecs,
one rule, and the two that cleared the fabric's rate flipped while the one that did
not stayed lost.

### The frames did not change

Re-measured at `studio-a`'s `en4` counters, five 64 MiB collectives per codec, with the
vectorised build:

| codec | bytes out | vs. uncompressed | scalar build measured |
|---|---|---|---|
| none | 336.3 MB | 1.00× | 336.1 MB |
| downcast | 168.1 MB | **2.00×** | 168.1 MB |
| int8/256 | 85.3 MB | **3.94×** | 85.3 MB |
| topk/0.01 | 6.7 MB | **49.9×** | 6.7 MB |

Identical to the byte, which is the wire-compatibility claim measured on the cable
rather than in a test. `CompressionInteropTests` makes the stronger version of it:
the vectorised encoders produce byte-for-byte the frames the scalar encoders
produced, including rounding ties and the treatment of NaN and ±infinity, and frames
from either encoder decode identically through either decoder.

### Wi-Fi re-check: the codecs now collect their full wire reduction

The same sweep over the Wi-Fi interfaces (`192.168.1.250` ↔ `192.168.1.238`), best of
three, vectorised:

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 1 MiB | 0.048 | 0.076 (1.57×) | 0.098 (2.02×) | **0.157 (3.25×)** |
| 16 MiB | 0.054 | 0.108 (**1.99×**) | 0.210 (**3.88×**) | **0.298 (5.51×)** |

Every codec still wins, and now wins by almost exactly its wire reduction: `downcast`
1.99× against a 2.00× ratio, `int8/256` 3.88× against 3.94×. The scalar build reached
1.91× and 2.33× at the same point — on a 0.05 GB/s path the scalar `downcast` was
already free, but scalar int8 was not, and vectorising is what turned its 3.94× of
saved bytes into 3.88× of saved time.

## Measured: n = 3, and the first test of the planner

Everything above is n = 2, where the ring degenerates to a point-to-point exchange and
the tree has nothing to be a tree about. The claim the library is built on — *pick the
algorithm from measured bandwidth and latency, and the crossover follows* — needs at
least three ranks before it says anything at all.

Three ranks, two Mac Studio M1 Max plus an M1 Pro, over Wi-Fi (the only fabric that
reaches all three), fp32 sum all-reduce, ring against tree at each size:

| size | ring | tree | winner |
|---|---|---|---|
| 16 KiB | 25.619 ms | **13.697 ms** | tree, 1.87× |
| 64 KiB | 28.982 ms | **18.334 ms** | tree, 1.58× |
| 256 KiB | 35.334 ms | **30.143 ms** | tree, 1.17× |
| 1 MiB | **76.165 ms** | 100.356 ms | ring, 1.32× |
| 4 MiB | **329.019 ms** | 338.961 ms | ring, 1.03× |
| 16 MiB | **1.078 s** | 1.357 s | ring, 1.26× |

**The qualitative claim holds, and this is the first evidence for it on hardware.** The
tree wins at small sizes, where the collective is latency-bound and `⌈log₂ n⌉` dependent
hops beat `2(n−1)`; the ring wins at large sizes, where it is bandwidth-bound and the
ring's `2(n−1)/n` bytes per link beat the tree's `2N` through the root. The crossing is
monotone — tree's margin decays 1.87× → 1.58× → 1.17× as the size grows, then flips —
which is the shape the model predicts, not merely a pair of endpoints that happen to
differ. The measured switch falls between 256 KiB and 1 MiB.

**The constant is wrong, and by a lot.** The closed-form crossover
`S* = α·B·(log₂n − (n−1)) / ((n−1)/n − log₂n)`, fed the probed α and B for this Wi-Fi
fabric, predicts ~75 KB. The measured crossover is somewhere in 256 KiB – 1 MiB, so the
formula is low by a factor of roughly 4–13. The likely reason is visible in the table
itself: the model assumes a single bandwidth `B`, and Wi-Fi's effective bandwidth is
strongly size-dependent — the bus figures climb from 0.001 GB/s at 16 KiB to 0.021 GB/s
at 16 MiB, a twentyfold change across the sweep. A crossover derived from one `B` cannot
be right at both ends of that. So: **right shape, wrong constant**, and the honest
summary is that the planner picks the correct algorithm on either side of a crossover it
locates in the wrong place.

**What is still unvalidated.** The hierarchical plan, which needs at least two islands of
two ranks and therefore n ≥ 4 — a uniform three-rank fabric has no islands, and the
sweep correctly produced no hierarchical rows for it. And this run is Wi-Fi, not
Thunderbolt: the interesting mixed-speed case, where island detection is supposed to
earn its place, is a four-machine fabric with both.

### Two operational notes from that run

**`mcclbench` used to drop inapplicable algorithms silently.** Asking a three-rank sweep
for `hierarchical` produced a table with no hierarchical rows and no explanation, which
reads as a broken harness rather than as a property of the world. Dropping the rows is
correct; saying nothing was not. It now prints

```
hierarchical: not applicable — needs at least two islands of two ranks; a 3-rank uniform fabric has none
```

before the table (`BenchAlgorithm.inapplicabilityReason(worldSize:)`).

**Detached ssh sessions break outbound dials on macOS 26.** This cost an hour of
diagnosis and will bite anyone orchestrating a multi-node run, so it is written up in
full at [USAGE.md § Launching across machines: keep the ssh session
attached](USAGE.md#launching-across-machines-keep-the-ssh-session-attached). The short
version: a `nohup`-detached mccl process whose ssh session has since exited gets
`No route to host` (EHOSTUNREACH) on every outbound LAN dial, while its *accepts* keep
working — so rank 0 always came up and every joining rank always failed. It is macOS's
local-network privacy control, which blocks orphaned non-platform binaries' unicast
dials. Keep the ssh session attached, or launch under launchd.


## Measured: reduction kernels

The codec work left a loose end. Its finding was not "SIMD is faster" but something more
specific and more portable: **a loop whose body switches on a runtime value cannot be
vectorised, and on this code that was worth an order of magnitude.** `WireCodec.encode`
walked its buffer with a stride read from the dtype and called `ElementIO.loadFloat`,
which switched on that dtype once per element. Neither the stride nor the element type
was a compile-time constant inside the loop.

`Kernels.reduce` had the same shape from the other direction. It was dtype-specialised on
the *outside* — a `switch dataType` selecting a typed loop — but every iteration called a
shared `combine(_:_:_:)` that switched on `op`. One runtime value in the loop body is
enough.

The fix is not SIMD; it is hoisting. Each `(dtype, op)` pair now gets a loop body that is
a single compile-time-known operation. fp32 is written as an explicit eight-lane body
because `min`/`max` need an exact NaN-asymmetric select that no hardware `fmin` provides
(`Swift.min(x, y)` is `y < x ? y : x`, which is not symmetric in NaN); the rest are plain
loops that the optimiser vectorises once the operation is fixed.

Measured on an M1 Pro, release build, 4 Mi elements, best of seven with the buffers
re-seeded before each timed pass. Rates count both streams read and the stream written:

| kernel | before | after | speedup |
|---|---|---|---|
| `reduce` fp32 sum | 16.88 GB/s | **75.54 GB/s** | 4.5× |
| `reduce` fp32 prod | 16.55 GB/s | **83.69 GB/s** | 5.1× |
| `reduce` fp32 min | 16.63 GB/s | **73.22 GB/s** | 4.4× |
| `reduce` fp32 max | 16.13 GB/s | **82.84 GB/s** | 5.1× |
| `reduce` fp16 sum | 7.36 GB/s | **93.54 GB/s** | 12.7× |
| `reduce` bf16 sum | 3.85 GB/s | **20.13 GB/s** | 5.2× |
| `reduce` int32 sum | 14.76 GB/s | **82.13 GB/s** | 5.6× |
| `reduce` int8 sum | 4.33 GB/s | **92.29 GB/s** | 21.3× |
| `scale` fp32 | 98.18 GB/s | 97.14 GB/s | 1.00× |

**`scale` was already fine, and that is the honest half of the result.** It never had the
per-element switch — the factor is hoisted to a local and each dtype has its own body —
so it was already auto-vectorising, and replacing it with hand-written SIMD changed
nothing measurable. It was reverted to the plain loop and the measurement written into
the source, so the next reader does not "optimise" it again. bf16 is the other honest
exception: its round-to-nearest-even store has a NaN branch in it, so it stays scalar and
gains only from the hoist.

**What this is worth in a collective.** Less than the codec numbers were, and the reason
is structural: reduction runs once per received chunk while a codec runs twice per call,
and `none` was never codec-bound. On the 1.06 GB/s Thunderbolt cable a 16.9 GB/s kernel
already spends only ~6% of the wall clock. The cases where it matters are the ones where
the fabric is not the bottleneck — in-process and loopback worlds, a future bulk-DMA
transport, and the reduction that top-k's gathered blocks feed.

Semantics did not move. `ReduceKernelTests` pins every `(dtype, op)` pair element-for-element
against an independently written reference, including the fp16/bf16 rounding, the integer
wrapping, `scale`'s integer saturation, and the NaN asymmetry of `min`/`max`. The
throughput test is release-only and skips in debug — an unoptimised build runs these
loops about a thousand times slower (1.15 GB/s against 75), so a debug run would measure
the absence of the optimiser rather than the kernel.


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
      the uncompressed result over repeated calls. The encoders are vectorised
      (`CodecKernels`) and byte-compatible with the scalar frames they replace, which
      is what makes them worth using on a Thunderbolt cable rather than only on Wi-Fi.
- [~] **5. C shim + MLX adapter; benchmark against MLX's built-in ring.**
      The C shim ships (`mccl.h`, `libmccl.dylib`, validated by a compiled C client) and
      `mcclbench` compares every algorithm against every codec across message sizes,
      now in-process *or* across machines (`--rank` / `--world-size`) — see
      §Measured for the real 2-node Thunderbolt numbers. **The MLX adapter ships**
      (`MCCLMLX`, §MLX adapter) with a working data-parallel training demo
      (`mccltrain`) whose 2-rank loss trajectory matches a single-process run on the
      combined batch to 4.8e-07. The head-to-head against MLX's *own* ring is the one
      part still open, and for a reason outside mccl: mlx-swift does not build MLX's
      distributed backends at all (its `Package.swift` excludes
      `mlx/distributed/{ring,mpi,nccl,jaccl}`) and exposes no distributed API, so a
      Swift-against-Swift comparison cannot be built from this package.

### Not yet built

- **A same-language head-to-head against MLX's built-in ring.** Blocked upstream, not
  here: mlx-swift ships no distributed support. Python MLX does, so a cross-language
  measurement over the same cable is possible and is recorded in COMPARISON.md, but it
  compares two runtimes rather than two schedulers.
- **A 3+ node run.** Everything measured so far is n = 2, where the ring degenerates to
  a single full-duplex exchange and neither the tree's `O(log n)` hops nor the
  hierarchical plan's single bridge crossing can pay. **Topology selection — the
  planner, which is the library's main claim — is therefore unvalidated on hardware.**
  This is the most valuable measurement still outstanding.
- **Thunderbolt-specific transport.** The `Transport` protocol is the seam: a bulk-DMA
  TB transport implements `listen`/`connect` and everything above it is unchanged.
  Today TB is reached as ordinary IP over the Thunderbolt bridge.
- **A real rendezvous service.** `Rendezvous` is one round trip through rank 0 —
  enough for `mcclCommInitRank`, not enough for a long-lived cluster. No discovery, no
  membership change, no recovery from a rank restarting, no reconnection.
- **Non-blocking C calls.** The shim runs every collective to completion. A genuine
  `mcclStream_t` execution context, with `mcclGroupStart`/`mcclGroupEnd` batching,
  waits on there being a device queue worth enqueueing onto.
- ~~**Vectorised reduction kernels.**~~ Done — see §Measured: reduction kernels. The
  same per-element-switch pattern was in `Kernels.reduce`, and the same fix applied:
  4.4–5.1× on fp32, up to 21× on int8. `Kernels.scale` needed nothing and got nothing.
- **Codec choice is still a caller's flag.** Now that two codecs win on Thunderbolt
  and one loses, `compression:` is a decision the planner has the data to make and
  the caller usually does not. The pieces exist: the probe measures the fabric, and
  `mcclbench --codec-bench` measures this machine's ceilings through the same
  `CodecKernelProbe` the collective runs.
