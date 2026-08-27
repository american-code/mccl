# mccl: a measured-topology collectives library for Apple Silicon clusters

## Abstract

Small clusters of Apple Silicon machines have become a plausible substrate for
distributed inference and fine-tuning, but they have no collectives library: MLX
ships a built-in ring all-reduce, and everything else is written per-application.
This paper describes mccl, an NCCL-shaped collectives library for such clusters,
and argues three positions. First, that on a Mac cluster the interconnect rather
than the arithmetic is the binding constraint, so the library's job is to move
fewer bytes rather than to reduce them faster. Second, that algorithm selection
should be *solved* from measured bandwidth and latency rather than tuned against
byte thresholds: we give the closed-form ring/tree crossover
`S* = α·B·(log₂n − (n−1)) / ((n−1)/n − log₂n)` and show it is the only size
threshold in the implementation. Third, that top-k sparsification is an algorithm
rather than a per-hop codec, because applying it per hop introduces a bias error
feedback cannot repair. We report a real-hardware evaluation on two Mac Studio
M1 Max machines over a direct Thunderbolt cable — 1.06 GB/s and 167.8 µs against
a 20 Gb/s negotiated link, and an all-reduce reaching 70.4% of that — together
with a measured rule for when in-band compression pays: *only when the fabric's
uncompressed all-reduce rate is below the codec's own encode/decode ceiling*.
That rule made a falsifiable prediction, which we then tested by vectorising the
encoders: two of the three codecs cleared the cable's rate and flipped from
losing 20–45% to winning 1.1–1.7×, while the third, whose ceiling stayed below
the cable, stayed a loss. Correctness is evaluated by 232 tests at 90.56% region
coverage, including byte-level interoperation between the vectorised and scalar
encoders.

---

## 1. Motivation

The hardware case for Mac clusters is straightforward. A Mac Studio's unified
memory is addressable by both CPU and GPU without a copy, and its capacity is
large relative to discrete accelerators at comparable cost, which matters for
inference on models that do not fit one device. Thunderbolt gives a handful of
them a direct link an order of magnitude faster than their Ethernet.

The software case is where it falls apart. On NVIDIA hardware an application does
not implement all-reduce; it links NCCL, which knows the topology, picks ring or
tree per message size, and exposes a stable C API every framework targets. On
Apple Silicon there is no equivalent layer. MLX ships a ring all-reduce inside
the framework; anything that is not MLX — a llama.cpp fork, a raw-Metal
application, a research prototype — writes its own or does not distribute.

The consequence is not merely duplicated effort. It is that nobody is optimising
the thing that actually limits these clusters. On a DGX box the fabric is NVLink
and the arithmetic is the interesting part; the library's job is to keep the
links saturated while the GPUs work. On a Mac cluster the ratio is inverted. Our
measurements below put a direct Thunderbolt link at 1.06 GB/s, while a single
M1 Max's memory bandwidth is roughly two orders of magnitude above that. Any
all-reduce big enough to matter is bounded by the wire, not the reduction kernel
— so the highest-leverage optimisation available is *sending fewer bytes*, and
that is precisely what an application-level ring cannot do on the application's
behalf.

mccl is an attempt at that missing layer: infrastructure, not a training
framework.

---

## 2. Related work

**NCCL** is the reference design and the source of mccl's API shape. Three of its
contributions matter here: the algorithm portfolio (ring for bandwidth-bound
messages, tree for latency-bound ones), the `ncclComm_t` / `ncclUniqueId`
bring-up idiom, and *bus bandwidth* as the reporting metric — payload throughput
scaled by the traffic each link actually carries, `2(n−1)/n` for all-reduce. mccl
adopts all three. Where it departs it is because a Mac cluster differs from a DGX
box: NCCL detects topology from NVLink and PCIe enumeration, which has no
analogue when every link is an IP interface, and NCCL has no in-flight
compression, because on an NVLink fabric the wire is not the bottleneck.

**MLX distributed** ships a ring all-reduce over Thunderbolt and Ethernet inside
the MLX framework, plus an MPI backend. It is what a Mac-cluster user reaches for
today, and it works. Its limitation is layering rather than quality: it is part
of a framework, so a non-MLX runtime cannot use it, and it is a fixed ring, so it
does not adapt to a measured fabric or a message size. mccl is positioned
underneath rather than against it — the intended endpoint is an adapter that lets
MLX hand mccl its arrays, and a head-to-head comparison. That adapter is not
built (§6).

**exo** approaches a related problem from the other end: running large models
across heterogeneous consumer devices by partitioning them according to each
device's memory. Its concern is placement and scheduling across an unequal device
pool, not collectives — a plausible consumer of a library like this rather than a
competitor.

Gradient compression has a substantial literature mccl draws on rather than
extends: error feedback for sparsified SGD is what makes top-k converge instead
of drift, and blockwise integer quantisation is standard. mccl's contribution is
not the schemes but the argument in §3.4 about *where in the collective* they may
be applied.

---

## 3. Design

### 3.1 A measured-topology planner

mccl plans over measurements, not assumptions. `mcclprobe serve` runs on each
node; `mcclprobe measure` drives real transfers and produces a `Topology` of
`Link` values. Bandwidth comes from large streaming transfers after a warm-up
chunk, so TCP's window is already open; latency is the *minimum* over many 8-byte
ping-pongs, because scheduling noise only ever adds time. A serving node will
probe a third node on request, so the map is a genuine pairwise mesh rather than
a star centred on the driving host.

The central decision is ring versus tree. With worst-case per-hop
latency α, bottleneck bandwidth B, message size S and n ranks:

```
t_ring = 2(n−1)·α + 2·(n−1)/n·S/B
t_tree = 2·log₂(n)·α + 2·log₂(n)·S/B
```

The ring pays `2(n−1)` latency-bound steps but each link carries only `(n−1)/n`
of the payload in each of its two phases, which is bandwidth-optimal. The tree
pays only `2·log₂(n)` steps but every level carries the whole message. Equating
and solving for S:

```
α(n−1−log₂n) = (S/B)(log₂n − (n−1)/n)

S* = α·B·(log₂n − (n−1)) / ((n−1)/n − log₂n)
```

Both factors are ≤ 0 for n ≥ 2, so `S* ≥ 0`. At n = 2 the numerator vanishes and
`S* = 0` — which is correct rather than degenerate, because a two-rank tree *is*
a ring. Below `S*` the planner picks tree; above it, ring.

Stating this in closed form removes the tuning knob.
`TopologyPlanner.crossoverBytes` is exactly this expression, and it is the only
size threshold anywhere in mccl's algorithm selection — there is no byte constant
to go stale when the fabric changes. On the Thunderbolt fabric of §5.1 it
evaluates to 0 bytes at n = 2, 139.0 KiB at n = 4 and 623.9 KiB at n = 16; on a
slower fabric it moves up, on a faster one down, and nothing is re-tuned.

Two smaller decisions fall out of the same measurements: the ring *order* is a
greedy maximum-bandwidth walk over the measured links rather than rank order, and
the tree *root* is the rank with the greatest total measured bandwidth.

### 3.2 Ratio-gap island detection

A mixed-speed fabric — Thunderbolt pairs bridged by Ethernet or Wi-Fi — wants
neither a flat ring nor a flat tree, since a flat ring routes the whole ring's
traffic across the slow bridge. The hierarchical plan rings inside each fast
island, rings between island leaders, and broadcasts back down a tree inside each
island, so the bridge carries one island-sum per island rather than everything.

Finding the islands is the interesting part, under a constraint we imposed: *no
link speed may be named in the code*. Testing bandwidth against "is this 10GbE or
TB5?" would encode today's hardware into a library meant to outlive it. Instead
mccl sorts the distinct measured bandwidths, finds the largest *ratio* gap
between adjacent values, places the cut at the geometric mean of the straddling
pair, and takes connected components of the edges above the cut — after gating on
the fastest link being at least 3× the slowest.

The 3× gate and the ratio-gap search do different jobs, and the distinction
produced a result we did not anticipate. In the four-node loopback map of §5.4
the overall spread is 3.74×, which passes the gate — but the largest adjacent
ratio gap is only 1.84×, so mccl reports a single uniform, noisy fabric rather
than inventing a hierarchy. That is the desired behaviour: overall spread is
necessary for a mixed fabric but not sufficient, and a genuinely bimodal fabric
shows up as a *gap*, not a range. On a map of two Thunderbolt pairs bridged by
Wi-Fi at the ratios measured in §5.1 (10.50×), island detection recovers
`[[0,1],[2,3]]` and the planner produces a hierarchical plan at 64 MiB and a
plain tree at 1 KiB.

The `Kind` label on a link (`thunderbolt5`, `usb4`, `ethernet`, `wifi`, …) is
deliberately cosmetic. It comes from the interface the peer is actually reached
on — `getifaddrs` for addresses and netmasks, SystemConfiguration for the
user-visible name, longest-prefix match to attribute a peer to a subnet — with
throughput inference as the fallback for routed peers. The planner never reads
it. What the interface buys is honesty in the operator-facing output: a congested
Thunderbolt bridge measuring 900 MB/s is still Thunderbolt, and fast Wi-Fi is
never a cable.

### 3.3 Per-hop wire compression

Compression is applied per hop and never changes the dtype the caller sees. The
codec id travels in the frame header, so a receiver always decodes with the
scheme the sender actually used and never has to be told out of band. Three
schemes are implemented:

| scheme | wire bytes for n fp32 elements | error bound |
|---|---|---|
| `downcast` | `2n` | relative ≤ 2⁻¹¹ |
| `int8Blockwise(b)` | `4⌈n/b⌉ + n` | absolute ≤ blockAbsmax/254 |
| `topK(f)` | `4 + k(4+4)` per rank, `k = ⌈fn⌉` | unbiased over repeated calls |

For a 64 MiB fp32 all-reduce over eight ranks, a dense ring puts 112.0 MiB across
each link; `downcast` halves that to 56.0 MiB, `int8/256` reduces it to 28.4 MiB,
and `topk/0.01` puts 10.2 MiB per rank on the wire.

A scheme that cannot help a given dtype degrades to raw rather than erroring:
`downcast` on an fp16 buffer is already done, and quantising an int32 accumulator
would be wrong rather than merely lossy. Compression is thus a hint the library
may decline, so a caller can set one policy across a mixed-dtype model.

### 3.4 Why top-k is an algorithm, not a codec

The most consequential position in mccl is that top-k sparsification is not
admissible as a per-hop wire codec and must instead replace the collective's
algorithm. The argument is about bias.

Top-k with error feedback works as follows. Each call adds the previous call's
residual to the caller's buffer, selects the `k` largest magnitudes of the sum,
sends those, and keeps the remainder as the new residual. Nothing is discarded,
only delayed: an element too small to be selected accumulates across calls until
it is large enough, so the sequence of compressed all-reduces converges on the
uncompressed answer. That argument depends on a specific invariant — *every unit
of mass not sent is retained by some rank's residual.*

Now consider applying top-k per hop inside a ring reduce-scatter. Each hop
receives a partially reduced chunk, adds its own contribution, and must
re-sparsify before forwarding. The mass dropped at that re-sparsification belongs
to a partial sum, not to any single rank's contribution. No rank's residual can
legitimately hold it: the sender did not contribute it and the receiver never saw
it. Whatever bookkeeping is chosen, some mass is dropped that no residual
accounts for, and the error is systematic rather than zero-mean — the same
elements are small on every hop of every call. Error feedback cannot repair a
bias it does not observe.

mccl therefore routes `.topK` to a different algorithm: sparsify once, at the
caller's buffer, then *all-gather* the fixed-size sparse blocks and sum them
locally. Given the sparsification, the reduction is then exact, which is the
property the convergence argument requires. The wire format is
`[nnz][indices][values]` — self-describing, so a receiver is never told `k` out
of band, and uniform across ranks, so an ordinary ring all-gather can move it.

Four restrictions follow, each enforced with a typed error rather than silently
accepted. **All-reduce only:** no other collective can account for the dropped
mass. **`.sum` and `.avg` only:** dropped elements are identities of a sum but
not of `min`, and a sparsified `min` would return the minimum of the *selection*.
**Floating-point dtypes only:** magnitude sparsification of an int32 accumulator
is not meaningful. **Residuals are `Float` whatever the caller's dtype:** an fp16
residual would flush the small values it exists to preserve back to zero.

Residuals are keyed by `(StreamID, elementCount)`, so two tensors reduced on one
communicator need two streams or they would share a history. Crucially,
`Communicator.topKResidual(for:)` exposes the residual, because it is part of the
model state: a checkpoint saved without it has silently discarded part of a
gradient.

---

## 4. Implementation

mccl is 3,982 lines of Swift in the core library, 796 in the C shim and its
hand-written header, and 926 in the benchmark harness and two CLIs, with **no
external package dependencies**. The transports are POSIX sockets directly.

**Transport stack.** The layering is `Collectives → Planner → Wire → Transport`,
and the seam that matters is the `Transport` protocol: `listen` returning a
`Listener`, `connect` returning a `Channel`, both blocking. `TCPTransport` and an
in-process `LoopbackTransport` implement it today; a Thunderbolt bulk-DMA
transport would slot in with no change above it. `MeshFabric` builds a full
rank-addressed mesh rather than only ring neighbours, because tree and
hierarchical plans need non-adjacent edges and the plan is chosen per operation,
after the fabric exists. Bring-up is deadlock-free by an ordering rule — for each
pair (i, j) with i < j the higher rank dials and the lower accepts — which
requires every listener to be bound before any rank bootstraps.

Blocking I/O is kept off the Swift concurrency cooperative pool: each
communicator runs its collectives on its own serial queue, and each channel has
distinct send and receive queues so a rank can send and receive at once. That
full-duplex overlap is where a ring's bandwidth comes from.

**C ABI.** `mccl.h` is hand-written rather than generated, because it is the
published contract. Three decisions are worth recording.

*Blocking semantics.* NCCL enqueues onto a CUDA stream and returns. mccl v0 has
no device queue to enqueue onto, so every call runs the collective to completion
(a dispatch semaphore over the async Swift API) and returns when the buffers are
safe to touch. This is the honest v0 behaviour rather than fake asynchrony, and
it is what makes the raw-pointer bridge safe by construction: the C caller cannot
free memory underneath a collective that has already returned.

*`mcclStream_t` is not an execution context.* It names an independent *sequence*
of collectives, which matters only to top-k's per-stream residual.

*The by-value struct ABI problem.* NCCL's `ncclCommInitRank` takes a 128-byte
`ncclUniqueId` **by value**, and a 128-byte struct passed by value has different
calling conventions on arm64 and x86_64 — so an exported symbol with that
signature is not portable across architectures a cluster may mix. mccl exports
`mcclCommInitRankFromId`, which takes the token by pointer, and the header
restores the idiomatic call site as a `static inline` wrapper. Callers write
NCCL's spelling; the ABI stays portable. `mcclResult_t` maps one-to-one onto
`MCCLError` — the enum was written flat for exactly this — and `mcclDataType_t`
is `DataType.wireCode`, the discriminator already in every frame header, so a C
caller and a Swift caller cannot disagree about what is in a frame.

---

## 5. Evaluation

### 5.1 First real-hardware validation

The first measurement of mccl on real cluster hardware: two Mac Studio M1 Max
machines connected by a direct Thunderbolt cable, negotiated at 20 Gb/s, probed
with `mcclprobe measure`.

| path | measured bandwidth | measured RTT | classified as |
|---|---|---|---|
| direct Thunderbolt cable | 1.06 GB/s | 167.8 µs | `usb4` |
| Wi-Fi | 101 MB/s | 2.62 ms | `wifi` |

Three things this establishes, and one it does not.

**The probe produces sane numbers on a real link.** 1.06 GB/s against a 20 Gb/s
(2.5 GB/s) negotiated line rate is 42.4%, squarely inside the 32–48% commonly
reported for NCCL over TCP sockets — the right comparison, since mccl reaches
Thunderbolt as IP over the TB bridge and pays the same stack costs. That is
evidence the methodology is sound and the framing layer has no gross
inefficiency, and simultaneously the strongest available argument for the
bulk-DMA transport of §6: roughly 58% of the link is lost to a protocol stack a
direct transport would not traverse.

**Interface-based link classification works.** The two paths between the same
pair of machines were attributed to their respective interfaces and labelled
`usb4` and `wifi` rather than both being inferred from throughput. The
*generation* label is necessarily a throughput inference — TB4 and TB5 are both
ordinary IP links to every API on the machine — but separating a cable from Wi-Fi
comes from interface media, exactly the case where throughput inference is least
trustworthy.

**The measured spread is what the mixed-fabric machinery is for.** The two paths
differ by 10.5× in bandwidth and 15.6× in latency: far past the 3× gate, and a
real instance of the bimodal fabric of §3.2 rather than a synthetic one.

**What it does not establish:** ring versus tree versus hierarchical, which needs
four or more machines. Two machines exercise the probe, the classifier and the
planner's degenerate case.

Collective throughput, which this section originally could not establish, is
measured in §5.1a.

### 5.1a Collective throughput on the Thunderbolt cable

`mcclbench` now runs distributed — one process per machine, joining through the
same `Rendezvous` token the C shim uses — so the same sweep that ran over
loopback runs over the cable of §5.1. Rank 0 on `lab-01` bound to
`169.254.152.222`, rank 1 on `lab-02` bound to `169.254.23.203`, all-reduce fp32
sum, ring, best of five sweeps. Bus bandwidth in GB/s; with n = 2 the bus factor
is 1, and the parenthesised figure is the fraction of the 1.06 GB/s probed link.

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 64 KiB | 0.122 (11.5%) | 0.129 (12.2%) | 0.089 (8.4%) | **0.158 (14.9%)** |
| 256 KiB | **0.299 (28.2%)** | 0.298 (28.1%) | 0.138 (13.0%) | 0.259 (24.4%) |
| 1 MiB | **0.577 (54.4%)** | 0.492 (46.4%) | 0.158 (14.9%) | 0.278 (26.2%) |
| 4 MiB | **0.651 (61.4%)** | 0.504 (47.5%) | 0.162 (15.3%) | 0.285 (26.9%) |
| 16 MiB | **0.746 (70.4%)** | 0.505 (47.6%) | 0.190 (17.9%) | 0.271 (25.6%) |
| 64 MiB | **0.616 (58.1%)** | 0.462 (43.6%) | 0.161 (15.2%) | 0.274 (25.8%) |

Uncompressed all-reduce reaches 70.4% of the probed link at 16 MiB. The probe
measures a one-directional stream; an all-reduce reduces every byte it receives
and turns the link around twice, so 70% is evidence that neither the framing nor
the ring schedule is where the time goes.

**The codec verdict, which §5.3 left open, goes against the codecs.** §5.3 called
it "an arithmetic question only the cluster can settle". The cluster has settled
it: on a 1.06 GB/s Thunderbolt link every codec loses above 256 KiB. `downcast`
puts exactly half as many bytes on the wire — verified at `lab-01`'s `en4` byte
counter, 168.1 MB against 336.1 MB over five 64 MiB collectives, with `int8/256`
at 3.94× and `topk/0.01` at 49.9× reduction — and is still 25–32% slower.

This is not the direction §5.3's loopback table pointed. There `downcast` cost
nothing measurable (1.209 against 1.195 GB/s bus at 16 MiB, inside the noise),
which made halving the wire bytes look like a free win waiting for a slow enough
link. On the cable it is not free at all.

The compressed columns explain themselves: they are flat in message size, because
what they measure is not the cable but the encoder. `downcast` plateaus at ~0.50
GB/s of payload, `topk/0.01` at ~0.28, `int8/256` at ~0.16. Those are this M1
Max's scalar encode/decode ceilings, and all three sit below what the cable
delivers uncompressed. The rule this yields is sharper than the prediction it
replaces:

> A codec pays only when the fabric's *uncompressed* all-reduce rate is below
> that codec's own encode/decode ceiling.

The same sweep over the two machines' Wi-Fi interfaces confirms it from the other
side. At 0.052 GB/s uncompressed — below every ceiling — every codec wins, and
`downcast` wins by 1.89×, almost exactly the 2.00× its wire reduction predicts,
because at that rate the encode is entirely hidden behind the transfer:

| size | none | downcast | int8/256 | topk/0.01 |
|---|---|---|---|---|
| 1 MiB | 0.044 | 0.071 (1.63×) | 0.075 (1.71×) | **0.132 (3.01×)** |
| 16 MiB | 0.054 | 0.103 (1.91×) | 0.126 (2.33×) | **0.295 (5.44×)** |
| 64 MiB | 0.052 | 0.099 (1.89×) | 0.110 (2.11×) | **0.256 (4.90×)** |

Two consequences. First, §3.3 treats compression as a caller's flag; it should be
a planned decision. The planner already measures link bandwidth, and a codec's
ceiling is a local-machine property that could be calibrated once the way the
probe calibrates a cable. Second, the ceilings are an artefact of scalar Swift,
not of the schemes — `downcast` at 0.50 GB/s is ~3 ns per element for what should
be a handful of vector instructions. Vectorising the codecs is the single change
that would flip this verdict on Thunderbolt, and this table is the argument for
making it.

§5.1b takes that prediction and runs it.

One expectation also failed. With n = 2 the ring and the binomial tree move the
same total bytes, and were expected to be equivalent; they are not. Over four
paired runs the ring beat the tree by 1.16× at 1 MiB and 1.72× at 16 MiB
(27.9 ms against 47.9 ms), because the ring's reduce-scatter and all-gather move
`N/2` in both directions at once while the tree serialises `N` up to the root and
`N` back down. The planner prefers the ring for large messages here; this says it
is right to even in the degenerate case.

*Method.* `lab-01` was shared with an unrelated workload throughout, and
worst-case points reached 5× the best within a sweep, so every figure is the best
of several complete sweeps. The verdict does not depend on that choice: `none`
beats every codec above 256 KiB on Thunderbolt, and loses to every codec on
Wi-Fi, under both the best-of and the median-of statistic. The wall time reported
per point is the slowest rank's, agreed with a one-element max all-reduce; a
barrier separates each point's warm-up from its timed region. Full tables in
[ARCHITECTURE.md § Measured: 2-node Thunderbolt](ARCHITECTURE.md#measured-2-node-thunderbolt).

### 5.1b Vectorising the codecs, and testing the prediction it was made to test

§5.1a ends with a falsifiable claim: the codecs lose on Thunderbolt because their
scalar encode/decode ceilings sit below what the cable delivers, so lifting a
codec's ceiling above that rate should flip it to a win. This section reports the
experiment. The change under test is `CodecKernels`, a vector implementation of
the same wire formats — byte-for-byte the same frames, checked in both directions
against a copy of the original scalar loops (§5.2).

**Measuring the ceiling instead of inferring it.** The ceilings in §5.1a were read
off the plateau of a distributed sweep, which conflates the codec with everything
else in a compressed call. `mcclbench --codec-bench` now measures the kernels
directly, in payload GB/s, through the same code path a ring step runs. The first
result is a correction to §5.1a: the scalar `downcast` kernel on an *idle* M1 Max
runs at 3.67 GB/s round-trip, five times the cable, not 0.50. The low figures
belong to `lab-01`, which shares its machine with an inference workload and, as
the slower rank, sets the reported time. Its scalar ceilings are 0.89 (downcast),
0.21 (int8/256), 0.19 (topk/0.01) GB/s — and the compressed plateaus of §5.1a
land at ~56% of those, the remainder being sockets, framing and reduction. The
rule was right; the constants attached to it were a property of one busy node.

**Round-trip ceilings, scalar → vectorised**, best point per codec:

| codec | M1 Pro | M1 Max, idle | M1 Max, shared (the rank that gates the run) |
|---|---|---|---|
| downcast | 3.55 → 36.89 | 3.67 → 31.80 | 0.89 → **10.79** |
| int8/256 | 0.57 → 5.96 | 0.61 → 6.41 | 0.21 → **1.69** |
| topk/0.01 | 0.76 → 1.47 | 0.93 → 1.62 | 0.19 → **0.36** |

**The verdict flips for two codecs out of three.** 32 scalar and 28 vectorised
sweeps, run alternately in one session because the shared machine's load moved by
more than the effect under test. The paired statistic — each codec against `none`
*within* the same sweep, median over sweeps — is the one that survives that drift:

| size | downcast scalar → vector | int8/256 scalar → vector | topk/0.01 scalar → vector |
|---|---|---|---|
| 64 KiB | 1.01× → **1.17×** | 0.74× → **1.27×** | 1.25× → **1.92×** |
| 1 MiB | 0.79× → **1.41×** | 0.55× → **1.40×** | 0.69× → 0.88× |
| 16 MiB | 1.04× → **1.53×** | 0.80× → **1.70×** | 0.82× → 1.15× |
| 64 MiB | 0.97× → **1.52×** | 0.76× → **1.71×** | 0.79× → 1.04× |

`downcast` and `int8/256`, whose ceilings cleared the 0.53–0.66 GB/s the cable
delivers uncompressed, now win at every size. `topk/0.01`, whose ceiling improved
by a genuine 1.9× and remains *below* that rate, still loses at bandwidth-bound
sizes while still winning at 64 KiB where the run is latency-bound. One experiment,
the rule confirmed from both sides.

A second prediction falls out and is also visible in the table: past the crossover
a codec's rate stops mattering and only its ratio does. `int8/256` costs six times
the CPU of `downcast` and beats it, because it puts 3.94× fewer bytes on the wire
against downcast's 2.00×. The Wi-Fi sweep shows the same regime from further
along — with vectorised codecs the speed-ups converge on the wire reductions
themselves: downcast 1.99× of a possible 2.00×, int8 3.88× of 3.94×, topk 5.5×.

**What made the codecs slow was not the absence of intrinsics.** The dominant cost
was a runtime stride: `WireCodec.encode` indexed with `i * width`, where `width`
comes from the dtype at run time, and read every element through a per-element
dtype switch, so no loop had a compile-time-constant stride or element type to
vectorise. With the fp32 stride fixed, the *plain* fp16 conversion loop
auto-vectorises to 50–80 GB/s and beats every hand-written SIMD spelling tried;
`SIMD8<Float>(SIMD8<Float16>)` in fact lowers to eight scalar converts (4.8 GB/s).
Explicit SIMD earns its keep only in the int8 scan and top-k's residual fold, and
there two standard-library conveniences must be avoided because neither vectorises
today — `clamped`/`pointwiseMin`/`pointwiseMax` (2.2 GB/s against 19 GB/s for the
same clamp via `replace(with:where:)`) and every float→integer SIMD conversion
(0.2 GB/s), which is why the quantiser prepares floats in SIMD and hands the
narrowing to `vDSP_vfix8`. Top-k gains least because its cost is selection
traffic, not arithmetic: five passes over the tensor whatever the instruction set.

**The wire is unchanged**, measured rather than asserted: five 64 MiB collectives
per codec moved 336.3 / 168.1 / 85.3 / 6.7 MB across `lab-01`'s `en4` counters
under the vectorised build, against 336.1 / 168.1 / 85.3 / 6.7 MB under the
scalar one.

### 5.2 Correctness

The test suite is 232 tests across 17 suites, all passing, run with
`swift test`:

| suite | tests | what it covers |
|---|---|---|
| CollectiveTests | 25 | all five collectives × dtypes × ops × plans |
| BenchmarkTests | 20 | the benchmark harness and its argument parsing |
| PlannerTests | 20 | crossover, island detection, ring order, trees |
| DistributedBenchmarkTests | 20 | distributed bring-up, cross-rank agreement, and real spawned `mcclbench` worlds |
| TopKTests | 17 | selection, block layout, residuals, convergence |
| TCPTransportTests | 15 | socket error paths: bind, resolve, hang-up |
| CShimTests | 13 | the C ABI, incl. a compiled four-rank C client |
| CodecBenchmarkTests | 10 | the codec-throughput mode and the ceilings it derives |
| CompressionTests | 12 | codec round-trips within their error bounds |
| CompressionInteropTests | 12 | vectorised frames against the scalar reference, both directions |
| FabricTests | 12 | mesh bring-up failures, plan sanitising |
| RendezvousProtocolTests | 11 | forged and malformed rendezvous frames |
| RendezvousTests | 11 | token encoding, address exchange, `join` |
| DiagnosticsTests | 9 | error text, plan text, analysis accessors |
| NetworkInterfaceTests | 9 | interface enumeration and media classification |
| ProbeTests | 8 | probe protocol and topology construction |
| TransportTests | 8 | framing, loopback and TCP round trips |

`DistributedBenchmarkTests` spawns real `mcclbench` processes with `Process` and
asserts on their output — two ranks and four ranks over loopback, checking that
exactly one table appears (on rank 0), that its bus figures carry the right
`2(n−1)/n` factor for the world size, and that a rank launched with a mismatched
sweep fails rather than deadlocking. The distributed path cannot be exercised
in-process without becoming the in-process path.

The load-bearing test is `CShimTests.testCProgramLinksAndPasses`, which compiles
a four-rank C client with `cc` against the built `libmccl.dylib` and runs it —
the only check that simultaneously proves `mccl.h` is valid C, that its
declarations match the symbols the dylib exports, and that a non-Swift process
can link and run the library. A Swift test calling `@_cdecl` functions directly
would prove none of the three.

Coverage, regenerated with `swift test --enable-code-coverage` and `llvm-cov`:

| metric | covered |
|---|---|
| region | **90.56%** (224 of 2373 regions missed) |
| function | **94.43%** (35 of 628 missed) |
| line | **97.05%** (141 of 4783 missed) |

The lowest files are `Collectives.swift` (87.07% region) and `DataType.swift`
(86.15%), where the residue is dtype × op × plan combinations covered
behaviourally through other paths. What remains uncovered is dominated by
defensive code unreachable without injecting faults into libc — `EINTR` retry
loops, `socket()` returning a negative descriptor — which we did not chase.

### 5.3 Loopback benchmarks, and what they measure

`mcclbench` sweeps message size × algorithm × codec over N ranks in one process,
bringing one world up for the whole sweep and pinning the plan per point, so the
table compares algorithms rather than bootstrap costs. It reports algorithmic
bandwidth (payload over wall time) and bus bandwidth (scaled by `2(n−1)/n` for
all-reduce), the latter comparable to NCCL's `busbw`.

Measured on the M1 Pro development machine (16 GB, macOS 26.5.1, Swift 6.2.3),
four ranks over loopback TCP, fp32 sum — selected rows from the full sweep:

```
        size  algorithm     codec           iters          wall    GB/s alg    GB/s bus
       1 KiB  ring          none               50      548.5 µs       0.002       0.003
       1 KiB  tree          none               50      105.4 µs       0.010       0.015
      64 KiB  ring          none               50      554.7 µs       0.118       0.177
      64 KiB  tree          none               50      250.9 µs       0.261       0.392
       1 MiB  ring          none               50      1.310 ms       0.800       1.201
      16 MiB  ring          none                8     21.060 ms       0.797       1.195
      16 MiB  ring          downcast            8     20.809 ms       0.806       1.209
      16 MiB  ring          int8/256            8     51.285 ms       0.327       0.491
      16 MiB  hierarchical  topk/0.01           8     24.967 ms       0.672       1.008
      16 MiB  tree          none                8     72.826 ms       0.230       0.346
```

The algorithm columns behave as the model predicts: tree wins decisively at 1 KiB
(105 µs against the ring's 549 µs) and loses decisively at 16 MiB (72.8 ms
against 21.1 ms). That is the crossover doing its job, on a fabric where the
"hops" are memcpys and the latency term is therefore much smaller than it would
be across four machines.

**The codec columns are a cost measurement, not a verdict.** Over loopback there
is no wire: the "interconnect" is memory bandwidth, an order of magnitude faster
than the Thunderbolt link of §5.1. Every codec pays its full encode cost and buys
back nothing, so `int8/256` duly comes last at 0.491 GB/s bus against `none` at
1.195. What the table gives is each codec's CPU price per byte, the input needed
to predict where a codec starts *winning* on a slower link. `downcast` halves the
bytes at a cost already invisible here (1.209 vs 1.195 GB/s bus at 16 MiB, within
run-to-run noise); `int8/256` quarters them at roughly 2.4× the CPU time. Whether
those trades win on a 1.06 GB/s link is an arithmetic question only the cluster
can settle, and settling it is what `mcclbench` exists for.

One anomaly should be recorded rather than smoothed over. At 64 MiB the ring
without a codec measured 391 ms, an order of magnitude worse than its 16 MiB
point, over only three iterations. Four ranks each holding 64 MiB payloads plus
scratch exceeds what a 16 GB machine keeps resident, so that is memory pressure
on the test host rather than a property of the collective; the 64 MiB row should
not be used for anything.

### 5.4 Probe measurements over loopback

Three `mcclprobe serve` processes on one machine, measured pairwise, produce a
genuinely asymmetric four-node map (1.32–4.93 GB/s, RTTs 12.8–69.4 µs) from which
the planner derives a 71.5 KiB crossover, a ring order of `0→1→3→2` rather than
rank order, and a tree rooted at rank 3 rather than rank 0. The 3.74× spread with
a 1.84× maximum ratio gap is the "noisy single fabric" case of §3.2. These
numbers say nothing about real link speeds; what they establish is that pairwise
measurement, JSON persistence and plan derivation work end to end on data the
library did not generate synthetically.

---

## 6. Limitations and future work

**Evaluation is two machines, not a cluster.** §5.1a measures real collective
throughput over a real cable, which settles the codec question and puts a number
on ring versus tree at n = 2. It does not validate the planner's central claim:
with two ranks there are no islands, the crossover is 0 bytes, and the
hierarchical plan does not exist. Four or more machines on a mixed
Thunderbolt/Ethernet fabric remains the experiment that would, and `mcclbench`
in distributed mode is now the harness ready to run it.

**No MLX adapter, and therefore no head-to-head numbers.** The comparison that
would establish whether mccl is worth using — mccl against MLX's ring, same
workload, same cluster — needs a shim letting MLX hand mccl its arrays.
`mcclbench` is the harness; the adapter and the cluster are what is missing.

**No Thunderbolt-specific transport.** Thunderbolt is reached as ordinary IP over
the TB bridge, and the 42.4% of line rate in §5.1 is the direct cost. A bulk-DMA
transport implementing `listen`/`connect` would leave everything above it
unchanged, and is the largest identified performance opportunity in the system.

**The rendezvous is not a service.** One round trip through rank 0 — no
discovery, no membership change, no recovery from a rank restarting. Discovery
(mDNS, with interface enumeration already in place) and a durable coordinator are
additive; membership change and reconnection would reach into the collectives.

**The C calls all block, and the reduction kernels are still scalar loops.** A
genuine `mcclStream_t` execution context with `mcclGroupStart`/`mcclGroupEnd`
batching waits on there being a device queue worth enqueueing onto. The *codecs*
are no longer scalar (§5.1b), and that turned out to matter more than expected:
`Kernels.reduce` and `Kernels.scale` carry the same runtime stride and
per-element dtype switch that cost the encoders an order of magnitude, and the
same fix applies. Reduction runs once per received chunk against the codec's
twice per call, so the ceiling it caps is `none`'s rather than a codec's.

**Compression is still a caller's flag, and should not be.** §5.1b leaves the
library in a state where two codecs win on Thunderbolt and one loses, which is a
decision the planner has the data to make and the caller usually does not: the
probe measures the fabric, `mcclbench --codec-bench` measures the machine, and
the rule connecting them is one comparison. Nothing in the API expresses it yet.

**An open research question.** Because mccl knows the exact partition boundaries
and the compression error introduced at each hop, it can emit ground truth for a
question we cannot answer here: do lossy collectives change interpretability
guarantees for distributed inference? If activations are reduced across machines
with per-hop quantisation, the quantities an interpretability method reads are
not the quantities the unsharded model would have produced, and the divergence is
bounded by numbers mccl already computes. Whether that bound is tight enough to
preserve any particular guarantee is unknown; the infrastructure to investigate
it is a side effect of building the library rather than a goal of it.

---

## 7. Conclusion

Mac clusters lack a collectives library, and the absence matters more than it
would elsewhere, because the thing that limits them — the interconnect — is
exactly the thing an application-level ring cannot optimise on the application's
behalf. mccl is an attempt at that layer, built around three positions: measure
the fabric rather than assume it, solve the algorithm crossover in closed form
rather than tune a threshold, and refuse to apply sparsification anywhere the
error-feedback argument does not hold.

The implementation is complete enough to evaluate: all five collectives across
five dtypes and five reductions, three executable plans, three wire codecs, a
probe, a planner, a C ABI validated by a compiled C client, and a benchmark
harness — 232 tests at 90.56% region coverage, no external dependencies. The
first real-hardware measurement puts a direct Thunderbolt link at 1.06 GB/s and
167.8 µs, 42.4% of its negotiated line rate, with both paths between the machines
correctly classified. On that link the compression rule of §5.1a has now been
tested rather than merely stated: vectorising the encoders lifted two codecs'
ceilings above the fabric's rate and both flipped from losing to winning, while
the third stayed below it and stayed a loss.

What is not established is the claim the design rests on: that a measured-topology
planner with wire compression beats a fixed ring on a real mixed-speed Mac
cluster. That needs four or more machines, an MLX adapter, and the head-to-head
numbers `mcclbench` was built to produce. Everything required to run that
experiment is in place except the cluster.

---

*Measurements on the development machine were taken on an Apple M1 Pro (16 GB
unified memory, macOS 26.5.1, Swift 6.2.3). The two-machine Thunderbolt figures
were measured on a pair of Mac Studio M1 Max machines. Coverage figures are
reproducible with `swift test --enable-code-coverage` followed by `xcrun llvm-cov
report`; see [USAGE.md](USAGE.md) for the tool invocations and
[ARCHITECTURE.md](ARCHITECTURE.md) for the component-level design.*
