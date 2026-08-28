# mccl vs. NCCL: a measured comparison

*2026-08-27. Local numbers measured with mcclbench/mcclprobe on two Mac Studio M1 Max
machines joined by a direct Thunderbolt cable (negotiated 20 Gb/s), plus an M1 Pro laptop
reachable from either only over Wi-Fi for the n = 3 runs; CUDA numbers cited.
Absolute cross-platform numbers are meaningless — the comparison is efficiency
fractions: what share of the respective interconnect's ceiling each stack extracts.*

## Published NCCL anchors

| setting | achieved (busbw) | ceiling | fraction |
|---|---|---|---|
| 8× A100, NVLink3, all-reduce | ~200 GB/s | 430–520 GB/s measured P2P | ~40–50% |
| 100 Gbps Ethernet, TCP sockets (no RDMA), 8–128 MB | 4–6 GB/s | 12.5 GB/s line rate | ~32–48% |
| TCP fallback vs. RDMA generally | — | — | 5–20× collapse |

Sources: [nccl-tests #149](https://github.com/NVIDIA/nccl-tests/issues/149),
[nccl #450](https://github.com/NVIDIA/nccl/issues/450),
[nccl #209](https://github.com/NVIDIA/nccl/issues/209).

## mccl measured (2-node Thunderbolt, fp32 sum all-reduce, ring, best of 5, scalar codecs)

Probe-measured achievable link **for this run**: 1.06 GB/s (42% of the 20 Gb/s line —
the IP-over-TB stack tax, paid before mccl runs). Per-pair path probing later read the
same cable at 1.46 GB/s (58% of the line) in the mixed-fabric world; the fractions in
the table below are against the probe taken alongside the sweep, which is the only
denominator its own numbers were produced against.

| size | busbw | of measured link | of raw line rate |
|---|---|---|---|
| 1 MiB | 0.577 GB/s | 54.4% | 23% |
| 4 MiB | 0.651 GB/s | 61.4% | 26% |
| 16 MiB | **0.746 GB/s** | **70.4%** | 30% |
| 64 MiB | 0.616 GB/s | 58.1% | 25% |

Two denominators, both honest. Against the raw line, mccl's 25–30% sits below
NCCL-over-TCP's 32–48%; the difference is the IP-over-TB tax that the probe itself
pays, and which mccl's RDMA transport is built to avoid — though it has never run
on hardware, so that remains a prediction rather than a column in this table (see
docs/RDMA.md). Against what the link demonstrably carried in the same session —
the fraction mccl's own scheduling controls — the collective extracts 70%.

## The compression comparison NCCL cannot enter

NCCL has no in-band lossy compression, so this is the one axis on which there is nothing
to compare against — and it is where mccl's most portable result lives. The result is not
a speedup; it is a rule:

> **A codec pays only when the fabric's uncompressed all-reduce rate is below that
> codec's own encode/decode ceiling.**

Both quantities are measurable on the machine in front of you — the fabric's rate with
`mcclbench`, the codec's ceiling with `mcclbench --codec-bench` — so the rule settles
whether to switch compression on, rather than deferring it to a benchmark of your own
workload. It is stated in terms that do not mention Thunderbolt, Apple silicon, or any
particular codec, and it is falsifiable in both directions: lift a codec's ceiling from
below the fabric's rate to above it and that codec must flip from a loss to a win; leave
a codec's ceiling below and it must keep losing however much faster it got.

mccl also has exact wire-byte accounting to hold the rule to (verified on interface
counters: downcast 2.00×, int8 3.94×, topK/0.01 49.9× fewer bytes).

The rule was stated from the scalar encoders, whose ceilings all sat below what the
cable delivers, so every codec lost on Thunderbolt and every codec won on Wi-Fi.
Vectorising the encoders was the experiment that tested the prediction. The tables below
are that experiment's evidence, not the claim. Ceilings are round-trip payload GB/s,
measured directly with `mcclbench --codec-bench` on the rank that gates the
collective:

| codec | ceiling, scalar | ceiling, vectorised | TB verdict, scalar | TB verdict, vectorised |
|---|---|---|---|---|
| downcast | 0.89 GB/s | **10.79 GB/s** | 0.79–1.04× (loses) | **1.07–1.53× (wins)** |
| int8/256 | 0.21 GB/s | **1.69 GB/s** | 0.51–0.80× (loses) | **1.27–1.71× (wins)** |
| topK/0.01 | 0.19 GB/s | 0.36 GB/s | 0.69–1.25× (loses) | 0.72–1.15× (still loses) |

**The prediction held.** The two codecs whose ceilings cleared the fabric's 0.53–0.66
GB/s uncompressed rate flipped to wins; top-k, whose ceiling rose 1.9× and stayed
*below* the fabric, stayed a loss — the same rule, confirmed from both sides in one
experiment. Past the crossover the ranking changes too: `int8` costs six times the
CPU of `downcast` and now beats it, because once both clear the fabric only the
compression ratio matters, and int8 saves 3.94× the bytes against downcast's 2.00×.

On the 0.05 GB/s Wi-Fi path every codec still wins, and now collects almost exactly
its wire reduction — downcast 1.99× of a theoretical 2.00×, int8 3.88× of 3.94×,
topK 5.5×.

Also worth recording as a correction: the scalar ceilings originally quoted here
(~0.50/0.28/0.16 GB/s) were read off a distributed sweep's plateau, not measured. The
scalar kernels on an *idle* M1 Max run at 3.67 GB/s for downcast; the low figures
belong to the shared node that gates the collective. The rule survived the
correction, the numbers attached to it did not.

## Against naive theory, at n=2 and n=3

At n=2 the ring beats the tree by up to 1.72× — same bytes, but the ring is full-duplex
while the tree serializes through the root. Read this narrowly. At n=2 the ring
degenerates to a single full-duplex point-to-point exchange, its best case, and the
result is near-definitional once the two schedules are written out. It corrects the
naive expectation that the two tie, and it shows the executor really does use the link
in both directions at once. It is **not** evidence that topology *selection* works,
because the plans that justify having a planner cannot pay at two ranks.

**At n=3 selection was tested, and the qualitative claim holds.** Three ranks (2× M1
Max + M1 Pro) over Wi-Fi, fp32 sum all-reduce, ring against tree at each size:

| size | winner | margin |
|---|---|---|
| 16 KiB | tree | 1.87× |
| 64 KiB | tree | 1.58× |
| 256 KiB | tree | 1.17× |
| 1 MiB | ring | 1.32× |
| 4 MiB | ring | 1.03× |
| 16 MiB | ring | 1.26× |

Tree wins where the collective is latency-bound, ring wins where it is bandwidth-bound,
and the tree's margin decays monotonically before it flips — the shape the model
predicts, not two endpoints that happen to differ.

**The constant is wrong by 4–13×.** The closed-form `S*`, fed the probed α and B for
this fabric, predicts ~75 KB; the measured crossover is between 256 KiB and 1 MiB. The
model assumes a single bandwidth `B`, and Wi-Fi's effective bandwidth moves twentyfold
across this sweep (0.001 → 0.021 GB/s bus). Right shape, wrong constant: the planner
picks correctly on either side of a crossover it locates in the wrong place.

## Island detection, on a fabric that has islands

The mixed-speed fabric island detection exists for is the third measured case: the same
three machines, but with the two Studios on a Thunderbolt cable and the laptop reachable
from either only over Wi-Fi. A 20.76× ratio inside one world, and the probe splits it
`[0,1] [2]` from its own measurements with nothing told to the planner.

| size | ring | tree | hierarchical | best |
|---|---|---|---|---|
| 256 KiB | 31.0 / 27.9 ms | 27.8 / 31.7 ms | **24.8 / 21.5 ms** | hierarchical |
| 1 MiB | 64.1 / 61.9 ms | 84.2 / 50.9 ms | **49.7 / 44.9 ms** | hierarchical |
| 4 MiB | 185.1 / 219.5 ms | 135.8 / 156.5 ms | **117.4 / 118.2 ms** | hierarchical |
| 16 MiB | 803.3 / 763.6 ms | 492.7 / 490.0 ms | **465.4 / 433.6 ms** | hierarchical |

Two independent runs; the hierarchical plan wins in both at every size from 256 KiB to
16 MiB, by **1.74× over the ring** at 16 MiB. Below 256 KiB the run-to-run spread equals
the between-algorithm spread and the rows measure Wi-Fi jitter rather than a plan. At
64 MiB, one run, the tree takes it back by 8%. NCCL has no comparable published result to
score against here, because NCCL's fabric is homogeneous and enumerable by construction;
the point of the row is that the planner reached the split without being told it.

The closed form also lands correctly on this fabric — 139.9 KiB predicted, 64–256 KiB
measured — where it was 4–13× low on uniform Wi-Fi. The difference is in the fabric: a
mixed fabric's bottleneck bandwidth really is one constant.

Still untested on hardware: two islands of two ranks, which needs n ≥ 4 and a fourth
machine the lab does not have, and anything at all about scale beyond n = 3.

Full tables and method notes: [ARCHITECTURE.md](ARCHITECTURE.md) §Measured: mixed
fabric, n = 3, [WHITEPAPER.md](WHITEPAPER.md) §6; the value thesis this comparison feeds is
[WHITEPAPER.md §3](WHITEPAPER.md#3-value-proposition-competing-with-the-cuda-cluster-stack).


## The comparison that cannot be run

The obvious head-to-head — mccl's ring against MLX's own ring, same workload, same cable
— is not currently buildable, and the obstacle is upstream rather than here.
**mlx-swift does not build MLX's distributed backends at all**: its `Package.swift`
excludes `mlx/distributed/{ring,mpi,nccl,jaccl}` with the comment "do not build
distributed support (yet)", and the Swift layer exposes no distributed API to call. This
is not a metallib problem or a lab-machine problem; there is nothing to link against.

Python MLX (0.29.3 on these nodes) *does* ship the ring backend, so a cross-language
measurement over the same Thunderbolt pair is possible in principle, and a benchmark
matching `mcclbench`'s definitions was written for it. It was not completed: MLX's
launcher (`python3 -m mlx.distributed_run --backend ring`) requires passwordless ssh
*between* the nodes, and these two have ssh trust only from the driving workstation, not
to each other. Establishing that trust is a change to the lab machines rather than a
measurement, so it was left alone.

Two caveats to record for whoever finishes it. Such a number compares two language
runtimes as much as two schedulers, so it belongs in the "sanity check on the fabric"
column rather than the verdict column. And a genuine same-language comparison needs
mlx-swift to expose `mlx::distributed`, or a build of mlx-swift with those sources put
back in — that, not lab access, is the real blocker.

## Against JACCL, Apple's own collective library

`ml-explore/mlx` now ships **JACCL** — "Jack and Angelos' Collective Communication
Library", the `jaccl` backend of MLX Distributed, co-developed with the RDMA-over-
Thunderbolt hardware support and linked directly from Apple's TN3205. It is
MIT-licensed and, contrary to a natural assumption, builds standalone from the MLX
tree with a C++ API: Apple's WWDC26 session 233 says it "can be built without MLX"
and "is not limited to machine learning".

**No numbers are offered here, because none would be honest.** mccl's RDMA
transport has never run on hardware, so there is no measured mccl-over-RDMA figure
to place beside anything. Apple has published no JACCL microbenchmark table
either; the MLX documentation's claim is a ratio ("communication latency an order
of magnitude lower than the ring backend") with no absolute figure, and the WWDC
figures are end-to-end MLX throughput on a 4× M3 Ultra cluster rather than
collective bandwidth. A table here would be two blanks and a rumour.

What can be compared is scope.

| | JACCL | mccl |
|---|---|---|
| Fabric | RDMA over TB5 only; no TCP data path | TCP anywhere, RDMA where available, mixed in one world |
| Hardware floor | Apple silicon with TB5, macOS 26.2+ | any Mac; RDMA is an optional accelerant |
| Interface | C++ | C ABI (`libmccl.dylib`), plus Swift |
| Topology | validated against a matrix supplied to it; discovery is a separate ssh helper | measured in-process by `mcclprobe`, planned per collective and message size |
| Heterogeneous fabrics | not addressed | per-pair path selection, ratio-gap islands, hierarchical plans |
| Wire compression | none | int8 / downcast / top-k, with the measured crossover rule above |
| Collectives | `all_sum`, `all_max`, `all_min`, `all_gather`, `send`, `recv`, `barrier` | plus `broadcast`, `reduce`, `reduce_scatter` |
| Algorithm choice | size-based (single-phase below a threshold, reduce-scatter + all-gather above) | solved from measured bandwidth and latency |
| RDMA validated on hardware | yes, by Apple | **no** |

**The row that matters most is the last one.** On a uniform TB5 mesh, JACCL is the
better choice and this document should not pretend otherwise. The rows above it
describe where mccl is not competing with JACCL at all: a C ABI for runtimes that
cannot link C++, and fabrics that are not a uniform TB5 mesh — which is where the
compression rule at the top of this document earns its keep, since a codec pays
only on links slow enough that RDMA was never an option.
