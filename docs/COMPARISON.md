# mccl vs. NCCL: a measured comparison

*2026-08-26. Local numbers measured with mcclbench/mcclprobe on two Mac Studio M1 Max
machines joined by a direct Thunderbolt cable (negotiated 20 Gb/s); CUDA numbers cited.
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

Probe-measured achievable link: 1.06 GB/s (42% of the 20 Gb/s line — the IP-over-TB
stack tax, paid before mccl runs).

| size | busbw | of measured link | of raw line rate |
|---|---|---|---|
| 1 MiB | 0.577 GB/s | 54.4% | 23% |
| 4 MiB | 0.651 GB/s | 61.4% | 26% |
| 16 MiB | **0.746 GB/s** | **70.4%** | 30% |
| 64 MiB | 0.616 GB/s | 58.1% | 25% |

Two denominators, both honest. Against the raw line, mccl's 25–30% sits at the low edge
of NCCL-over-TCP's 32–48%; the difference is the IP-over-TB tax that the probe itself
pays, which a bulk-DMA transport would attack. Against what the link demonstrably
carries — the fraction mccl's own scheduling controls — the collective extracts 70%.

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

Still untested: the hierarchical plan, which needs at least two islands of two ranks and
so n ≥ 4, and the mixed-speed fabric island detection exists for.

Full tables and method notes: [ARCHITECTURE.md](ARCHITECTURE.md) §Measured,
[WHITEPAPER.md](WHITEPAPER.md) §5.


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
