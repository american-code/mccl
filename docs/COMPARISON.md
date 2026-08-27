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

NCCL has no in-band lossy compression. mccl does, with exact wire-byte accounting
(verified on interface counters: downcast 2.00×, int8 3.94×, topK/0.01 49.9× fewer
bytes), and the measured rule for when it pays:

**A codec pays only when the fabric's uncompressed all-reduce rate is below the codec's
own encode/decode ceiling.**

The rule was stated from the scalar encoders, whose ceilings all sat below what the
cable delivers, so every codec lost on Thunderbolt and every codec won on Wi-Fi. That
made a falsifiable prediction — lift the ceilings and the Thunderbolt verdict should
flip — and vectorising the encoders tested it. Ceilings are round-trip payload GB/s,
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

Also measured against naive theory: at n=2 the ring beats the tree by up to 1.72× —
same bytes, but the ring is full-duplex while the tree serializes through the root.

Full tables and method notes: [ARCHITECTURE.md](ARCHITECTURE.md) §Measured,
[WHITEPAPER.md](WHITEPAPER.md) §5.
