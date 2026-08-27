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

## mccl measured (2-node Thunderbolt, fp32 sum all-reduce, ring, best of 5)

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
own encode/decode ceiling** (measured scalar-Swift ceilings on M1 Max: downcast
~0.50 GB/s, topK ~0.28, int8 ~0.16). On the ~1 GB/s TB link every codec loses; on the
0.05 GB/s WiFi path every codec wins — topK by 5.4×, downcast at 1.89× of its
theoretical 2.00×. Vectorizing the scalar encoders raises the ceilings and moves the
crossover onto faster fabrics.

Also measured against naive theory: at n=2 the ring beats the tree by up to 1.72× —
same bytes, but the ring is full-duplex while the tree serializes through the root.

Full tables and method notes: [ARCHITECTURE.md](ARCHITECTURE.md) §Measured,
[WHITEPAPER.md](WHITEPAPER.md) §5.
