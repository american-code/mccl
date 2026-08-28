# RDMA over Thunderbolt

> **Status: built to specification, never run on hardware.**
>
> Every machine used to develop this is M1-generation with Thunderbolt 4. RDMA
> over Thunderbolt needs Thunderbolt 5. Not one byte of the data path in this
> document has crossed a queue pair. What *is* verified is stated precisely in
> [What is actually verified](#what-is-actually-verified), and the experiment that
> would settle the rest is in [Validation checklist](#validation-checklist).

macOS 26.2 added an InfiniBand Verbs API for the Thunderbolt 5 controller,
documented in Apple's Technical Note
[TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
(published 2026-03-19, revised 2026-04-13). It is a public API: `infiniband/verbs.h`
and `librdma.tbd` ship in the macOS SDK. mccl has a transport built to it.

---

## What the hardware actually offers

TN3205 describes a deliberately small subset of verbs, and every constraint below
shapes something in mccl's design. They are worth reading before the code.

| | |
|---|---|
| Operations | **Two-sided only.** `IBV_WR_SEND` and `ibv_post_recv`. No RDMA read, no RDMA write. |
| Memory access | `IBV_ACCESS_LOCAL_WRITE` only — the hardware initiates no remote writes |
| Queue-pair type | `IBV_QPT_UC` — unreliable connection |
| Queue pairs | 10 maximum |
| Port | Always 1; Apple controllers have a single port |
| Path MTU | `IBV_MTU_4096` — 4 KiB frames |
| Work requests | 4095 maximum |
| Maximum message | 16,773,120 bytes (4095 × 4096) |
| Buffers | Page-aligned, via `posix_memalign` — each controller sits behind an IOMMU |
| Same-size rule | Sender and receiver must post messages **the same number of frames long** |
| Addressing | 128-bit GID; index 1 is the IPv4-mapped address of the paired IP interface, index 2 the link-local IPv6 |
| Bootstrap | GID, LID, QPN and PSN exchanged out of band — TN3205 suggests TCP over IP-over-Thunderbolt |
| Devices | `rdma_en2` pairs with the Thunderbolt IP interface `en2` |
| Routing | None. Point-to-point only; the application forwards |
| Delivery | **No hardware acknowledgement.** Send completion does not mean the peer received the data, or that it arrived uncorrupted |

Two of those do most of the work in what follows: the same-size rule, which
determines the framing, and the absence of acknowledgement, which determines how
loss is handled.

---

## Linkage strategy

Two facts were measured before choosing, not assumed.

**One.** `infiniband/verbs.h` and `librdma.tbd` are present in the macOS 26.x
SDK — including the Command Line Tools SDK, and including on Macs with no
Thunderbolt 5. `librdma.dylib` is likewise in the dyld shared cache on macOS
26.5, and `ibv_get_device_list` returns a valid *empty* list on an M1. The
library is available far more widely than the hardware.

**Two.** Passing `-lrdma` against any pre-26.2 SDK is a hard link error
(`ld: library 'rdma' not found`). An unconditional `linkedLibrary("rdma")` would
break the build on every older SDK, including CI's.

So the two halves are resolved differently:

- **Compile time — Apple's own header, behind `__has_include`.** Struct layouts,
  enum values and the `ops` vtable offsets come from the SDK rather than from a
  transcription of TN3205's prose. Transcribed structs would be an ABI bet
  against a header we can simply read, and there is no reason to take it. Where
  the header is absent, the shim compiles as a stub whose every entry point fails
  with a precise reason.
- **Link time — `dlopen` plus `dlsym`, and no `-lrdma` at all.** The resulting
  binary carries zero load commands referencing librdma, so it loads on macOS 14
  and runs everywhere.

**The ABI risk this carries, stated plainly.** It is confined to the signatures of
the seventeen `dlsym`'d functions. Those signatures come from the same SDK header
that ships alongside the dylib, so a mismatch would need Apple to change a
signature without changing the header — the same exposure any normally-linked
client has, minus the build-time SDK coupling. The constants mccl hard-codes are
cross-checked against the SDK's enums by `_Static_assert` in
`Sources/CRDMA/shim.c`, so a renumbering fails the build rather than quietly
mis-programming a queue pair.

**The hot path costs nothing extra.** `ibv_post_send`, `ibv_post_recv` and
`ibv_poll_cq` are `static inline` in the header and dispatch through
`qp->context->ops` — a plain indirect call through a vtable in the context
object, identical to what a normally-linked caller executes. They are not
`dlsym`'d and do not appear in the bound symbol table.

### What the SDKs on hand actually contained

| machine | macOS | SDK | `verbs.h` | `librdma.tbd` | `ibv_devices` |
|---|---|---|---|---|---|
| dev (M1) | 26.5.1 | CLT `MacOSX.sdk` | present | present | empty |
| lab-01 (M1 Max) | 26.5.1 | Xcode `MacOSX.sdk` | present | present | empty |
| lab-02 (M1 Max) | 26.5.1 | CLT `MacOSX.sdk` | present | present | empty |
| — | — | `MacOSX15.4.sdk` | absent | absent | — |

Both header and stub library were present everywhere current, and absent from the
15.4 SDK — which is what makes both halves of the strategy necessary rather than
theoretical.

---

## The transport

`RDMATransport` implements the existing `Transport` protocol, so `MeshFabric`,
`Rendezvous`, the planner and the collectives are unchanged above it. That seam
was designed for this.

### Bootstrap

A queue pair has no listener and no accept: it cannot learn about a peer that has
not already been described to it. TN3205 names the fix, and mccl already had the
channel it asks for.

```
dialer                                    acceptor
  |                                          |
  |-- TCP connect (bootstrap transport) ---->|   TCPTransport, per TN3205
  |                                          |
  |-- 40-byte metadata: GID/LID/QPN/PSN ---->|
  |<-- 40-byte metadata --------------------- |   both write first, then read:
  |                                          |   40 bytes fit any socket buffer,
  |                                          |   so neither side can deadlock
  |   Init -> RTR -> RTS                     |   Init -> RTR -> RTS
  |   post slotCount receives                |   post slotCount receives
  |                                          |
  |========= collective traffic =============|   over the queue pair
  |                                          |
  |-- TCP stays open, idle ----------------->|   close notification, and the
                                                 source address MeshFabric's
                                                 announcement check needs
```

The TCP channel is kept for the life of the connection. It carries no collective
traffic; it exists so a peer going away is observable, and so
`MeshFabric.bootstrap`'s address check still has a source address to check
against.

### Device pairing

TN3205: `rdma_en2` corresponds to `en2`, and GID index 1 on `rdma_en2` is `en2`'s
IPv4 address. So the device for a given peer follows from the subnet match
`PathSelection` already performs — find the local interface reaching the peer,
open `rdma_<that interface>`. When nothing matches, the transport fails rather
than picking a device: on a three-port Mac the wrong device is a cable that does
not go where the caller thinks it does.

An interface with a paired RDMA device classifies as a new `InterfaceMedia.rdma`,
which sorts **first** in `PathSelection.mediaOrder`. It also implies
`Topology.Link.Kind.thunderbolt5` outright, with no measurement: TN3205 ships only
on Thunderbolt 5, which is a harder fact than any bandwidth probe.

### Framing

One line of TN3205 determines the whole design:

> Thunderbolt devices require a receiver and sender to post messages which are the
> same number of frames long. For example, if the sender sends a 16K message and
> the receiver posts a 4K or 32K receive buffer the receive operation will fail.

A receive buffer must be posted *before* its message arrives, so the receiver has
to know the size of a message it has not seen. There are two ways out, and only
one survives:

- **Announce each size first.** Every payload becomes two messages — a fixed-size
  length announcement, then the payload. That doubles the message count and puts a
  round trip in front of every write, which on a fabric bought for its latency is
  the wrong trade. It does not even remove the problem: the announcement itself
  must be a size both sides agreed on in advance.
- **Fix the size for the life of the connection.** The receiver keeps receives
  permanently posted and the rule holds by construction.

mccl takes the second. Every message on the wire is exactly `slotBytes`,
negotiated once during the metadata exchange above.

```
slot (slotBytes on the wire, a whole number of 4 KiB frames)
+----------------------------------+---------------------------------+--------+
| header (16 B)                    | payload (0 .. slotBytes-16)     | unused |
+----------------------------------+---------------------------------+--------+
  0..3   magic 'MCR1'
  4..7   sequence     per-direction, from 0, wraps at 2^32
  8..11  payloadBytes
 12..15  flags        bit 0 = closing
```

The unused tail is deliberately **not** zeroed — that would be a full memset per
slot on the hot path to hide nothing, since the receiver honours `payloadBytes`.

**Defaults.** 64 KiB slots (16 frames), 32 slots per direction: 2 MiB of
registered memory per connection and 64 of the 4095 available work requests. Not
tuned against hardware, because there is none to tune against.

**Chunking.** A write longer than the payload capacity spans as many slots as it
needs, so the 16,773,120-byte message ceiling is respected by construction — the
default slot is 256× below it. `RDMASlotGeometry` additionally rejects any
configured slot above the ceiling, below one frame, not a whole number of frames,
or needing more than 4095 work requests.

**The padding cost, named.** A write smaller than a slot still occupies a whole
slot. mccl's traffic makes that cheap: `Channel.sendFrame` emits its 24-byte
`WireHeader` and the payload as a *single* write, so padding is paid once per
collective step rather than once per header. A 4 MiB ring step at the default slot
is 64 full slots and one partial — under 1% waste. A stream of tiny writes would
be far worse, and this transport is not built for one.

**Back-pressure.** The receive ring is the flow-control window, exactly as a
socket buffer is. TN3205: send completion "indicates that the peer has posted
sufficient receive requests". A sender that outruns the ring blocks until the peer
consumes some — which is why every collective already runs its send and its
receive on separate queues (`runConcurrently`), and why a single-threaded
send-then-receive of more than a ring's worth would deadlock here just as it would
over TCP.

### Handling UC honestly

TN3205 is explicit:

> Completion of a send operation indicates that the peer has posted sufficient
> receive requests to receive all sent frames and that the Thunderbolt hardware
> has sent the buffers to the peer. Completion does not indicate the receiver has
> successfully received the sent data or that data was not corrupted in flight
> because Thunderbolt does not perform Acks in hardware.

A Thunderbolt link is reliable in practice — a short, point-to-point,
CRC-protected cable, which is why Apple considers UC sufficient. But "reliable in
practice" is not something a collectives library may quietly depend on. **A
dropped slot in an all-reduce does not announce itself.** It becomes a
plausible-looking tensor with a hole in it, which then trains a model. Silent
numerical corruption is the worst outcome this library can produce.

So every slot carries a sequence number and the receiver checks it. A gap — or a
repeat — is a hard error that fails the collective and says what it means.

mccl deliberately does **not** attempt recovery. A retransmit protocol over UC
would reimplement the reliability TCP already provides, and if a cluster turns out
to need one, TCP is the right transport for that cluster. Detecting is honest;
hiding would not be.

**What this does not catch, stated plainly:** corruption. A sequence number
detects a *missing* slot, not a *wrong* one. TN3205 says data may be "corrupted in
flight"; the Thunderbolt link layer does its own CRC, but mccl adds no end-to-end
payload checksum, so silent corruption within a delivered slot would pass. That is
a known gap, not an oversight — a checksum is a full pass over every byte, and
adding one should be a decision made against a measurement, on hardware, by
someone who has seen the error rate. The slot header reserves a `flags` word for
it.

---

## What is actually verified

Being precise about this matters more than the feature.

**Verified by compilation, on this machine:**

- The shim compiles against the macOS 26.5 SDK **with** `infiniband/verbs.h`, and
  every `_Static_assert` cross-checking mccl's constants against Apple's enums
  passes.
- The shim compiles against `MacOSX15.4.sdk`, which has **no** verbs header, as a
  stub — and the whole suite passes on that path, exercised on a machine that
  *does* have the header via `swift test -Xcc -DMCCL_RDMA_FORCE_STUB`. Without
  that flag the no-header branch would only ever run on CI, where nobody watches
  it; CI now builds both paths explicitly for the same reason.
- The built binaries contain **zero** load-time references to librdma.

**Verified at runtime, on an M1 with no Thunderbolt 5:**

- `dlopen("/usr/lib/librdma.dylib")` succeeds and all seventeen symbols resolve.
- `ibv_get_device_list` returns a valid empty list.
- `RDMATransport.unusableReason()` reports the no-devices case with enablement
  instructions, and `mcclbench --transport rdma --distributed` refuses at parse
  time quoting it.

**Verified against a mock that enforces TN3205's rules** (`MockVerbs`, 70 tests):

- Reset → Init → RTR → RTS in that order, with attribute masks 57 / 1,053,057 /
  65,537 — the values TN3205's listings imply.
- `IBV_QPT_UC`, `IBV_MTU_4096`, `IBV_ACCESS_LOCAL_WRITE`, `IBV_WR_SEND`, port 1,
  GID index 1, hop limit 1.
- The ring is page-aligned (the mock refuses a registration that is not) and both
  directions are covered by one memory region.
- Byte-stream round trips at every boundary: one byte, exactly one slot, one over,
  a payload larger than the whole ring, and reads at sizes the writer never used.
- Every message posted in either direction is exactly `slotBytes` — the same-size
  rule, asserted rather than assumed.
- A dropped slot surfaces as `rdmaSequenceGap`, never as corrupt data.
- Back-pressure: a sender outrunning the receive ring blocks rather than failing.
- Teardown releases every context, protection domain, memory region, completion
  queue and queue pair, including after a failed construction.
- The eleventh queue pair is refused.

**Not verified by anything here:** that Apple's hardware behaves as its technote
says. Every number below is an expectation, not a measurement.

---

## Validation checklist

For the first person with Thunderbolt 5 hardware. Please run these in order and
record what you see — the results belong in `docs/ARCHITECTURE.md` alongside the
existing measured runs, and this section should be replaced by them.

### 1. Enable RDMA on every node

Cannot be done remotely, even with `sudo`.

1. Boot into macOS Recovery ([how](https://support.apple.com/en-us/102518/)).
2. Utilities → Terminal.
3. `rdma_ctl enable`
4. Reboot.

While you are there, TN3205 also recommends for cluster use: disable idle sleep
(Thunderbolt has no Wake-on-LAN, so a sleeping node is simply unreachable), enable
automatic login, and turn on "Start up automatically after a power failure".

### 2. Confirm the devices exist

```sh
ibv_devices          # expect one rdma_enN per Thunderbolt port
ibv_devinfo          # expect transport "Thunderbolt (100)", phys_port_cnt 1
ibv_devinfo -d rdma_en2 -v | grep GID
```

GID[1] should be `::ffff:<the IPv4 address of en2>`. If `ibv_devices` prints an
empty table, RDMA is not enabled or the Mac has no Thunderbolt 5 controller.

Then mccl's own view, which should agree:

```sh
swift test --filter RDMAHardwareTests
```

On a machine without RDMA these skip with the reason. On a machine with it, the
three skipped tests run: device/interface pairing, IPv4-mapped GID at index 1, and
a two-device loopback transfer. **`testLoopbackTransferOverRealHardware` is the
first moment any of this code touches hardware.** If it fails, everything below is
moot — please open an issue with the failure and `ibv_devinfo -v` output.

### 3. Cable the cluster

All RDMA links are point-to-point. Fully connect every pair if you can; TN3205
diagrams two, three, four and five-node layouts. **If you cable a loop, disable
the Thunderbolt bridge** — System Settings → Network → Thunderbolt Bridge → Make
Inactive — or `bridge0` will spanning-tree your ring.

Confirm IP-over-Thunderbolt still works between every pair (`ping`), because the
RDMA bootstrap rides on it.

### 4. Establish the TCP baseline first

Do this **before** the RDMA sweep, on the same cables, in the same session. The
existing numbers in this repo were taken on Thunderbolt 4 and are not a baseline
for Thunderbolt 5.

```sh
# On each node:
mcclprobe serve

# From node 0:
mcclprobe measure <node1>:7777 <node2>:7777 --out topology-tb5.json
```

Record the measured bandwidth and RTT per link, and the negotiated line rate.

Rank 0 mints the token and the joiners are given it. **Keep every ssh session
attached for the whole run** — a rank orphaned by an exiting ssh session fails its
outbound dial with `EHOSTUNREACH` on macOS 26, for reasons that have nothing to do
with RDMA and look exactly like a bug in mccl
([USAGE.md](USAGE.md#launching-across-machines-keep-the-ssh-session-attached)).

```sh
SWEEP="--world-size 4 --min-bytes 1024 --max-bytes 268435456"

# rank 0 mints the token; --emit-token prints MCCL_TOKEN=<text> as its first line
ssh node-0 "mcclbench --rank 0 $SWEEP --transport tcp --bind <node-0-tb-address> \
            --emit-token --csv tcp-tb5.csv --label tcp-tb5" &
# then, with that token, ranks 1..3
ssh node-N "mcclbench --rank N $SWEEP --transport tcp --bind <node-N-tb-address> \
            --token '<token>'" &
wait
```

`--bind` matters here: on a multi-homed machine it is what pins the rendezvous —
and therefore the advertised address — to the Thunderbolt cable rather than to
whatever Wi-Fi happens to answer first.

### 5. The RDMA sweep

Identical, with one word changed:

```sh
ssh node-0 "mcclbench --rank 0 $SWEEP --transport rdma --bind <node-0-tb-address> \
            --emit-token --csv rdma-tb5.csv --label rdma-tb5" &
ssh node-N "mcclbench --rank N $SWEEP --transport rdma --bind <node-N-tb-address> \
            --token '<token>'" &
wait
```

If a rank cannot use RDMA it says so and exits at argument-parse time rather than
part way into bring-up, naming which of "no library", "no devices" or "not
enabled" applies.

Sweep the slot geometry too — 16 KiB, 64 KiB (default), 256 KiB, 1 MiB — since it
was chosen from the spec rather than from measurement, and it is the single
likeliest thing to be wrong. It is currently a compile-time default in
`RDMASlotGeometry`; a `--slot-bytes` flag is worth adding if the sweep shows it
matters.

### 6. What to expect, and what would falsify it

**Bandwidth.** On Thunderbolt 4, mccl over IP-over-TB reached 42–58% of line rate
before mccl's own scheduling even began — the stack tax measured in
`WHITEPAPER.md` §6.1. RDMA does not traverse that stack. If the transport works,
the honest prediction is that most of that 42–58% gap closes, and the remaining
loss is mccl's framing and scheduling rather than the kernel's. **A large-message
`busbw` that is not meaningfully above the TCP run on the same cable means
something is wrong** — most likely slot geometry, or the ring being too shallow to
keep the link full.

**Latency.** Small-message latency is where RDMA should win by the largest factor,
and it is the number most worth publishing.

⚠️ **On the figures circulating in secondary reporting.** A "~3 µs versus ~300 µs"
pairing has appeared in blog coverage of TN3205. **It does not come from Apple.**
TN3205 contains no latency figures of any kind — its only latency language is
qualitative ("IP interfaces have higher latency and CPU overhead than equivalent
RDMA interfaces"). The circulating numbers are unattributed, mutually inconsistent
across sources (under 50 µs, 5–9 µs, 3 µs), and could not be traced to any primary
measurement. **Do not use them as a target.** The strongest sourced claim is the
MLX documentation's, that its JACCL backend achieves "communication latency an
order of magnitude lower than the ring backend" — a ratio, with no absolute figure.
Measure it yourself; that measurement would be a genuinely novel contribution.

**The crossover.** mccl's tree/ring crossover and its compression rule are both
functions of measured bandwidth and latency. If latency drops by an order of
magnitude, the crossover moves substantially toward ring, and the compression
crossover moves against compression — a codec pays only when the fabric's
uncompressed all-reduce rate is below that codec's own encode/decode ceiling
(`COMPARISON.md`), and a much faster fabric puts more of the range above that
ceiling. Both should be re-derived from the new measurements rather than adjusted
by intuition. `mcclprobe plan topology-tb5.json --bytes N` replays the planner
over the new topology and is the quickest way to see where it now switches.

### 7. The head-to-head against JACCL

This is the comparison every adopter will ask for, and it needs TB5 hardware on
both sides, so it belongs here rather than in the repo's existing measurements.

Run the same sweep — same message sizes, same collective, same dtype, same
machines, same cables, same session — under JACCL and under mccl-over-RDMA:

- **JACCL**: `mlx.distributed` with `backend="jaccl"`, or the standalone C++
  `allreduce_bench` in `mlx/distributed/jaccl/lib/examples/`, which the MLX repo
  describes as "similar in spirit to the NCCL all-reduce benchmark".
- **mccl**: the `--transport rdma` sweep from step 5.

Match the definitions before comparing the numbers: `mcclbench` reports
algorithmic and bus bandwidth with the bus factor stated
([COMPARISON.md](COMPARISON.md)), and a benchmark that divides by a different
constant will disagree with it for reasons that have nothing to do with either
implementation.

Expect JACCL to win on a uniform TB5 mesh, and report it plainly if it does. It
was co-developed with the hardware and has been run on it; mccl's transport, at
the time you are reading this, has not. A result showing mccl ahead on that
fabric is more likely to be a methodology error than a finding, and should be
checked twice before it is published.

**On the cluster this was written on, the comparison is not a benchmark at all.**
These are M1-generation machines with Thunderbolt 4: mccl runs on them, and JACCL
cannot form a group at all, because its fabric needs a TB5 controller. That is a
works-versus-does-not-work difference in coverage, not a performance win, and it
should never be reported as one.

### 8. Correctness under load

Run the existing distributed correctness path over RDMA, not just the benchmark:

```sh
mccltrain   # 2-rank loss trajectory against a single-process run
```

The loss-trajectory equivalence (4.8e-07 over Thunderbolt 4) is the check that
matters most, because it is the one that would catch a framing bug that a
throughput sweep would not. **Watch for `mcclProtocolViolation` mentioning a slot
sequence gap.** On a link Apple considers reliable enough for UC, that should never
fire. If it does, it is the single most important thing to report: it means either
mccl's framing is wrong or the link is losing data, and both are worth knowing.

---

## Where this stands next to Apple's own stack

Apple co-developed RDMA over Thunderbolt with **JACCL** — "Jack and Angelos'
Collective Communication Library", named for Jack Beasley, who led the RDMA work.
It is the `jaccl` backend of MLX Distributed, lives in
[`ml-explore/mlx`](https://github.com/ml-explore/mlx/tree/main/mlx/distributed/jaccl),
and is MIT-licensed. TN3205 links to it directly.

**The honest headline: for a uniform Thunderbolt 5 mesh running MLX, use JACCL.**
It is Apple's, co-developed with the silicon, in-tree, tested on the hardware, and
selected with one argument (`mx.distributed.init(backend="jaccl")`). mccl's RDMA
transport is built to the same technote and has never been run. There is no
version of this comparison in which mccl is the better choice for that case, and
pretending otherwise would waste the reader's time.

Three corrections to a framing that is easy to reach for and is wrong:

- **JACCL is not MLX-only.** It builds standalone from the MLX tree and exposes a
  C++ API; Apple's WWDC26 session 233 says explicitly that it "can be built
  without MLX" and "is not limited to machine learning". A C++ project on a
  uniform TB5 mesh is well served by it. So mccl's C ABI is a *parity* feature
  here, useful for runtimes that cannot link C++ but not a moat — "works outside
  MLX" describes both libraries.
- **JACCL is not mesh-only.** The MLX docs describe fully-connected topologies,
  but the library also implements a ring mode (`prefer_ring`, `JACCL_RING`) for
  large messages.
- **JACCL already chooses its own algorithm.** It selects between mesh and ring
  by message size. mccl's planner is therefore not "planning versus none"; the
  narrower and accurate claim is *measured* per-link bandwidth and latency across
  *mixed-speed* fabrics with a solved crossover, against size-based selection on
  a uniform mesh.

With that said, the two do different jobs, and the differences are real:

| | JACCL | mccl |
|---|---|---|
| Fabric | RDMA over TB5 only; no TCP data path | TCP anywhere, RDMA where available, **mixed in one world** |
| Hardware | Apple silicon with Thunderbolt 5, macOS 26.2+ | any Mac; RDMA is an optional accelerant |
| Interface | C++ API, standalone-buildable | C ABI (`libmccl.dylib`), plus Swift — parity, not a moat |
| Topology | validated against a matrix you supply; mesh/ring chosen by message size; discovery is a separate ssh helper | measured in-process by `mcclprobe`; mixed-speed fabrics; tree/ring crossover solved from bandwidth and latency |
| Heterogeneous fabrics | not addressed | per-pair path selection, island detection, hierarchical plans |
| Wire compression | none | int8 / downcast / top-k, with a measured crossover rule |
| Collectives | `all_sum`, `all_max`, `all_min`, `all_gather`, `send`, `recv`, `barrier` | plus `broadcast`, `reduce`, `reduce_scatter` |
| Hardware validation | Apple's, on TB5 clusters | TB4 and Wi-Fi for TCP; **none for RDMA** |

**Where mccl stands alone**, and these are narrow but genuine:

1. **Mixed fabrics — the one JACCL structurally cannot reach.** A Thunderbolt
   island plus an Ethernet or Wi-Fi bridge rank is a world mccl forms, measures
   and plans over, every pair on the best cable the two of them share. JACCL
   hard-requires RDMA over TB5 and has no data path for a node that is not on the
   mesh; there is no degrading onto Ethernet for one awkward rank. This is the
   segment Apple's stack does not serve, and it is the common shape of a cluster
   assembled from machines someone already owned.
2. **Compression where the fabric is slow.** Nothing in JACCL does in-band lossy
   compression. The crossover rule pays exactly where links are slow — which is to
   say, precisely where RDMA is not available. The two are complements, not
   competitors.
3. **Hardware older than Thunderbolt 5.** The TB5 requirement, plus its
   Recovery-mode enablement, excludes every M1- and M2-generation Mac. mccl runs
   on all of them today — including the machines this was written on.

And now a fifth thing that is not differentiation but the opposite: mccl speaks
the same TN3205 verbs JACCL rides. On the hardware where Apple's stack is the
right answer, mccl meets it rather than competing with it — the same queue pairs,
the same constraints, reached through a C ABI and a planner that also work on the
machines Apple's stack cannot use.

---

## Reading the code

| file | what is in it |
|---|---|
| `Sources/CRDMA/include/crdma.h` | the C surface, and the linkage strategy in full |
| `Sources/CRDMA/shim.c` | dlopen binding, `_Static_assert` cross-checks, both compilation paths |
| `Sources/MCCL/RDMA/RDMAVerbs.swift` | the `RDMAVerbs` protocol, `RDMASpec` constants, `SystemVerbs` |
| `Sources/MCCL/RDMA/RDMAFraming.swift` | slot header, geometry rules, sequence checker, chunking |
| `Sources/MCCL/RDMA/RDMAConnection.swift` | queue-pair lifecycle, ring, completion arbitration |
| `Sources/MCCL/RDMA/RDMATransport.swift` | `Transport`/`Listener`/`Channel`, availability, device pairing |
| `Tests/MCCLTests/RDMAMockVerbs.swift` | the mock that enforces TN3205's rules |
| `Tests/MCCLTests/RDMATests.swift` | 70 tests, three of them hardware-gated |
