/*
 * crdma.h — a flat C surface over Apple's RDMA-over-Thunderbolt verbs.
 *
 * Apple shipped an InfiniBand Verbs API for the Thunderbolt 5 controller in
 * macOS 26.2 (Technical Note TN3205). This shim is the only place in mccl that
 * touches `<infiniband/verbs.h>`. Everything above it — the QP state machine,
 * the framing, the chunking, the loss check — is Swift, and is written against
 * the `RDMAVerbs` protocol so it can be driven by a mock on hardware that has
 * no Thunderbolt 5 (which, today, is all of our hardware).
 *
 * ---------------------------------------------------------------------------
 * Linkage strategy, and why it is a hybrid
 * ---------------------------------------------------------------------------
 *
 * Two facts decide this, both measured rather than assumed:
 *
 *   1. `infiniband/verbs.h` and `librdma.tbd` are present in the macOS 26.x
 *      SDK — including the Command Line Tools SDK, and including on machines
 *      with no Thunderbolt 5 hardware at all. `librdma.dylib` is likewise
 *      present in the dyld shared cache on macOS 26.5, and `ibv_get_device_list`
 *      returns a valid empty list on an M1. So the *library* is available far
 *      more widely than the *hardware*.
 *
 *   2. Passing `-lrdma` against any pre-26.2 SDK is a hard link error
 *      ("ld: library 'rdma' not found"). An unconditional `linkedLibrary("rdma")`
 *      in Package.swift would therefore break the build on every older SDK,
 *      which includes the CI runner.
 *
 * So the two halves are resolved differently, each in the way that carries the
 * least risk:
 *
 *   * COMPILE TIME — use Apple's own header, gated on `__has_include`. The
 *     struct layouts, enum values and the `ops` vtable offsets come from the
 *     SDK, not from a transcription of the prose in TN3205. Transcribed structs
 *     would be an ABI bet against a header we can read; there is no reason to
 *     take that bet. When the header is absent the shim still compiles, as a
 *     stub whose every entry point fails with a precise reason.
 *
 *   * LINK TIME — no `-lrdma`, no link-time dependency of any kind. The
 *     exported entry points are resolved once with `dlopen("/usr/lib/librdma.dylib")`
 *     plus `dlsym`. The resulting binary has zero load commands referencing
 *     librdma, so it loads on macOS 14 and runs everywhere.
 *
 * The residual ABI risk is small and worth naming precisely: it is confined to
 * the signatures of the ~17 dlsym'd functions. Those signatures come from the
 * same SDK header that ships alongside the dylib, so a mismatch would require
 * Apple to change a signature without changing the header — the same exposure
 * any normally-linked client has, minus the build-time SDK coupling.
 *
 * The hot path carries no dlsym cost at all. `ibv_post_send`, `ibv_post_recv`
 * and `ibv_poll_cq` are `static inline` in the header and dispatch through
 * `qp->context->ops.*` — a plain indirect call through a vtable in the context
 * object, identical to what a normally-linked caller executes.
 */

#ifndef MCCL_CRDMA_H
#define MCCL_CRDMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Whether this translation unit was compiled against a real verbs header.
 * 0 means every call below fails with MCCL_RDMA_NO_HEADER.
 */
int mccl_rdma_built_with_headers(void);

/* ------------------------------------------------------------------------ */
/* Constants                                                                */
/* ------------------------------------------------------------------------ */
/*
 * These are the values the Swift state machine passes back down, so that the
 * attribute masks in `RDMAQueuePair` are the spec's masks and are assertable in
 * a unit test without any hardware.
 *
 * When the SDK header is present these are cross-checked against it with
 * `_Static_assert` in shim.c, so a future SDK that renumbered anything fails
 * the build rather than corrupting a queue pair at runtime.
 */

enum {
    MCCL_IBV_QPT_UC              = 3,
    MCCL_IBV_MTU_4096            = 5,
    MCCL_IBV_ACCESS_LOCAL_WRITE  = 1,
    MCCL_IBV_WR_SEND             = 2,
    MCCL_IBV_SEND_SIGNALED       = 2,
    MCCL_IBV_WC_SUCCESS          = 0,
    MCCL_IBV_PORT_ACTIVE         = 4,

    MCCL_IBV_QPS_RESET           = 0,
    MCCL_IBV_QPS_INIT            = 1,
    MCCL_IBV_QPS_RTR             = 2,
    MCCL_IBV_QPS_RTS             = 3,
    MCCL_IBV_QPS_ERR             = 6,

    MCCL_IBV_QP_STATE            = 1,
    MCCL_IBV_QP_ACCESS_FLAGS     = 8,
    MCCL_IBV_QP_PKEY_INDEX       = 16,
    MCCL_IBV_QP_PORT             = 32,
    MCCL_IBV_QP_AV               = 128,
    MCCL_IBV_QP_PATH_MTU         = 256,
    MCCL_IBV_QP_RQ_PSN           = 4096,
    MCCL_IBV_QP_SQ_PSN           = 65536,
    MCCL_IBV_QP_DEST_QPN         = 1048576
};

/* Failure codes returned by the entry points below (negated errno otherwise). */
enum {
    MCCL_RDMA_OK          =  0,
    MCCL_RDMA_NO_HEADER   = -1001, /* built against an SDK without verbs.h   */
    MCCL_RDMA_NO_LIBRARY  = -1002, /* dlopen("/usr/lib/librdma.dylib") failed */
    MCCL_RDMA_NO_SYMBOL   = -1003, /* dlsym failed for a required entry point */
    MCCL_RDMA_FAILED      = -1004  /* the verbs call itself failed            */
};

/* ------------------------------------------------------------------------ */
/* Library loading                                                          */
/* ------------------------------------------------------------------------ */

/*
 * Attempts the dlopen + dlsym once (idempotent, thread-safe). Returns
 * MCCL_RDMA_OK, MCCL_RDMA_NO_HEADER, MCCL_RDMA_NO_LIBRARY or MCCL_RDMA_NO_SYMBOL.
 */
int mccl_rdma_load(void);

/*
 * A human-readable reason the library is unusable, or NULL when it loaded.
 * Points at static or once-allocated storage; the caller must not free it.
 */
const char *mccl_rdma_load_error(void);

/* ------------------------------------------------------------------------ */
/* Devices                                                                  */
/* ------------------------------------------------------------------------ */

/* Number of RDMA devices, or a negative error code. 0 is not an error: it is
 * what every Mac without Thunderbolt 5 (or without `rdma_ctl enable`) reports. */
int mccl_rdma_device_count(void);

/* Writes the NUL-terminated name of device `index` (e.g. "rdma_en2"). */
int mccl_rdma_device_name(int index, char *out, size_t capacity);

/* Opaque handles. Each is really the corresponding `struct ibv_*` pointer. */
typedef void *mccl_rdma_context;
typedef void *mccl_rdma_pd;
typedef void *mccl_rdma_mr;
typedef void *mccl_rdma_cq;
typedef void *mccl_rdma_qp;

mccl_rdma_context mccl_rdma_open_device(int index);
void mccl_rdma_close_device(mccl_rdma_context context);

/*
 * Port attributes. TN3205: Apple Thunderbolt controllers have a single port, so
 * `port` is always 1.
 */
int mccl_rdma_query_port(mccl_rdma_context context, uint8_t port,
                         uint16_t *out_lid, uint32_t *out_state);

/*
 * GID at `index` on `port`. TN3205: index 1 is the IPv4-mapped address of the
 * paired IP-over-Thunderbolt interface, index 2 the link-local IPv6.
 */
int mccl_rdma_query_gid(mccl_rdma_context context, uint8_t port, int index,
                        uint8_t out_gid[16]);

/* ------------------------------------------------------------------------ */
/* Memory                                                                   */
/* ------------------------------------------------------------------------ */

mccl_rdma_pd mccl_rdma_alloc_pd(mccl_rdma_context context);
void mccl_rdma_dealloc_pd(mccl_rdma_pd pd);

/*
 * Registers `length` bytes at `address`. TN3205 requires page-aligned memory
 * because each Thunderbolt controller sits behind an IOMMU, and requires
 * `access` to be MCCL_IBV_ACCESS_LOCAL_WRITE only — the hardware does not do
 * remote writes.
 */
mccl_rdma_mr mccl_rdma_reg_mr(mccl_rdma_pd pd, void *address, size_t length, int access);
uint32_t mccl_rdma_mr_lkey(mccl_rdma_mr mr);
void mccl_rdma_dereg_mr(mccl_rdma_mr mr);

/* ------------------------------------------------------------------------ */
/* Queues                                                                   */
/* ------------------------------------------------------------------------ */

mccl_rdma_cq mccl_rdma_create_cq(mccl_rdma_context context, int depth);
void mccl_rdma_destroy_cq(mccl_rdma_cq cq);

mccl_rdma_qp mccl_rdma_create_qp(mccl_rdma_pd pd, mccl_rdma_cq cq,
                                 int max_send_wr, int max_recv_wr, int qp_type);
uint32_t mccl_rdma_qp_num(mccl_rdma_qp qp);
void mccl_rdma_destroy_qp(mccl_rdma_qp qp);

/*
 * The three state transitions, each taking its attribute mask from the caller
 * so the mask is a Swift-side, unit-testable decision.
 */
int mccl_rdma_modify_qp_to_init(mccl_rdma_qp qp, uint8_t port_num,
                                uint16_t pkey_index, uint32_t access_flags,
                                int attr_mask);

int mccl_rdma_modify_qp_to_rtr(mccl_rdma_qp qp, uint32_t path_mtu,
                               uint32_t rq_psn, uint32_t dest_qp_num,
                               const uint8_t dgid[16], uint16_t dlid,
                               uint8_t sgid_index, uint8_t hop_limit,
                               uint8_t port_num, int attr_mask);

int mccl_rdma_modify_qp_to_rts(mccl_rdma_qp qp, uint32_t sq_psn, int attr_mask);

/* ------------------------------------------------------------------------ */
/* Data path                                                                */
/* ------------------------------------------------------------------------ */

int mccl_rdma_post_send(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey, unsigned int send_flags,
                        int opcode);

int mccl_rdma_post_recv(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey);

/* One completion, flattened to the fields mccl uses. */
typedef struct {
    uint64_t wr_id;
    uint32_t status;
    uint32_t opcode;
    uint32_t byte_len;
} mccl_rdma_completion;

/* Returns the number of completions written, 0 for none, or a negative code. */
int mccl_rdma_poll_cq(mccl_rdma_cq cq, int max_entries, mccl_rdma_completion *out);

/* Static string for a work-completion status (`ibv_wc_status_str`). */
const char *mccl_rdma_status_string(uint32_t status);

#ifdef __cplusplus
}
#endif

#endif /* MCCL_CRDMA_H */
