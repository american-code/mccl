/*
 * mccl.h — C ABI for mccl, the Metal Collective Communications Library.
 *
 * Deliberately shaped like NCCL: a runtime that already speaks ncclComm_t
 * should need renaming and nothing else. Where mccl differs from NCCL, it is
 * because a Mac cluster differs from a DGX box, and the difference is called
 * out in a comment rather than hidden.
 *
 * Link against libmccl.dylib:
 *
 *     cc app.c -I<mccl>/Sources/MCCLShim/include \
 *              -L<mccl>/.build/debug -lmccl -Wl,-rpath,<mccl>/.build/debug
 *
 * Threading. The plain collective calls block until the collective has
 * completed on this rank. `mcclStream_t` is not an execution context — it names
 * an independent *sequence* of collectives, which matters only to
 * `mcclCompressTopK`, whose error-feedback residual is per-stream.
 *
 * For non-blocking issue, see mcclAllReduceAsync / mcclRequestWait /
 * mcclRequestTest below. That is deliberately a *separate* entry point rather
 * than a change of meaning for the existing one: NCCL gets its asynchrony from
 * the CUDA stream the caller already has, mccl has no device queue to borrow,
 * and turning `mcclAllReduce` non-blocking would break every caller that reads
 * its buffer on return.
 *
 * One communicator must not be used concurrently from two threads. Different
 * communicators are independent. Issuing several asynchronous operations from
 * one thread is fine and is the point of them.
 *
 * License: Apache-2.0.
 */

#ifndef MCCL_H
#define MCCL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MCCL_MAJOR 0
#define MCCL_MINOR 5
#define MCCL_PATCH 0
#define MCCL_VERSION_CODE (MCCL_MAJOR * 10000 + MCCL_MINOR * 100 + MCCL_PATCH)

/* ---------------------------------------------------------------- results */

/*
 * One-to-one with Swift's MCCLError. Never renumber: these values cross the
 * ABI boundary.
 */
typedef enum {
    mcclSuccess                = 0,
    mcclNotImplemented         = 1,
    mcclInvalidArgument        = 2,
    mcclRankOutOfRange         = 3,
    /* The communicator has no transport fabric. */
    mcclNoFabric               = 4,
    mcclBufferTooSmall         = 5,
    mcclConnectionClosed       = 6,
    mcclProtocolViolation      = 7,
    mcclSocketFailure          = 8,
    mcclTimedOut               = 9,
    mcclUnsupportedCompression = 10,
    mcclTopologyInvalid        = 11,
    /* An error that did not originate in mccl (an allocation failure, say). */
    mcclInternalError          = 12,
    /*
     * RDMA over Thunderbolt failed, or is unavailable on this machine. Call
     * mcclGetLastError for which: the message distinguishes "no librdma"
     * (macOS older than 26.2), "no devices" (no Thunderbolt 5 controller, or
     * RDMA not enabled with `rdma_ctl enable`) and a verbs call that failed.
     * A lost slot is reported as mcclProtocolViolation, not as this.
     */
    mcclRdmaFailure            = 13,
    mcclNumResults             = 14
} mcclResult_t;

/* Static, human-readable name of a result code. Never NULL. */
const char* mcclGetErrorString(mcclResult_t result);

/* ------------------------------------------------------------ data types */

/*
 * Values match DataType.wireCode, the discriminator mccl already puts in every
 * frame header, so a C caller and a Swift caller cannot disagree about a dtype.
 */
typedef enum {
    mcclFloat32  = 0,
    mcclFloat16  = 1,
    mcclBfloat16 = 2,
    mcclInt32    = 3,
    mcclInt8     = 4,
    mcclNumTypes = 5
} mcclDataType_t;

/* NCCL spellings, for source compatibility. */
#define mcclFloat  mcclFloat32
#define mcclHalf   mcclFloat16
#define mcclInt    mcclInt32
#define mcclChar   mcclInt8

typedef enum {
    mcclSum     = 0,
    mcclProd    = 1,
    mcclMin     = 2,
    mcclMax     = 3,
    mcclAvg     = 4,
    mcclNumOps  = 5
} mcclRedOp_t;

/*
 * In-flight wire compression. mccl-specific: NCCL has no equivalent, because on
 * an NVLink fabric the wire is not the bottleneck. On a Mac cluster it is.
 *
 * Compression is per-hop and never changes the dtype the caller sees. A scheme
 * that cannot help a given dtype degrades to raw rather than failing.
 */
typedef enum {
    mcclCompressNone          = 0,
    /* fp32 -> fp16 on the wire. Relative error <= 2^-11. */
    mcclCompressDowncast      = 1,
    /* Per-block absmax scale + int8 payload. `blockSize` selects the block. */
    mcclCompressInt8Blockwise = 2,
    /*
     * Top-k sparsification with error feedback. `fraction` in (0, 1].
     * All-reduce only, sum/avg only, floating-point only: the un-sent remainder
     * is accumulated per (communicator, stream) and added back on the next
     * call, so a training run converges on the uncompressed answer. That state
     * is why it is not available on the other collectives.
     */
    mcclCompressTopK          = 3
} mcclCompression_t;

/* ---------------------------------------------------------- communicators */

typedef struct mcclComm* mcclComm_t;

/*
 * Not an execution context. See the threading note at the top: this names a
 * sequence of collectives so that stateful compression can keep one residual
 * per sequence. NULL is the default stream.
 */
typedef void* mcclStream_t;

/* Builds a stream handle from an integer id. Ids are opaque and need only be
 * distinct; rank r and rank s must use the same id for the same tensor. */
static inline mcclStream_t mcclStreamFromId(uint32_t id) {
    return (mcclStream_t)(uintptr_t)id;
}

/*
 * Detail for the most recent failure on `comm` — the full Swift error text,
 * including the offending values. Pass NULL for failures that happened before a
 * communicator existed (mcclGetUniqueId, mcclCommInitRank). Never NULL; the
 * empty string means "no failure recorded". The pointer stays valid until the
 * next failing call on the same communicator.
 */
const char* mcclGetLastError(mcclComm_t comm);

#define MCCL_UNIQUE_ID_BYTES 128

/*
 * The token that lets ranks find each other, as in NCCL: opaque bytes, produced
 * once and distributed out of band (MPI broadcast, a shared file, an env var).
 *
 * Unlike NCCL's, mccl's payload is printable text — `mccl1:<nonce>:<host>:<port>`
 * — precisely so it *can* travel through a shell. Use mcclUniqueIdToString /
 * mcclUniqueIdFromString rather than reading `internal` directly.
 */
typedef struct { char internal[MCCL_UNIQUE_ID_BYTES]; } mcclUniqueId;

/*
 * Binds rank 0's rendezvous listener and fills in the token describing it.
 *
 * Call exactly once, in the process that will host rank 0, before any rank
 * calls mcclCommInitRank. The listener stays bound until the rendezvous
 * completes; mcclUniqueIdDiscard releases it if the job is abandoned.
 */
mcclResult_t mcclGetUniqueId(mcclUniqueId* uniqueId);

/* Releases a token's listener without running a rendezvous. */
mcclResult_t mcclUniqueIdDiscard(const mcclUniqueId* uniqueId);

/* Token <-> text. `buffer` needs MCCL_UNIQUE_ID_BYTES to always fit. */
mcclResult_t mcclUniqueIdToString(const mcclUniqueId* uniqueId, char* buffer, size_t bufferSize);
mcclResult_t mcclUniqueIdFromString(const char* text, mcclUniqueId* uniqueId);

/*
 * Joins the world named by `commId` as `rank` of `nranks`, and blocks until
 * every rank has done the same.
 *
 * The by-value signature is NCCL's; the exported symbol takes the token by
 * pointer instead, because a 128-byte struct passed by value has a different
 * ABI on arm64 and x86_64 and mccl would rather be portable than clever.
 */
mcclResult_t mcclCommInitRankFromId(mcclComm_t* comm, int nranks,
                                    const char* commIdBytes, int rank);

static inline mcclResult_t mcclCommInitRank(mcclComm_t* comm, int nranks,
                                            mcclUniqueId commId, int rank) {
    return mcclCommInitRankFromId(comm, nranks, commId.internal, rank);
}

/* Tears down the transport and frees the handle. NULL is a no-op. */
mcclResult_t mcclCommDestroy(mcclComm_t comm);

mcclResult_t mcclCommCount(mcclComm_t comm, int* count);
mcclResult_t mcclCommUserRank(mcclComm_t comm, int* rank);

/* Fills `version` with MCCL_VERSION_CODE as built into the library. */
mcclResult_t mcclGetVersion(int* version);

/* The algorithm the planner would pick for a message of `bytes`, as text
 * ("ring(0->1->2)", "tree(root: 0, …)", "hierarchical(…)"). mccl-specific:
 * NCCL does not let you ask. Writes at most `bufferSize` bytes including NUL. */
mcclResult_t mcclCommPlanDescription(mcclComm_t comm, size_t bytes,
                                     char* buffer, size_t bufferSize);

/* ------------------------------------------------------------ collectives */

/*
 * In-place is expressed the NCCL way: pass the same pointer as sendbuff and
 * recvbuff. Passing different pointers copies first, so `sendbuff` is never
 * modified.
 */

mcclResult_t mcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           mcclDataType_t datatype, mcclRedOp_t op,
                           mcclComm_t comm, mcclStream_t stream);

/*
 * As mcclAllReduce, with in-flight compression.
 * `blockSize` is used by mcclCompressInt8Blockwise (0 selects the default 256).
 * `fraction`  is used by mcclCompressTopK.
 */
mcclResult_t mcclAllReduceCompressed(const void* sendbuff, void* recvbuff, size_t count,
                                     mcclDataType_t datatype, mcclRedOp_t op,
                                     mcclCompression_t compression,
                                     int32_t blockSize, double fraction,
                                     mcclComm_t comm, mcclStream_t stream);

/* recvbuff holds nranks * sendcount elements, rank r's contribution at slot r. */
mcclResult_t mcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           mcclDataType_t datatype,
                           mcclComm_t comm, mcclStream_t stream);

mcclResult_t mcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           mcclDataType_t datatype, int root,
                           mcclComm_t comm, mcclStream_t stream);

/* NCCL's in-place broadcast spelling. */
static inline mcclResult_t mcclBcast(void* buff, size_t count, mcclDataType_t datatype,
                                     int root, mcclComm_t comm, mcclStream_t stream) {
    return mcclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

/* sendbuff holds nranks * recvcount elements; this rank keeps segment `rank`. */
mcclResult_t mcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               mcclDataType_t datatype, mcclRedOp_t op,
                               mcclComm_t comm, mcclStream_t stream);

/* Result lands on `root` only; other ranks' recvbuff is scratch. */
mcclResult_t mcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        mcclDataType_t datatype, mcclRedOp_t op, int root,
                        mcclComm_t comm, mcclStream_t stream);

/* ------------------------------------------------- non-blocking collectives */

/*
 * An issued-but-not-finished collective.
 *
 * WHY A REQUEST AND NOT A STREAM. NCCL's asynchrony is the CUDA stream's:
 * ncclAllReduce enqueues onto a stream the caller already owns and the caller
 * synchronises that stream. mccl has no device queue to borrow, and its
 * mcclStream_t already means something else — the sequence a top-k residual
 * belongs to. Giving that handle a second job, and simultaneously turning every
 * existing blocking call non-blocking, would break callers that read their
 * buffer on return. So mccl takes MPI's shape instead: issue returns a handle,
 * and the handle is waited or tested. mcclAllReduce still blocks and still
 * means what it always meant.
 *
 * WHAT THE OVERLAP BUYS. Collectives on one communicator run on that
 * communicator's serial queue, so two outstanding requests execute one after
 * the other in the order they were issued — and that ordering is required, not
 * a limitation: ranks must run collectives in the same order or they deadlock,
 * and issue order is the only order every rank agrees on. What overlaps is the
 * caller's own work with the collective's socket traffic, which on a Mac
 * cluster is where the time goes.
 */
typedef struct mcclRequest* mcclRequest_t;

/*
 * Issues an all-reduce and returns immediately. `*request` receives the handle.
 *
 * The buffers belong to the collective until mcclRequestWait returns: do not
 * read, write, or free them before then. In-place is expressed as it is
 * everywhere else — pass the same pointer as sendbuff and recvbuff.
 *
 * Every request must eventually be waited. There is no free-without-wait, on
 * purpose: abandoning a request would leave a collective writing into memory
 * the caller believes it has back.
 */
mcclResult_t mcclAllReduceAsync(const void* sendbuff, void* recvbuff, size_t count,
                                mcclDataType_t datatype, mcclRedOp_t op,
                                mcclComm_t comm, mcclStream_t stream,
                                mcclRequest_t* request);

/* As mcclAllReduceAsync, with in-flight compression. Arguments as
 * mcclAllReduceCompressed. */
mcclResult_t mcclAllReduceCompressedAsync(const void* sendbuff, void* recvbuff, size_t count,
                                          mcclDataType_t datatype, mcclRedOp_t op,
                                          mcclCompression_t compression,
                                          int32_t blockSize, double fraction,
                                          mcclComm_t comm, mcclStream_t stream,
                                          mcclRequest_t* request);

/*
 * Blocks until `request` completes, then frees it. The handle is invalid on
 * return, whether the result was success or failure. Returns the collective's
 * own result code; mcclGetLastError on the communicator has the detail.
 */
mcclResult_t mcclRequestWait(mcclRequest_t request);

/*
 * Reports whether `request` has completed, without blocking and without
 * freeing it. `*done` is set to 1 or 0. A request that tests done must still be
 * waited — that is what releases the handle.
 *
 * The return value describes the *test*, not the collective: it is mcclSuccess
 * whenever the question could be answered. A failed collective reports its
 * failure from mcclRequestWait.
 */
mcclResult_t mcclRequestTest(mcclRequest_t request, int* done);

#ifdef __cplusplus
}
#endif

#endif /* MCCL_H */
