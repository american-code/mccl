/*
 * mccl_smoke.c — exercises the whole C ABI from C, exactly as a non-Swift
 * runtime would. Compiled with `cc` against the built libmccl.dylib by
 * CShimTests; run with no arguments, exits 0 on success.
 *
 * It is deliberately a C file rather than a Swift test calling @_cdecl symbols:
 * the point is to prove that mccl.h is valid C, that the declared signatures
 * match the exported ones, and that the dylib links and runs standalone.
 *
 * The four ranks are pthreads in one process. That is the honest in-process
 * analogue of four machines: each rank binds its own listener on an ephemeral
 * port and finds the others through the rendezvous, over real loopback sockets.
 */

#include <mccl.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define RANKS 4
#define COUNT 1024

static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            __sync_fetch_and_add(&failures, 1);                                \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr, comm)                                                   \
    do {                                                                       \
        mcclResult_t _r = (expr);                                              \
        CHECK(_r == mcclSuccess, "%s -> %s (%s)", #expr,                       \
              mcclGetErrorString(_r), mcclGetLastError(comm));                 \
    } while (0)

typedef struct {
    int rank;
    mcclUniqueId id;
} rank_args;

/* rank r contributes (r + 1) * (i % 7 + 1); the sum over four ranks is
 * 10 * (i % 7 + 1), exactly representable everywhere. */
static float contribution(int rank, int index) {
    return (float)((rank + 1) * ((index % 7) + 1));
}

static float expected_sum(int index) { return (float)(10 * ((index % 7) + 1)); }

static void* rank_main(void* raw) {
    rank_args* args = (rank_args*)raw;
    mcclComm_t comm = NULL;

    /* NCCL-idiom bring-up: one token, one call, by value. */
    mcclResult_t init = mcclCommInitRank(&comm, RANKS, args->id, args->rank);
    if (init != mcclSuccess) {
        fprintf(stderr, "FAIL rank %d init: %s (%s)\n", args->rank,
                mcclGetErrorString(init), mcclGetLastError(NULL));
        __sync_fetch_and_add(&failures, 1);
        return NULL;
    }

    int nranks = 0, myrank = -1;
    CHECK_OK(mcclCommCount(comm, &nranks), comm);
    CHECK_OK(mcclCommUserRank(comm, &myrank), comm);
    CHECK(nranks == RANKS, "rank %d: count %d", args->rank, nranks);
    CHECK(myrank == args->rank, "rank reported %d, expected %d", myrank, args->rank);

    float* send = malloc(sizeof(float) * COUNT);
    float* recv = malloc(sizeof(float) * COUNT);
    float* gathered = malloc(sizeof(float) * COUNT * RANKS);
    CHECK(send && recv && gathered, "allocation failed");

    /* ---- all-reduce, out of place ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = contribution(args->rank, i);
    CHECK_OK(mcclAllReduce(send, recv, COUNT, mcclFloat32, mcclSum, comm, NULL), comm);
    for (int i = 0; i < COUNT; ++i) {
        CHECK(fabsf(recv[i] - expected_sum(i)) < 1e-3f,
              "allreduce[%d] = %f, expected %f", i, recv[i], expected_sum(i));
        CHECK(send[i] == contribution(args->rank, i),
              "allreduce must not modify sendbuff at %d", i);
    }

    /* ---- all-reduce, in place, averaged ---- */
    for (int i = 0; i < COUNT; ++i) recv[i] = contribution(args->rank, i);
    CHECK_OK(mcclAllReduce(recv, recv, COUNT, mcclFloat32, mcclAvg, comm, NULL), comm);
    for (int i = 0; i < COUNT; ++i) {
        float want = expected_sum(i) / (float)RANKS;
        CHECK(fabsf(recv[i] - want) < 1e-3f, "avg[%d] = %f, expected %f", i, recv[i], want);
    }

    /* ---- compressed all-reduce: fp16 downcast on the wire ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = contribution(args->rank, i);
    CHECK_OK(mcclAllReduceCompressed(send, recv, COUNT, mcclFloat32, mcclSum,
                                     mcclCompressDowncast, 0, 0.0, comm, NULL), comm);
    for (int i = 0; i < COUNT; ++i) {
        CHECK(fabsf(recv[i] - expected_sum(i)) < 0.5f,
              "downcast allreduce[%d] = %f, expected %f", i, recv[i], expected_sum(i));
    }

    /* ---- compressed all-reduce: top-k at fraction 1 is exact ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = contribution(args->rank, i);
    CHECK_OK(mcclAllReduceCompressed(send, recv, COUNT, mcclFloat32, mcclSum,
                                     mcclCompressTopK, 0, 1.0, comm,
                                     mcclStreamFromId(7)), comm);
    for (int i = 0; i < COUNT; ++i) {
        CHECK(fabsf(recv[i] - expected_sum(i)) < 1e-3f,
              "topk allreduce[%d] = %f, expected %f", i, recv[i], expected_sum(i));
    }

    /* ---- all-gather ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = (float)(args->rank * 1000 + (i % 10));
    CHECK_OK(mcclAllGather(send, gathered, COUNT, mcclFloat32, comm, NULL), comm);
    for (int r = 0; r < RANKS; ++r) {
        for (int i = 0; i < COUNT; ++i) {
            float want = (float)(r * 1000 + (i % 10));
            CHECK(gathered[r * COUNT + i] == want,
                  "allgather slot %d [%d] = %f, expected %f", r, i,
                  gathered[r * COUNT + i], want);
        }
    }

    /* ---- broadcast from rank 2 ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = (args->rank == 2) ? (float)(i % 13) : -1.0f;
    CHECK_OK(mcclBroadcast(send, recv, COUNT, mcclFloat32, 2, comm, NULL), comm);
    for (int i = 0; i < COUNT; ++i) {
        CHECK(recv[i] == (float)(i % 13), "broadcast[%d] = %f, expected %d", i, recv[i], i % 13);
    }

    /* ---- in-place broadcast, NCCL's mcclBcast spelling ---- */
    for (int i = 0; i < COUNT; ++i) recv[i] = (args->rank == 0) ? (float)(i % 5) : -1.0f;
    CHECK_OK(mcclBcast(recv, COUNT, mcclFloat32, 0, comm, NULL), comm);
    for (int i = 0; i < COUNT; ++i) {
        CHECK(recv[i] == (float)(i % 5), "bcast[%d] = %f", i, recv[i]);
    }

    /* ---- reduce-scatter: COUNT elements in, COUNT/RANKS out ---- */
    {
        const int segment = COUNT / RANKS;
        for (int i = 0; i < COUNT; ++i) send[i] = contribution(args->rank, i);
        CHECK_OK(mcclReduceScatter(send, recv, segment, mcclFloat32, mcclSum, comm, NULL), comm);
        for (int i = 0; i < segment; ++i) {
            float want = expected_sum(args->rank * segment + i);
            CHECK(fabsf(recv[i] - want) < 1e-3f,
                  "reducescatter[%d] = %f, expected %f", i, recv[i], want);
        }
    }

    /* ---- reduce onto rank 1 ---- */
    for (int i = 0; i < COUNT; ++i) send[i] = contribution(args->rank, i);
    CHECK_OK(mcclReduce(send, recv, COUNT, mcclFloat32, mcclSum, 1, comm, NULL), comm);
    if (args->rank == 1) {
        for (int i = 0; i < COUNT; ++i) {
            CHECK(fabsf(recv[i] - expected_sum(i)) < 1e-3f,
                  "reduce[%d] = %f, expected %f", i, recv[i], expected_sum(i));
        }
    }

    /* ---- mccl-specific: ask which algorithm the planner chose ---- */
    {
        char plan[256] = {0};
        CHECK_OK(mcclCommPlanDescription(comm, COUNT * sizeof(float), plan, sizeof(plan)), comm);
        CHECK(strlen(plan) > 0, "plan description was empty");
        if (args->rank == 0) printf("plan: %s\n", plan);
    }

    /* ---- error paths return codes, never crash ---- */
    {
        mcclResult_t r = mcclAllReduce(send, recv, COUNT, (mcclDataType_t)99,
                                       mcclSum, comm, NULL);
        CHECK(r == mcclInvalidArgument, "bad dtype gave %s", mcclGetErrorString(r));
        CHECK(strlen(mcclGetLastError(comm)) > 0, "a failure should leave detail behind");

        r = mcclAllReduce(send, NULL, COUNT, mcclFloat32, mcclSum, comm, NULL);
        CHECK(r == mcclInvalidArgument, "NULL recvbuff gave %s", mcclGetErrorString(r));

        r = mcclBroadcast(send, recv, COUNT, mcclFloat32, RANKS + 5, comm, NULL);
        CHECK(r == mcclRankOutOfRange, "root out of range gave %s", mcclGetErrorString(r));

        /* Top-k on a collective that cannot account for the dropped mass. */
        r = mcclAllReduceCompressed(send, recv, COUNT, mcclInt32, mcclSum,
                                    mcclCompressTopK, 0, 0.1, comm, NULL);
        CHECK(r == mcclUnsupportedCompression, "topk on int32 gave %s", mcclGetErrorString(r));
    }

    free(send);
    free(recv);
    free(gathered);
    CHECK_OK(mcclCommDestroy(comm), NULL);
    return NULL;
}

static void test_standalone(void) {
    int version = 0;
    CHECK(mcclGetVersion(&version) == mcclSuccess, "mcclGetVersion failed");
    CHECK(version == MCCL_VERSION_CODE, "header says %d, library says %d",
          MCCL_VERSION_CODE, version);

    CHECK(mcclGetVersion(NULL) == mcclInvalidArgument, "NULL out-parameter must be rejected");

    /* Every result code names itself; unknown codes do not crash. */
    for (int i = 0; i < mcclNumResults; ++i) {
        const char* name = mcclGetErrorString((mcclResult_t)i);
        CHECK(name != NULL && strlen(name) > 0, "result %d has no name", i);
    }
    CHECK(mcclGetErrorString((mcclResult_t)4242) != NULL, "unknown result must still return a string");
    CHECK(strcmp(mcclGetErrorString(mcclSuccess), "mcclSuccess") == 0, "mcclSuccess misnamed");

    /* Unique id text round-trip. */
    mcclUniqueId id, restored;
    CHECK(mcclGetUniqueId(&id) == mcclSuccess, "mcclGetUniqueId failed: %s",
          mcclGetLastError(NULL));
    char text[MCCL_UNIQUE_ID_BYTES] = {0};
    CHECK(mcclUniqueIdToString(&id, text, sizeof(text)) == mcclSuccess, "id -> text failed");
    CHECK(strncmp(text, "mccl1:", 6) == 0, "unexpected id text '%s'", text);
    CHECK(mcclUniqueIdFromString(text, &restored) == mcclSuccess, "text -> id failed");
    CHECK(memcmp(id.internal, restored.internal, strlen(text) + 1) == 0, "id did not round-trip");
    CHECK(mcclUniqueIdFromString("not-an-id", &restored) == mcclInvalidArgument,
          "garbage should be rejected");
    /* This token is not going to be used; hand its listener back. */
    CHECK(mcclUniqueIdDiscard(&id) == mcclSuccess, "discard failed");

    /* NULL communicator is an error, not a crash. */
    int count = 0;
    CHECK(mcclCommCount(NULL, &count) == mcclInvalidArgument, "NULL comm must be rejected");
    CHECK(mcclCommDestroy(NULL) == mcclSuccess, "destroying NULL is a no-op");
    CHECK(mcclGetLastError(NULL) != NULL, "global error slot must never be NULL");
}

int main(void) {
    test_standalone();

    rank_args args[RANKS];
    pthread_t threads[RANKS];

    mcclResult_t r = mcclGetUniqueId(&args[0].id);
    if (r != mcclSuccess) {
        fprintf(stderr, "FAIL mcclGetUniqueId: %s (%s)\n",
                mcclGetErrorString(r), mcclGetLastError(NULL));
        return 1;
    }
    for (int i = 0; i < RANKS; ++i) {
        args[i].rank = i;
        memcpy(&args[i].id, &args[0].id, sizeof(mcclUniqueId));
    }
    for (int i = 0; i < RANKS; ++i) {
        if (pthread_create(&threads[i], NULL, rank_main, &args[i]) != 0) {
            fprintf(stderr, "FAIL pthread_create for rank %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < RANKS; ++i) pthread_join(threads[i], NULL);

    if (failures == 0) {
        printf("mccl C ABI smoke test: OK (%d ranks, %d elements)\n", RANKS, COUNT);
        return 0;
    }
    fprintf(stderr, "mccl C ABI smoke test: %d failure(s)\n", failures);
    return 1;
}
