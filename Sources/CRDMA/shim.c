/*
 * shim.c — see crdma.h for the linkage strategy this file implements.
 *
 * Two compilation paths:
 *
 *   * With <infiniband/verbs.h> (macOS 26.2+ SDK): real bodies, real struct
 *     layouts, entry points bound with dlopen/dlsym at first use.
 *   * Without it (any older SDK, including CI): every entry point is a stub
 *     returning MCCL_RDMA_NO_HEADER, and `mccl_rdma_load_error` says so. The
 *     package still builds and every non-RDMA test still runs.
 */

#include "include/crdma.h"

#include <string.h>

/*
 * `MCCL_RDMA_FORCE_STUB` compiles the no-header path on a machine that does
 * have the header. Without it that path is reachable only on an SDK older than
 * 26.2 — which is what CI runs on, and what nobody here can run the suite
 * against. It exists so `swift test -Xcc -DMCCL_RDMA_FORCE_STUB` checks the
 * branch CI will actually take.
 */
#if __has_include(<infiniband/verbs.h>) && !defined(MCCL_RDMA_FORCE_STUB)
#define MCCL_HAVE_VERBS_HEADER 1
#else
#define MCCL_HAVE_VERBS_HEADER 0
#endif

int mccl_rdma_built_with_headers(void) { return MCCL_HAVE_VERBS_HEADER; }

#if MCCL_HAVE_VERBS_HEADER

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <infiniband/verbs.h>

/*
 * Cross-check every constant mccl hard-codes against the SDK it was compiled
 * with. These are the numbers the Swift state machine builds its attribute
 * masks from; if a future SDK renumbers one, this fails the build rather than
 * silently mis-programming a queue pair.
 */
_Static_assert(MCCL_IBV_QPT_UC             == IBV_QPT_UC,             "IBV_QPT_UC");
_Static_assert(MCCL_IBV_MTU_4096           == IBV_MTU_4096,           "IBV_MTU_4096");
_Static_assert(MCCL_IBV_ACCESS_LOCAL_WRITE == IBV_ACCESS_LOCAL_WRITE, "IBV_ACCESS_LOCAL_WRITE");
_Static_assert(MCCL_IBV_WR_SEND            == IBV_WR_SEND,            "IBV_WR_SEND");
_Static_assert(MCCL_IBV_SEND_SIGNALED      == IBV_SEND_SIGNALED,      "IBV_SEND_SIGNALED");
_Static_assert(MCCL_IBV_WC_SUCCESS         == IBV_WC_SUCCESS,         "IBV_WC_SUCCESS");
_Static_assert(MCCL_IBV_PORT_ACTIVE        == IBV_PORT_ACTIVE,        "IBV_PORT_ACTIVE");
_Static_assert(MCCL_IBV_QPS_RESET          == IBV_QPS_RESET,          "IBV_QPS_RESET");
_Static_assert(MCCL_IBV_QPS_INIT           == IBV_QPS_INIT,           "IBV_QPS_INIT");
_Static_assert(MCCL_IBV_QPS_RTR            == IBV_QPS_RTR,            "IBV_QPS_RTR");
_Static_assert(MCCL_IBV_QPS_RTS            == IBV_QPS_RTS,            "IBV_QPS_RTS");
_Static_assert(MCCL_IBV_QPS_ERR            == IBV_QPS_ERR,            "IBV_QPS_ERR");
_Static_assert(MCCL_IBV_QP_STATE           == IBV_QP_STATE,           "IBV_QP_STATE");
_Static_assert(MCCL_IBV_QP_ACCESS_FLAGS    == IBV_QP_ACCESS_FLAGS,    "IBV_QP_ACCESS_FLAGS");
_Static_assert(MCCL_IBV_QP_PKEY_INDEX      == IBV_QP_PKEY_INDEX,      "IBV_QP_PKEY_INDEX");
_Static_assert(MCCL_IBV_QP_PORT            == IBV_QP_PORT,            "IBV_QP_PORT");
_Static_assert(MCCL_IBV_QP_AV              == IBV_QP_AV,              "IBV_QP_AV");
_Static_assert(MCCL_IBV_QP_PATH_MTU        == IBV_QP_PATH_MTU,        "IBV_QP_PATH_MTU");
_Static_assert(MCCL_IBV_QP_RQ_PSN          == IBV_QP_RQ_PSN,          "IBV_QP_RQ_PSN");
_Static_assert(MCCL_IBV_QP_SQ_PSN          == IBV_QP_SQ_PSN,          "IBV_QP_SQ_PSN");
_Static_assert(MCCL_IBV_QP_DEST_QPN        == IBV_QP_DEST_QPN,        "IBV_QP_DEST_QPN");

/* ------------------------------------------------------------------------ */
/* Dynamic binding                                                          */
/* ------------------------------------------------------------------------ */

/*
 * The exported entry points. `ibv_post_send`, `ibv_post_recv` and
 * `ibv_poll_cq` are deliberately absent: they are `static inline` in the header
 * and dispatch through `qp->context->ops`, so they need no symbol and cost no
 * indirection beyond the vtable call a normally-linked caller makes.
 *
 * `ibv_reg_mr` is a macro in the header (it forwards to `ibv_reg_mr_iova2` for
 * access flags that need it). We bind the underlying symbol and call it
 * directly, which is correct here because the only access flag mccl ever passes
 * is IBV_ACCESS_LOCAL_WRITE — the only one TN3205 permits.
 */
struct verbs_api {
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *,
                                struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    const char *(*wc_status_str)(enum ibv_wc_status);
};

static struct verbs_api api;
static int load_status = MCCL_RDMA_NO_LIBRARY;
static char load_error[256];
static pthread_once_t load_once = PTHREAD_ONCE_INIT;

static void *bind_symbol(void *handle, const char *name, int *failed) {
    void *symbol = dlsym(handle, name);
    if (symbol == NULL && !*failed) {
        *failed = 1;
        snprintf(load_error, sizeof(load_error),
                 "librdma.dylib is loaded but does not export %s", name);
    }
    return symbol;
}

static void load_library(void) {
    /*
     * RTLD_LOCAL so we never leak these symbols into a host process that has
     * its own libibverbs (a Linux-style stack under Rosetta, say).
     */
    void *handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
        const char *why = dlerror();
        snprintf(load_error, sizeof(load_error),
                 "/usr/lib/librdma.dylib could not be loaded (%s); "
                 "RDMA over Thunderbolt needs macOS 26.2 or later",
                 why ? why : "no detail from dlopen");
        load_status = MCCL_RDMA_NO_LIBRARY;
        return;
    }

    int failed = 0;
#define BIND(field, name) \
    api.field = (__typeof__(api.field))bind_symbol(handle, name, &failed)

    BIND(get_device_list,  "ibv_get_device_list");
    BIND(free_device_list, "ibv_free_device_list");
    BIND(get_device_name,  "ibv_get_device_name");
    BIND(open_device,      "ibv_open_device");
    BIND(close_device,     "ibv_close_device");
    BIND(query_port,       "ibv_query_port");
    BIND(query_gid,        "ibv_query_gid");
    BIND(alloc_pd,         "ibv_alloc_pd");
    BIND(dealloc_pd,       "ibv_dealloc_pd");
    BIND(reg_mr,           "ibv_reg_mr");
    BIND(dereg_mr,         "ibv_dereg_mr");
    BIND(create_cq,        "ibv_create_cq");
    BIND(destroy_cq,       "ibv_destroy_cq");
    BIND(create_qp,        "ibv_create_qp");
    BIND(destroy_qp,       "ibv_destroy_qp");
    BIND(modify_qp,        "ibv_modify_qp");
    BIND(wc_status_str,    "ibv_wc_status_str");
#undef BIND

    if (failed) {
        load_status = MCCL_RDMA_NO_SYMBOL;
        dlclose(handle);
        memset(&api, 0, sizeof(api));
        return;
    }
    /* Intentionally never dlclose'd on success: the handles handed out below
     * stay valid for the process lifetime. */
    load_status = MCCL_RDMA_OK;
    load_error[0] = '\0';
}

int mccl_rdma_load(void) {
    pthread_once(&load_once, load_library);
    return load_status;
}

const char *mccl_rdma_load_error(void) {
    if (mccl_rdma_load() == MCCL_RDMA_OK) return NULL;
    return load_error;
}

#define REQUIRE_LOADED(failure) \
    do { if (mccl_rdma_load() != MCCL_RDMA_OK) return (failure); } while (0)

/* ------------------------------------------------------------------------ */
/* Devices                                                                  */
/* ------------------------------------------------------------------------ */

int mccl_rdma_device_count(void) {
    REQUIRE_LOADED(load_status);
    int count = 0;
    struct ibv_device **list = api.get_device_list(&count);
    if (list == NULL) return MCCL_RDMA_FAILED;
    api.free_device_list(list);
    return count;
}

int mccl_rdma_device_name(int index, char *out, size_t capacity) {
    REQUIRE_LOADED(load_status);
    if (out == NULL || capacity == 0) return MCCL_RDMA_FAILED;
    int count = 0;
    struct ibv_device **list = api.get_device_list(&count);
    if (list == NULL) return MCCL_RDMA_FAILED;
    int result = MCCL_RDMA_FAILED;
    if (index >= 0 && index < count) {
        const char *name = api.get_device_name(list[index]);
        if (name != NULL) {
            strlcpy(out, name, capacity);
            result = MCCL_RDMA_OK;
        }
    }
    api.free_device_list(list);
    return result;
}

mccl_rdma_context mccl_rdma_open_device(int index) {
    REQUIRE_LOADED(NULL);
    int count = 0;
    struct ibv_device **list = api.get_device_list(&count);
    if (list == NULL) return NULL;
    struct ibv_context *context = NULL;
    if (index >= 0 && index < count) context = api.open_device(list[index]);
    api.free_device_list(list);
    return context;
}

void mccl_rdma_close_device(mccl_rdma_context context) {
    if (context == NULL || mccl_rdma_load() != MCCL_RDMA_OK) return;
    api.close_device((struct ibv_context *)context);
}

int mccl_rdma_query_port(mccl_rdma_context context, uint8_t port,
                         uint16_t *out_lid, uint32_t *out_state) {
    REQUIRE_LOADED(load_status);
    if (context == NULL) return MCCL_RDMA_FAILED;
    struct ibv_port_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    int rc = api.query_port((struct ibv_context *)context, port, &attributes);
    if (rc != 0) return rc > 0 ? -rc : rc;
    if (out_lid != NULL) *out_lid = attributes.lid;
    if (out_state != NULL) *out_state = (uint32_t)attributes.state;
    return MCCL_RDMA_OK;
}

int mccl_rdma_query_gid(mccl_rdma_context context, uint8_t port, int index,
                        uint8_t out_gid[16]) {
    REQUIRE_LOADED(load_status);
    if (context == NULL || out_gid == NULL) return MCCL_RDMA_FAILED;
    union ibv_gid gid;
    memset(&gid, 0, sizeof(gid));
    int rc = api.query_gid((struct ibv_context *)context, port, index, &gid);
    if (rc != 0) return rc > 0 ? -rc : rc;
    memcpy(out_gid, gid.raw, 16);
    return MCCL_RDMA_OK;
}

/* ------------------------------------------------------------------------ */
/* Memory                                                                   */
/* ------------------------------------------------------------------------ */

mccl_rdma_pd mccl_rdma_alloc_pd(mccl_rdma_context context) {
    REQUIRE_LOADED(NULL);
    if (context == NULL) return NULL;
    return api.alloc_pd((struct ibv_context *)context);
}

void mccl_rdma_dealloc_pd(mccl_rdma_pd pd) {
    if (pd == NULL || mccl_rdma_load() != MCCL_RDMA_OK) return;
    api.dealloc_pd((struct ibv_pd *)pd);
}

mccl_rdma_mr mccl_rdma_reg_mr(mccl_rdma_pd pd, void *address, size_t length, int access) {
    REQUIRE_LOADED(NULL);
    if (pd == NULL || address == NULL || length == 0) return NULL;
    return api.reg_mr((struct ibv_pd *)pd, address, length, access);
}

uint32_t mccl_rdma_mr_lkey(mccl_rdma_mr mr) {
    if (mr == NULL) return 0;
    return ((struct ibv_mr *)mr)->lkey;
}

void mccl_rdma_dereg_mr(mccl_rdma_mr mr) {
    if (mr == NULL || mccl_rdma_load() != MCCL_RDMA_OK) return;
    api.dereg_mr((struct ibv_mr *)mr);
}

/* ------------------------------------------------------------------------ */
/* Queues                                                                   */
/* ------------------------------------------------------------------------ */

mccl_rdma_cq mccl_rdma_create_cq(mccl_rdma_context context, int depth) {
    REQUIRE_LOADED(NULL);
    if (context == NULL) return NULL;
    return api.create_cq((struct ibv_context *)context, depth, NULL, NULL, 0);
}

void mccl_rdma_destroy_cq(mccl_rdma_cq cq) {
    if (cq == NULL || mccl_rdma_load() != MCCL_RDMA_OK) return;
    api.destroy_cq((struct ibv_cq *)cq);
}

mccl_rdma_qp mccl_rdma_create_qp(mccl_rdma_pd pd, mccl_rdma_cq cq,
                                 int max_send_wr, int max_recv_wr, int qp_type) {
    REQUIRE_LOADED(NULL);
    if (pd == NULL || cq == NULL) return NULL;
    struct ibv_qp_init_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.send_cq = (struct ibv_cq *)cq;
    attributes.recv_cq = (struct ibv_cq *)cq;
    attributes.cap.max_send_wr = (uint32_t)max_send_wr;
    attributes.cap.max_recv_wr = (uint32_t)max_recv_wr;
    /* TN3205's listing uses one scatter/gather entry per work request, and the
     * transport above never builds a multi-entry list. */
    attributes.cap.max_send_sge = 1;
    attributes.cap.max_recv_sge = 1;
    attributes.qp_type = (enum ibv_qp_type)qp_type;
    return api.create_qp((struct ibv_pd *)pd, &attributes);
}

uint32_t mccl_rdma_qp_num(mccl_rdma_qp qp) {
    if (qp == NULL) return 0;
    return ((struct ibv_qp *)qp)->qp_num;
}

void mccl_rdma_destroy_qp(mccl_rdma_qp qp) {
    if (qp == NULL || mccl_rdma_load() != MCCL_RDMA_OK) return;
    api.destroy_qp((struct ibv_qp *)qp);
}

int mccl_rdma_modify_qp_to_init(mccl_rdma_qp qp, uint8_t port_num,
                                uint16_t pkey_index, uint32_t access_flags,
                                int attr_mask) {
    REQUIRE_LOADED(load_status);
    if (qp == NULL) return MCCL_RDMA_FAILED;
    struct ibv_qp_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = pkey_index;
    attributes.port_num = port_num;
    attributes.qp_access_flags = access_flags;
    int rc = api.modify_qp((struct ibv_qp *)qp, &attributes, attr_mask);
    return rc == 0 ? MCCL_RDMA_OK : (rc > 0 ? -rc : rc);
}

int mccl_rdma_modify_qp_to_rtr(mccl_rdma_qp qp, uint32_t path_mtu,
                               uint32_t rq_psn, uint32_t dest_qp_num,
                               const uint8_t dgid[16], uint16_t dlid,
                               uint8_t sgid_index, uint8_t hop_limit,
                               uint8_t port_num, int attr_mask) {
    REQUIRE_LOADED(load_status);
    if (qp == NULL || dgid == NULL) return MCCL_RDMA_FAILED;
    struct ibv_qp_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = (enum ibv_mtu)path_mtu;
    attributes.rq_psn = rq_psn;
    attributes.dest_qp_num = dest_qp_num;
    attributes.ah_attr.dlid = dlid;
    attributes.ah_attr.sl = 0;
    attributes.ah_attr.src_path_bits = 0;
    attributes.ah_attr.port_num = port_num;
    /* Always global: a Thunderbolt peer is addressed by GID, not by LID. */
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.grh.hop_limit = hop_limit;
    attributes.ah_attr.grh.sgid_index = sgid_index;
    memcpy(attributes.ah_attr.grh.dgid.raw, dgid, 16);
    int rc = api.modify_qp((struct ibv_qp *)qp, &attributes, attr_mask);
    return rc == 0 ? MCCL_RDMA_OK : (rc > 0 ? -rc : rc);
}

int mccl_rdma_modify_qp_to_rts(mccl_rdma_qp qp, uint32_t sq_psn, int attr_mask) {
    REQUIRE_LOADED(load_status);
    if (qp == NULL) return MCCL_RDMA_FAILED;
    struct ibv_qp_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.sq_psn = sq_psn;
    int rc = api.modify_qp((struct ibv_qp *)qp, &attributes, attr_mask);
    return rc == 0 ? MCCL_RDMA_OK : (rc > 0 ? -rc : rc);
}

/* ------------------------------------------------------------------------ */
/* Data path — no dlsym on this path, see the note on `struct verbs_api`.    */
/* ------------------------------------------------------------------------ */

int mccl_rdma_post_send(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey, unsigned int send_flags,
                        int opcode) {
    REQUIRE_LOADED(load_status);
    if (qp == NULL) return MCCL_RDMA_FAILED;
    struct ibv_sge entry;
    memset(&entry, 0, sizeof(entry));
    entry.addr = address;
    entry.length = length;
    entry.lkey = lkey;

    struct ibv_send_wr request;
    memset(&request, 0, sizeof(request));
    request.wr_id = wr_id;
    request.sg_list = &entry;
    request.num_sge = 1;
    request.opcode = (enum ibv_wr_opcode)opcode;
    request.send_flags = send_flags;

    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send((struct ibv_qp *)qp, &request, &bad);
    return rc == 0 ? MCCL_RDMA_OK : (rc > 0 ? -rc : rc);
}

int mccl_rdma_post_recv(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey) {
    REQUIRE_LOADED(load_status);
    if (qp == NULL) return MCCL_RDMA_FAILED;
    struct ibv_sge entry;
    memset(&entry, 0, sizeof(entry));
    entry.addr = address;
    entry.length = length;
    entry.lkey = lkey;

    struct ibv_recv_wr request;
    memset(&request, 0, sizeof(request));
    request.wr_id = wr_id;
    request.sg_list = &entry;
    request.num_sge = 1;

    struct ibv_recv_wr *bad = NULL;
    int rc = ibv_post_recv((struct ibv_qp *)qp, &request, &bad);
    return rc == 0 ? MCCL_RDMA_OK : (rc > 0 ? -rc : rc);
}

int mccl_rdma_poll_cq(mccl_rdma_cq cq, int max_entries, mccl_rdma_completion *out) {
    REQUIRE_LOADED(load_status);
    if (cq == NULL || out == NULL || max_entries <= 0) return MCCL_RDMA_FAILED;
    /* 16 is the transport's batch size; anything larger is polled in passes. */
    enum { BATCH = 16 };
    struct ibv_wc completions[BATCH];
    int wanted = max_entries < BATCH ? max_entries : BATCH;
    int polled = ibv_poll_cq((struct ibv_cq *)cq, wanted, completions);
    if (polled < 0) return MCCL_RDMA_FAILED;
    for (int i = 0; i < polled; i++) {
        out[i].wr_id = completions[i].wr_id;
        out[i].status = (uint32_t)completions[i].status;
        out[i].opcode = (uint32_t)completions[i].opcode;
        out[i].byte_len = completions[i].byte_len;
    }
    return polled;
}

const char *mccl_rdma_status_string(uint32_t status) {
    if (mccl_rdma_load() != MCCL_RDMA_OK) return "unknown";
    const char *text = api.wc_status_str((enum ibv_wc_status)status);
    return text != NULL ? text : "unknown";
}

#else /* !MCCL_HAVE_VERBS_HEADER */

/*
 * Built against an SDK with no verbs header. Everything fails with one reason,
 * and `RDMATransport.isAvailable` reports it verbatim.
 */

static const char *const no_header_reason =
    "this build has no RDMA support: it was compiled against a macOS SDK "
    "without <infiniband/verbs.h>, which needs the macOS 26.2 SDK or later";

int mccl_rdma_load(void) { return MCCL_RDMA_NO_HEADER; }
const char *mccl_rdma_load_error(void) { return no_header_reason; }

int mccl_rdma_device_count(void) { return MCCL_RDMA_NO_HEADER; }
int mccl_rdma_device_name(int index, char *out, size_t capacity) {
    (void)index; (void)out; (void)capacity;
    return MCCL_RDMA_NO_HEADER;
}

mccl_rdma_context mccl_rdma_open_device(int index) { (void)index; return NULL; }
void mccl_rdma_close_device(mccl_rdma_context context) { (void)context; }

int mccl_rdma_query_port(mccl_rdma_context context, uint8_t port,
                         uint16_t *out_lid, uint32_t *out_state) {
    (void)context; (void)port; (void)out_lid; (void)out_state;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_query_gid(mccl_rdma_context context, uint8_t port, int index,
                        uint8_t out_gid[16]) {
    (void)context; (void)port; (void)index; (void)out_gid;
    return MCCL_RDMA_NO_HEADER;
}

mccl_rdma_pd mccl_rdma_alloc_pd(mccl_rdma_context context) { (void)context; return NULL; }
void mccl_rdma_dealloc_pd(mccl_rdma_pd pd) { (void)pd; }

mccl_rdma_mr mccl_rdma_reg_mr(mccl_rdma_pd pd, void *address, size_t length, int access) {
    (void)pd; (void)address; (void)length; (void)access;
    return NULL;
}
uint32_t mccl_rdma_mr_lkey(mccl_rdma_mr mr) { (void)mr; return 0; }
void mccl_rdma_dereg_mr(mccl_rdma_mr mr) { (void)mr; }

mccl_rdma_cq mccl_rdma_create_cq(mccl_rdma_context context, int depth) {
    (void)context; (void)depth;
    return NULL;
}
void mccl_rdma_destroy_cq(mccl_rdma_cq cq) { (void)cq; }

mccl_rdma_qp mccl_rdma_create_qp(mccl_rdma_pd pd, mccl_rdma_cq cq,
                                 int max_send_wr, int max_recv_wr, int qp_type) {
    (void)pd; (void)cq; (void)max_send_wr; (void)max_recv_wr; (void)qp_type;
    return NULL;
}
uint32_t mccl_rdma_qp_num(mccl_rdma_qp qp) { (void)qp; return 0; }
void mccl_rdma_destroy_qp(mccl_rdma_qp qp) { (void)qp; }

int mccl_rdma_modify_qp_to_init(mccl_rdma_qp qp, uint8_t port_num,
                                uint16_t pkey_index, uint32_t access_flags,
                                int attr_mask) {
    (void)qp; (void)port_num; (void)pkey_index; (void)access_flags; (void)attr_mask;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_modify_qp_to_rtr(mccl_rdma_qp qp, uint32_t path_mtu,
                               uint32_t rq_psn, uint32_t dest_qp_num,
                               const uint8_t dgid[16], uint16_t dlid,
                               uint8_t sgid_index, uint8_t hop_limit,
                               uint8_t port_num, int attr_mask) {
    (void)qp; (void)path_mtu; (void)rq_psn; (void)dest_qp_num; (void)dgid;
    (void)dlid; (void)sgid_index; (void)hop_limit; (void)port_num; (void)attr_mask;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_modify_qp_to_rts(mccl_rdma_qp qp, uint32_t sq_psn, int attr_mask) {
    (void)qp; (void)sq_psn; (void)attr_mask;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_post_send(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey, unsigned int send_flags,
                        int opcode) {
    (void)qp; (void)wr_id; (void)address; (void)length; (void)lkey;
    (void)send_flags; (void)opcode;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_post_recv(mccl_rdma_qp qp, uint64_t wr_id, uint64_t address,
                        uint32_t length, uint32_t lkey) {
    (void)qp; (void)wr_id; (void)address; (void)length; (void)lkey;
    return MCCL_RDMA_NO_HEADER;
}

int mccl_rdma_poll_cq(mccl_rdma_cq cq, int max_entries, mccl_rdma_completion *out) {
    (void)cq; (void)max_entries; (void)out;
    return MCCL_RDMA_NO_HEADER;
}

const char *mccl_rdma_status_string(uint32_t status) { (void)status; return "unknown"; }

#endif /* MCCL_HAVE_VERBS_HEADER */
