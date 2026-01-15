#include "quickmp.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <dlfcn.h>
#include <veda.h>

#define VEDA_CHECK(err) veda_check(err, __FILE__, __LINE__)

namespace {

void veda_check(VEDAresult err, const char *file, int line) {
    if (err != VEDA_SUCCESS) {
        const char *name, *str;
        vedaGetErrorName(err, &name);
        vedaGetErrorString(err, &str);
        fprintf(stderr, "%s: %s @ %s:%i\n", name, str, file, line);
        exit(1);
    }
}

std::string get_kernel_lib_path() {
    Dl_info info;
    dladdr(reinterpret_cast<void *>(&get_kernel_lib_path), &info);

    std::filesystem::path path = info.dli_fname;
    path.replace_filename("libquickmp-device.vso");

    return path.string();
}

// Memory pool for VE device memory (single global pool with mutex)
// Note: Per-stream pools were tried but performed worse due to VEDA internal
// contention when multiple threads call vedaMemAlloc simultaneously.
class MemoryPool {
public:
    VEDAdeviceptr alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &free_list = free_blocks_[size];
        if (!free_list.empty()) {
            VEDAdeviceptr ptr = free_list.back();
            free_list.pop_back();
            return ptr;
        }
        VEDAdeviceptr ptr;
        VEDA_CHECK(vedaMemAlloc(&ptr, size));
        allocated_sizes_[ptr] = size;
        return ptr;
    }

    void free(VEDAdeviceptr ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t size = allocated_sizes_[ptr];
        free_blocks_[size].push_back(ptr);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[size, ptrs] : free_blocks_) {
            for (auto ptr : ptrs) {
                vedaMemFree(ptr);
            }
        }
        free_blocks_.clear();
        allocated_sizes_.clear();
    }

private:
    std::mutex mutex_;
    std::map<size_t, std::vector<VEDAdeviceptr>> free_blocks_;
    std::map<VEDAdeviceptr, size_t> allocated_sizes_;
};

VEDAcontext g_ctx;
VEDAmodule g_mod;
VEDAfunction g_selfjoin;
VEDAfunction g_abjoin;
VEDAfunction g_compute_mean_std;
VEDAfunction g_sliding_dot_product;
VEDAfunction g_sleep;
MemoryPool g_pool;

} // anonymous namespace

namespace quickmp {

void initialize(int device) {
    VEDA_CHECK(vedaInit(0));
    VEDA_CHECK(vedaCtxCreate(&g_ctx, VEDA_CONTEXT_MODE_SCALAR, device));
    VEDA_CHECK(vedaModuleLoad(&g_mod, get_kernel_lib_path().c_str()));
    VEDA_CHECK(vedaModuleGetFunction(&g_selfjoin, g_mod, "selfjoin_kernel"));
    VEDA_CHECK(vedaModuleGetFunction(&g_abjoin, g_mod, "abjoin_kernel"));
    VEDA_CHECK(vedaModuleGetFunction(&g_compute_mean_std, g_mod, "compute_mean_std_kernel"));
    VEDA_CHECK(vedaModuleGetFunction(&g_sliding_dot_product, g_mod, "sliding_dot_product_kernel"));
    VEDA_CHECK(vedaModuleGetFunction(&g_sleep, g_mod, "sleep_kernel"));
}

void finalize() {
    g_pool.clear();
    VEDA_CHECK(vedaExit());
}

void sliding_dot_product(const double *T, const double *Q, double *QT,
                         size_t n, size_t m, int stream) {
    VEDA_CHECK(vedaCtxSetCurrent(g_ctx));
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = g_pool.alloc(n * sizeof(double));
    VEDAdeviceptr Q_ptr = g_pool.alloc(m * sizeof(double));
    VEDAdeviceptr QT_ptr = g_pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, Q_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, QT_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyHtoDAsync(Q_ptr, Q, m * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_sliding_dot_product, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(QT, QT_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    g_pool.free(T_ptr);
    g_pool.free(Q_ptr);
    g_pool.free(QT_ptr);
}

void compute_mean_std(const double *T, double *mu, double *sigma,
                      size_t n, size_t m, int stream) {
    VEDA_CHECK(vedaCtxSetCurrent(g_ctx));
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = g_pool.alloc(n * sizeof(double));
    VEDAdeviceptr mu_ptr = g_pool.alloc((n - m + 1) * sizeof(double));
    VEDAdeviceptr sigma_ptr = g_pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, mu_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, sigma_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_compute_mean_std, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(mu, mu_ptr, (n - m + 1) * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyDtoHAsync(sigma, sigma_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    g_pool.free(T_ptr);
    g_pool.free(mu_ptr);
    g_pool.free(sigma_ptr);
}

void selfjoin(const double *T, double *P, size_t n, size_t m, int stream) {
    VEDA_CHECK(vedaCtxSetCurrent(g_ctx));
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = g_pool.alloc(n * sizeof(double));
    VEDAdeviceptr P_ptr = g_pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 2, n));
    VEDA_CHECK(vedaArgsSetU64(args, 3, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_selfjoin, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    g_pool.free(T_ptr);
    g_pool.free(P_ptr);
}

void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, int stream) {
    VEDA_CHECK(vedaCtxSetCurrent(g_ctx));
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T1_ptr = g_pool.alloc(n1 * sizeof(double));
    VEDAdeviceptr T2_ptr = g_pool.alloc(n2 * sizeof(double));
    VEDAdeviceptr P_ptr = g_pool.alloc((n1 - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T1_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, T2_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n1));
    VEDA_CHECK(vedaArgsSetU64(args, 4, n2));
    VEDA_CHECK(vedaArgsSetU64(args, 5, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T1_ptr, T1, n1 * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyHtoDAsync(T2_ptr, T2, n2 * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_abjoin, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n1 - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    g_pool.free(T1_ptr);
    g_pool.free(T2_ptr);
    g_pool.free(P_ptr);
}

void sleep_us(uint64_t microseconds, int stream) {
    VEDA_CHECK(vedaCtxSetCurrent(g_ctx));
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetU64(args, 0, microseconds));

    VEDA_CHECK(vedaLaunchKernelEx(g_sleep, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaStreamSynchronize(veda_stream));
}

} // namespace quickmp
