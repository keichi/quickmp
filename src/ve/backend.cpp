#include "quickmp.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
        throw std::runtime_error(std::string(name) + ": " + str + " @ " + file + ":" +
                                 std::to_string(line));
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

// Returns a pool allocation on scope exit, so exceptions do not leak device memory
class PoolPtr {
public:
    PoolPtr(MemoryPool &pool, size_t size) : pool_(pool), ptr_(pool.alloc(size)) {}
    ~PoolPtr() { pool_.free(ptr_); }
    PoolPtr(const PoolPtr &) = delete;
    PoolPtr &operator=(const PoolPtr &) = delete;
    operator VEDAdeviceptr() const { return ptr_; }

private:
    MemoryPool &pool_;
    VEDAdeviceptr ptr_;
};

// Device context holding all resources for a single VE device
struct DeviceContext {
    VEDAcontext ctx;
    VEDAmodule mod;
    VEDAfunction selfjoin;
    VEDAfunction selfjoin_batch;
    VEDAfunction abjoin;
    VEDAfunction compute_mean_std;
    VEDAfunction sliding_dot_product;
    MemoryPool pool;
};

std::vector<std::unique_ptr<DeviceContext>> g_devices;  // All device contexts
thread_local int g_current_device = -1;                 // Currently selected device ID (per-thread)

// Get the current device context
DeviceContext& current_device() {
    if (g_devices.empty()) {
        throw std::runtime_error("No device selected. Call initialize() first.");
    }
    // Threads that never called use_device() default to device 0
    if (g_current_device < 0 || g_current_device >= static_cast<int>(g_devices.size())) {
        g_current_device = 0;
        VEDA_CHECK(vedaCtxSetCurrent(g_devices[0]->ctx));
    }
    return *g_devices[g_current_device];
}

} // anonymous namespace

namespace quickmp {

void initialize() {
    if (!g_devices.empty()) {
        throw std::runtime_error("quickmp already initialized. Call finalize() first.");
    }

    VEDA_CHECK(vedaInit(0));

    // Roll back on failure so that initialize() can be retried
    try {
        // Get number of available devices
        int device_count;
        VEDA_CHECK(vedaDeviceGetCount(&device_count));

        if (device_count == 0) {
            throw std::runtime_error("No VE devices found.");
        }

        std::string kernel_path = get_kernel_lib_path();

        // Initialize all devices
        g_devices.reserve(device_count);
        for (int i = 0; i < device_count; ++i) {
            g_devices.emplace_back(std::make_unique<DeviceContext>());
            DeviceContext& dev = *g_devices.back();
            // OMP mode runs kernels on all VE cores via OpenMP from a single VEO context. With
            // SCALAR mode (one context per core), every in-flight call busy-polls on a host thread,
            // and the few host cores assigned per VE limit the throughput.
            VEDA_CHECK(vedaCtxCreate(&dev.ctx, VEDA_CONTEXT_MODE_OMP, i));
            VEDA_CHECK(vedaModuleLoad(&dev.mod, kernel_path.c_str()));
            VEDA_CHECK(vedaModuleGetFunction(&dev.selfjoin, dev.mod, "selfjoin_kernel"));
            VEDA_CHECK(vedaModuleGetFunction(&dev.selfjoin_batch, dev.mod, "selfjoin_batch_kernel"));
            VEDA_CHECK(vedaModuleGetFunction(&dev.abjoin, dev.mod, "abjoin_kernel"));
            VEDA_CHECK(vedaModuleGetFunction(&dev.compute_mean_std, dev.mod, "compute_mean_std_kernel"));
            VEDA_CHECK(vedaModuleGetFunction(&dev.sliding_dot_product, dev.mod, "sliding_dot_product_kernel"));
        }
    } catch (...) {
        g_devices.clear();
        vedaExit();
        throw;
    }

    // Select device 0 by default (vedaCtxCreate leaves the last device current)
    use_device(0);
}

void finalize() {
    // Clear memory pools for all devices (best effort; vedaExit releases the rest)
    for (auto& dev : g_devices) {
        if (vedaCtxSetCurrent(dev->ctx) == VEDA_SUCCESS) {
            dev->pool.clear();
        }
    }

    g_devices.clear();
    g_current_device = -1;

    VEDA_CHECK(vedaExit());
}

int get_device_count() {
    return static_cast<int>(g_devices.size());
}

void use_device(int device) {
    if (device < 0 || device >= static_cast<int>(g_devices.size())) {
        throw std::runtime_error("Invalid device ID: " + std::to_string(device));
    }
    g_current_device = device;
    VEDA_CHECK(vedaCtxSetCurrent(g_devices[device]->ctx));
}

int get_current_device() {
    return g_current_device;
}

void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m) {
    DeviceContext& dev = current_device();

    PoolPtr T_ptr(dev.pool, n * sizeof(double));
    PoolPtr Q_ptr(dev.pool, m * sizeof(double));
    PoolPtr QT_ptr(dev.pool, (n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, Q_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, QT_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), 0));
    VEDA_CHECK(vedaMemcpyHtoDAsync(Q_ptr, Q, m * sizeof(double), 0));
    VEDA_CHECK(vedaLaunchKernelEx(dev.sliding_dot_product, 0, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(QT, QT_ptr, (n - m + 1) * sizeof(double), 0));

    VEDA_CHECK(vedaStreamSynchronize(0));
}

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m) {
    DeviceContext& dev = current_device();

    PoolPtr T_ptr(dev.pool, n * sizeof(double));
    PoolPtr mu_ptr(dev.pool, (n - m + 1) * sizeof(double));
    PoolPtr sigma_ptr(dev.pool, (n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, mu_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, sigma_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), 0));
    VEDA_CHECK(vedaLaunchKernelEx(dev.compute_mean_std, 0, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(mu, mu_ptr, (n - m + 1) * sizeof(double), 0));
    VEDA_CHECK(vedaMemcpyDtoHAsync(sigma, sigma_ptr, (n - m + 1) * sizeof(double), 0));

    VEDA_CHECK(vedaStreamSynchronize(0));
}

void selfjoin(const double *T, double *P, size_t n, size_t m, bool normalize) {
    DeviceContext& dev = current_device();

    PoolPtr T_ptr(dev.pool, n * sizeof(double));
    PoolPtr P_ptr(dev.pool, (n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 2, n));
    VEDA_CHECK(vedaArgsSetU64(args, 3, m));
    VEDA_CHECK(vedaArgsSetI32(args, 4, normalize));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), 0));
    VEDA_CHECK(vedaLaunchKernelEx(dev.selfjoin, 0, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n - m + 1) * sizeof(double), 0));

    VEDA_CHECK(vedaStreamSynchronize(0));
}

void selfjoin_batch(const double *T, double *P, size_t count, size_t n, size_t m,
                    bool normalize) {
    DeviceContext& dev = current_device();
    if (count == 0) {
        return;
    }

    PoolPtr T_ptr(dev.pool, count * n * sizeof(double));
    PoolPtr P_ptr(dev.pool, count * (n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 2, count));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));
    VEDA_CHECK(vedaArgsSetI32(args, 5, normalize));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, count * n * sizeof(double), 0));
    VEDA_CHECK(vedaLaunchKernelEx(dev.selfjoin_batch, 0, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, count * (n - m + 1) * sizeof(double), 0));

    VEDA_CHECK(vedaStreamSynchronize(0));
}

void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, bool normalize) {
    DeviceContext& dev = current_device();

    PoolPtr T1_ptr(dev.pool, n1 * sizeof(double));
    PoolPtr T2_ptr(dev.pool, n2 * sizeof(double));
    PoolPtr P_ptr(dev.pool, (n1 - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T1_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, T2_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n1));
    VEDA_CHECK(vedaArgsSetU64(args, 4, n2));
    VEDA_CHECK(vedaArgsSetU64(args, 5, m));
    VEDA_CHECK(vedaArgsSetI32(args, 6, normalize));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T1_ptr, T1, n1 * sizeof(double), 0));
    VEDA_CHECK(vedaMemcpyHtoDAsync(T2_ptr, T2, n2 * sizeof(double), 0));
    VEDA_CHECK(vedaLaunchKernelEx(dev.abjoin, 0, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n1 - m + 1) * sizeof(double), 0));

    VEDA_CHECK(vedaStreamSynchronize(0));
}

} // namespace quickmp
