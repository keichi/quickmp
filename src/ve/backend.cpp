#include "quickmp.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
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

// Device context holding all resources for a single VE device
struct DeviceContext {
    VEDAcontext ctx;
    VEDAmodule mod;
    VEDAfunction selfjoin;
    VEDAfunction abjoin;
    VEDAfunction selfjoin_ed;
    VEDAfunction abjoin_ed;
    VEDAfunction compute_mean_std;
    VEDAfunction sliding_dot_product;
    VEDAfunction sleep;
    MemoryPool pool;
};

std::vector<std::unique_ptr<DeviceContext>> g_devices;  // All device contexts
int g_current_device = -1;                              // Currently selected device ID

// Get the current device context
DeviceContext& current_device() {
    if (g_current_device < 0 || g_current_device >= static_cast<int>(g_devices.size())) {
        throw std::runtime_error("No device selected. Call initialize() first.");
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
        VEDA_CHECK(vedaCtxCreate(&dev.ctx, VEDA_CONTEXT_MODE_SCALAR, i));
        VEDA_CHECK(vedaModuleLoad(&dev.mod, kernel_path.c_str()));
        VEDA_CHECK(vedaModuleGetFunction(&dev.selfjoin, dev.mod, "selfjoin_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.abjoin, dev.mod, "abjoin_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.selfjoin_ed, dev.mod, "selfjoin_ed_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.abjoin_ed, dev.mod, "abjoin_ed_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.compute_mean_std, dev.mod, "compute_mean_std_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.sliding_dot_product, dev.mod, "sliding_dot_product_kernel"));
        VEDA_CHECK(vedaModuleGetFunction(&dev.sleep, dev.mod, "sleep_kernel"));
    }

    // Select device 0 by default
    g_current_device = 0;
}

void finalize() {
    // Clear memory pools for all devices
    for (auto& dev : g_devices) {
        VEDA_CHECK(vedaCtxSetCurrent(dev->ctx));
        dev->pool.clear();
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

void sliding_dot_product(const double *T, const double *Q, double *QT,
                         size_t n, size_t m, int stream) {
    DeviceContext& dev = current_device();
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = dev.pool.alloc(n * sizeof(double));
    VEDAdeviceptr Q_ptr = dev.pool.alloc(m * sizeof(double));
    VEDAdeviceptr QT_ptr = dev.pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, Q_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, QT_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyHtoDAsync(Q_ptr, Q, m * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(dev.sliding_dot_product, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(QT, QT_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    dev.pool.free(T_ptr);
    dev.pool.free(Q_ptr);
    dev.pool.free(QT_ptr);
}

void compute_mean_std(const double *T, double *mu, double *sigma,
                      size_t n, size_t m, int stream) {
    DeviceContext& dev = current_device();
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = dev.pool.alloc(n * sizeof(double));
    VEDAdeviceptr mu_ptr = dev.pool.alloc((n - m + 1) * sizeof(double));
    VEDAdeviceptr sigma_ptr = dev.pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, mu_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, sigma_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n));
    VEDA_CHECK(vedaArgsSetU64(args, 4, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(dev.compute_mean_std, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(mu, mu_ptr, (n - m + 1) * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyDtoHAsync(sigma, sigma_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    dev.pool.free(T_ptr);
    dev.pool.free(mu_ptr);
    dev.pool.free(sigma_ptr);
}

void selfjoin(const double *T, double *P, size_t n, size_t m, int stream, bool normalize) {
    DeviceContext& dev = current_device();
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T_ptr = dev.pool.alloc(n * sizeof(double));
    VEDAdeviceptr P_ptr = dev.pool.alloc((n - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 2, n));
    VEDA_CHECK(vedaArgsSetU64(args, 3, m));

    VEDAfunction kernel = normalize ? dev.selfjoin : dev.selfjoin_ed;

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(kernel, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    dev.pool.free(T_ptr);
    dev.pool.free(P_ptr);
}

void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, int stream, bool normalize) {
    DeviceContext& dev = current_device();
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAdeviceptr T1_ptr = dev.pool.alloc(n1 * sizeof(double));
    VEDAdeviceptr T2_ptr = dev.pool.alloc(n2 * sizeof(double));
    VEDAdeviceptr P_ptr = dev.pool.alloc((n1 - m + 1) * sizeof(double));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T1_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, T2_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n1));
    VEDA_CHECK(vedaArgsSetU64(args, 4, n2));
    VEDA_CHECK(vedaArgsSetU64(args, 5, m));

    VEDAfunction kernel = normalize ? dev.abjoin : dev.abjoin_ed;

    VEDA_CHECK(vedaMemcpyHtoDAsync(T1_ptr, T1, n1 * sizeof(double), veda_stream));
    VEDA_CHECK(vedaMemcpyHtoDAsync(T2_ptr, T2, n2 * sizeof(double), veda_stream));
    VEDA_CHECK(vedaLaunchKernelEx(kernel, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n1 - m + 1) * sizeof(double), veda_stream));

    VEDA_CHECK(vedaStreamSynchronize(veda_stream));

    dev.pool.free(T1_ptr);
    dev.pool.free(T2_ptr);
    dev.pool.free(P_ptr);
}

void sleep_us(uint64_t microseconds, int stream) {
    DeviceContext& dev = current_device();
    VEDAstream veda_stream = static_cast<VEDAstream>(stream);

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetU64(args, 0, microseconds));

    VEDA_CHECK(vedaLaunchKernelEx(dev.sleep, veda_stream, args, 1, nullptr));
    VEDA_CHECK(vedaStreamSynchronize(veda_stream));
}

} // namespace quickmp
