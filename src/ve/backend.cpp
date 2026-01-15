#include "quickmp.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

VEDAcontext g_ctx;
VEDAmodule g_mod;
VEDAfunction g_selfjoin;
VEDAfunction g_abjoin;

} // anonymous namespace

namespace quickmp {

void init() {
    VEDA_CHECK(vedaInit(0));
    VEDA_CHECK(vedaCtxCreate(&g_ctx, VEDA_CONTEXT_MODE_SCALAR, 0));
    VEDA_CHECK(vedaModuleLoad(&g_mod, get_kernel_lib_path().c_str()));
    VEDA_CHECK(vedaModuleGetFunction(&g_selfjoin, g_mod, "selfjoin"));
    VEDA_CHECK(vedaModuleGetFunction(&g_abjoin, g_mod, "abjoin"));
}

void finalize() {
    VEDA_CHECK(vedaExit());
}

void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m) {
    throw std::runtime_error("sliding_dot_product is not implemented for VE backend");
}

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m) {
    throw std::runtime_error("compute_mean_std is not implemented for VE backend");
}

void selfjoin(const double *T, double *P, size_t n, size_t m) {
    VEDAstream stream = 0;

    VEDAdeviceptr T_ptr, P_ptr;
    VEDA_CHECK(vedaMemAllocAsync(&T_ptr, n * sizeof(double), stream));
    VEDA_CHECK(vedaMemAllocAsync(&P_ptr, (n - m + 1) * sizeof(double), stream));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 2, n));
    VEDA_CHECK(vedaArgsSetU64(args, 3, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T_ptr, T, n * sizeof(double), stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_selfjoin, stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n - m + 1) * sizeof(double), stream));

    VEDA_CHECK(vedaMemFreeAsync(T_ptr, stream));
    VEDA_CHECK(vedaMemFreeAsync(P_ptr, stream));

    VEDA_CHECK(vedaStreamSynchronize(stream));
}

void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m) {
    VEDAstream stream = 0;

    VEDAdeviceptr T1_ptr, T2_ptr, P_ptr;
    VEDA_CHECK(vedaMemAllocAsync(&T1_ptr, n1 * sizeof(double), stream));
    VEDA_CHECK(vedaMemAllocAsync(&T2_ptr, n2 * sizeof(double), stream));
    VEDA_CHECK(vedaMemAllocAsync(&P_ptr, (n1 - m + 1) * sizeof(double), stream));

    VEDAargs args;
    VEDA_CHECK(vedaArgsCreate(&args));
    VEDA_CHECK(vedaArgsSetVPtr(args, 0, T1_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 1, T2_ptr));
    VEDA_CHECK(vedaArgsSetVPtr(args, 2, P_ptr));
    VEDA_CHECK(vedaArgsSetU64(args, 3, n1));
    VEDA_CHECK(vedaArgsSetU64(args, 4, n2));
    VEDA_CHECK(vedaArgsSetU64(args, 5, m));

    VEDA_CHECK(vedaMemcpyHtoDAsync(T1_ptr, T1, n1 * sizeof(double), stream));
    VEDA_CHECK(vedaMemcpyHtoDAsync(T2_ptr, T2, n2 * sizeof(double), stream));
    VEDA_CHECK(vedaLaunchKernelEx(g_abjoin, stream, args, 1, nullptr));
    VEDA_CHECK(vedaMemcpyDtoHAsync(P, P_ptr, (n1 - m + 1) * sizeof(double), stream));

    VEDA_CHECK(vedaMemFreeAsync(T1_ptr, stream));
    VEDA_CHECK(vedaMemFreeAsync(T2_ptr, stream));
    VEDA_CHECK(vedaMemFreeAsync(P_ptr, stream));

    VEDA_CHECK(vedaStreamSynchronize(stream));
}

} // namespace quickmp
