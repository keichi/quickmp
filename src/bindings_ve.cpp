#include <iostream>
#include <vector>
#include <filesystem>

#include <veda.h>
#include <dlfcn.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
using namespace nb::literals;

using const_pyarr_t =
    nb::ndarray<const double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using pyarr_t = nb::ndarray<double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

#define VEDA(err) check(err, __FILE__, __LINE__)

void check(VEDAresult err, const char* file, const int line) {
    if(err != VEDA_SUCCESS) {
        const char *name, *str;
        vedaGetErrorName	(err, &name);
        vedaGetErrorString	(err, &str);
        printf("%s: %s @ %s:%i\n", name, str, file, line);
        exit(1);
    }
}

std::string get_kernel_lib_path()
{
    Dl_info info;
    dladdr(reinterpret_cast<void*>(&get_kernel_lib_path), &info);

    std::filesystem::path path = info.dli_fname;
    path.replace_filename("libquickmp-device.vso");

    return path.string();
}

NB_MODULE(_quickmp, m)
{
    VEDA(vedaInit(0));

    VEDAcontext ctx;
    VEDA(vedaCtxCreate(&ctx, VEDA_CONTEXT_MODE_SCALAR, 0));

    VEDAmodule mod;
    VEDA(vedaModuleLoad(&mod, get_kernel_lib_path().c_str()));

    VEDAfunction selfjoin, abjoin;
    VEDA(vedaModuleGetFunction(&selfjoin, mod, "selfjoin"));
    VEDA(vedaModuleGetFunction(&abjoin, mod, "abjoin"));

    m.def("selfjoin", [selfjoin](const_pyarr_t T, size_t m, VEDAstream stream) {
        size_t n = T.shape(0);
        std::vector<double> P(n - m + 1);

        VEDAdeviceptr T_ptr, P_ptr;
        VEDA(vedaMemAllocAsync(&T_ptr, n * sizeof(double), stream));
        VEDA(vedaMemAllocAsync(&P_ptr, (n - m + 1) * sizeof(double), stream));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, T_ptr));
        VEDA(vedaArgsSetVPtr(args, 1, P_ptr));
        VEDA(vedaArgsSetU64(args, 2, n));
        VEDA(vedaArgsSetU64(args, 3, m));

        VEDA(vedaMemcpyHtoDAsync(T_ptr, T.data(), n * sizeof(double), stream));
        VEDA(vedaLaunchKernelEx(selfjoin, stream, args, 1, nullptr));
        VEDA(vedaMemcpyDtoHAsync(P.data(), P_ptr, (n - m + 1) * sizeof(double), stream));

        VEDA(vedaMemFreeAsync(T_ptr, stream));
        VEDA(vedaMemFreeAsync(P_ptr, stream));

        VEDA(vedaStreamSynchronize(stream));

        return pyarr_t(P.data(), {P.size()}).cast();
    }, "T"_a, "m"_a, "stream"_a = 0);

    m.def("abjoin", [abjoin](const_pyarr_t T1, const_pyarr_t T2, size_t m, VEDAstream stream) {
        size_t n1 = T1.shape(0);
        size_t n2 = T2.shape(0);
        std::vector<double> P(n1 - m + 1);

        VEDAdeviceptr T1_ptr, T2_ptr, P_ptr;
        VEDA(vedaMemAllocAsync(&T1_ptr, n1 * sizeof(double), stream));
        VEDA(vedaMemAllocAsync(&T2_ptr, n2 * sizeof(double), stream));
        VEDA(vedaMemAllocAsync(&P_ptr, (n1 - m + 1) * sizeof(double), stream));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, T1_ptr));
        VEDA(vedaArgsSetVPtr(args, 1, T2_ptr));
        VEDA(vedaArgsSetVPtr(args, 2, P_ptr));
        VEDA(vedaArgsSetU64(args, 3, n1));
        VEDA(vedaArgsSetU64(args, 4, n2));
        VEDA(vedaArgsSetU64(args, 5, m));

        VEDA(vedaMemcpyHtoDAsync(T1_ptr, T1.data(), n1 * sizeof(double), stream));
        VEDA(vedaMemcpyHtoDAsync(T2_ptr, T2.data(), n2 * sizeof(double), stream));
        VEDA(vedaLaunchKernelEx(abjoin, stream, args, 1, nullptr));
        VEDA(vedaMemcpyDtoHAsync(P.data(), P_ptr, (n1 - m + 1) * sizeof(double), stream));

        VEDA(vedaMemFreeAsync(T1_ptr, stream));
        VEDA(vedaMemFreeAsync(T2_ptr, stream));
        VEDA(vedaMemFreeAsync(P_ptr, stream));

        VEDA(vedaStreamSynchronize(stream));

        return pyarr_t(P.data(), {P.size()}).cast();
    }, "T1"_a, "T2"_a, "m"_a, "stream"_a = 0);

    auto finalize = [](void *ptr) noexcept { VEDA(vedaExit()); };
    m.attr("_cleanup") = nb::capsule(reinterpret_cast<void*>(+finalize), finalize);
}
