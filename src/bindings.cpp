#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "quickmp.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using const_pyarr_t =
    nb::ndarray<const double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using pyarr_t = nb::ndarray<double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using const_pyarr2_t =
    nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using pyarr2_t = nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

static bool g_initialized = false;

static void check_window(size_t n, size_t m) {
    if (m == 0 || m > n) {
        throw std::invalid_argument("Window size m must satisfy 1 <= m <= len(T).");
    }
}

NB_MODULE(_quickmp, m) {
    m.doc() = "Quickly compute matrix profiles";

    m.def(
        "initialize",
        []() {
            if (g_initialized) {
                throw std::runtime_error("quickmp already initialized. Call finalize() first.");
            }
            quickmp::initialize();
            g_initialized = true;
        },
        R"doc(
        Initialize the quickmp backend.

        Initializes all available devices and selects device 0.
    )doc");

    m.def(
        "finalize",
        []() {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized.");
            }
            quickmp::finalize();
            g_initialized = false;
        },
        R"doc(
        Finalize the quickmp backend.
    )doc");

    m.def(
        "get_device_count",
        []() {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            return quickmp::get_device_count();
        },
        R"doc(
        Get the number of available devices.

        Returns:
          Number of available devices (VE: number of VE devices, CPU: always 1)
    )doc");

    m.def(
        "use_device",
        [](int device) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            quickmp::use_device(device);
        },
        "device"_a,
        R"doc(
        Switch to the specified device.

        Args:
          device: Device ID to use
    )doc");

    m.def(
        "get_current_device",
        []() {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            return quickmp::get_current_device();
        },
        R"doc(
        Get the currently selected device ID.

        Returns:
          Currently selected device ID
    )doc");

    m.def(
        "sliding_dot_product",
        [](const_pyarr_t T, const_pyarr_t Q) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            size_t n = T.shape(0);
            size_t m = Q.shape(0);
            check_window(n, m);

            std::vector<double> QT(n - m + 1);

            {
                nb::gil_scoped_release release;
                quickmp::sliding_dot_product(T.data(), Q.data(), QT.data(), n, m);
            }

            return pyarr_t(QT.data(), {QT.size()}).cast();
        },
        "T"_a, "Q"_a,
        R"doc(
        Compute the sliding dot product between time series T and Q.

        Args:
          T: Time series
          Q: Time series

        Returns:
          Sliding dot product
    )doc");

    m.def(
        "compute_mean_std",
        [](const_pyarr_t T, size_t m) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            size_t n = T.shape(0);
            check_window(n, m);
            std::vector<double> mu(n - m + 1);
            std::vector<double> sigma(n - m + 1);

            {
                nb::gil_scoped_release release;
                quickmp::compute_mean_std(T.data(), mu.data(), sigma.data(), n, m);
            }

            return std::make_pair(pyarr_t(mu.data(), {mu.size()}).cast(),
                                  pyarr_t(sigma.data(), {sigma.size()}).cast());
        },
        "T"_a, "m"_a,
        R"doc(
        Compute the mean and standard deviation of every subsequence in time series T.

        Args:
          T: Time series
          m: Window size

        Returns:
          Tuple of mean and standard deviation
    )doc");

    m.def(
        "selfjoin",
        [](const_pyarr_t T, size_t m, bool normalize) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            size_t n = T.shape(0);
            check_window(n, m);
            std::vector<double> P(n - m + 1);

            {
                nb::gil_scoped_release release;
                quickmp::selfjoin(T.data(), P.data(), n, m, normalize);
            }

            return pyarr_t(P.data(), {P.size()}).cast();
        },
        "T"_a, "m"_a, "normalize"_a = true,
        R"doc(
        Compute the matrix profile for time series T.

        Args:
          T: Time series
          m: Window size
          normalize: If True (default), use Z-normalized Euclidean distance. If False, use raw Euclidean distance.

        Returns:
          Matrix profile
    )doc");

    m.def(
        "selfjoin_batch",
        [](const_pyarr2_t T, size_t m, bool normalize) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            size_t count = T.shape(0);
            size_t n = T.shape(1);
            check_window(n, m);
            std::vector<double> P(count * (n - m + 1));

            {
                nb::gil_scoped_release release;
                quickmp::selfjoin_batch(T.data(), P.data(), count, n, m, normalize);
            }

            return pyarr2_t(P.data(), {count, n - m + 1}).cast();
        },
        "T"_a, "m"_a, "normalize"_a = true,
        R"doc(
        Compute the matrix profiles for a batch of time series of the same length.

        Time series are processed in parallel, which is faster than calling selfjoin() for each
        time series when there are many of them.

        Args:
          T: 2D array where each row is a time series
          m: Window size
          normalize: If True (default), use Z-normalized Euclidean distance. If False, use raw Euclidean distance.

        Returns:
          2D array where each row is the matrix profile of the corresponding time series
    )doc");

    m.def(
        "abjoin",
        [](const_pyarr_t T1, const_pyarr_t T2, size_t m, bool normalize) {
            if (!g_initialized) {
                throw std::runtime_error("quickmp not initialized. Call initialize() first.");
            }
            size_t n1 = T1.shape(0);
            size_t n2 = T2.shape(0);
            check_window(n1, m);
            check_window(n2, m);
            std::vector<double> P(n1 - m + 1);

            {
                nb::gil_scoped_release release;
                quickmp::abjoin(T1.data(), T2.data(), P.data(), n1, n2, m, normalize);
            }

            return pyarr_t(P.data(), {P.size()}).cast();
        },
        "T1"_a, "T2"_a, "m"_a, "normalize"_a = true,
        R"doc(
        Compute the matrix profile between time series T1 and T2.

        Args:
          T1: Time series
          T2: Time series
          m: Window size
          normalize: If True (default), use Z-normalized Euclidean distance. If False, use raw Euclidean distance.

        Returns:
          Matrix profile
    )doc");

    // Register cleanup function to be called at module unload
    static int dummy = 0;
    m.attr("_cleanup") = nb::capsule(&dummy, [](void *) noexcept {
        if (g_initialized) {
            quickmp::finalize();
            g_initialized = false;
        }
    });
}
