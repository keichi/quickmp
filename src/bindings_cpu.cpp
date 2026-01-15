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

NB_MODULE(_quickmp, m)
{
    m.doc() = "Quickly compute matrix profiles";

    m.def("sliding_dot_product", [](const_pyarr_t T, const_pyarr_t Q) {
        size_t n = T.shape(0);
        size_t m = Q.shape(0);

        std::vector<double> QT(n - m + 1);

        sliding_dot_product_fft(T.data(), Q.data(), QT.data(), n, m);

        return pyarr_t(QT.data(), {QT.size()}).cast();
    }, "T"_a, "Q"_a, R"doc(
        Compute the sliding dot product between time series T and Q.

        Args:
          T: Time series
          Q: Time series
          m: Window size

        Returns:
          Sliding dot product
    )doc");

    m.def("compute_mean_std", [](const_pyarr_t T, size_t m) {
        size_t n = T.shape(0);
        std::vector<double> mu(n - m + 1);
        std::vector<double> sigma(n - m + 1);

        compute_mean_std(T.data(), mu.data(), sigma.data(), n, m);

        return std::make_pair(pyarr_t(mu.data(), {mu.size()}).cast(),
                              pyarr_t(sigma.data(), {sigma.size()}).cast());
    }, "T"_a, "m"_a, R"doc(
        Compute the mean and standard deviation of every subsequence in time series T.

        Args:
          T: Time series
          m: Window size

        Returns:
          Tuple of mean and standard deviation
    )doc");

    m.def("selfjoin", [](const_pyarr_t T, size_t m) {
        size_t n = T.shape(0);
        std::vector<double> P(n - m + 1);

        selfjoin(T.data(), P.data(), n, m);

        return pyarr_t(P.data(), {P.size()}).cast();
    }, "T"_a, "m"_a, R"doc(
        Compute the matrix profile for time series T.

        Args:
          T: Time series
          m: Window size

        Returns:
          Matrix profile
    )doc");

    m.def("abjoin", [](const_pyarr_t T1, const_pyarr_t T2, size_t m) {
        size_t n1 = T1.shape(0);
        size_t n2 = T2.shape(0);
        std::vector<double> P(n1 - m + 1);

        abjoin(T1.data(), T2.data(), P.data(), n1, n2, m);

        return pyarr_t(P.data(), {P.size()}).cast();
    }, "T1"_a, "T2"_a, "m"_a, R"doc(
        Compute the matrix profile between time series T1 and T2.

        Args:
          T1: Time series
          T2: Time series
          m: Window size

        Returns:
          Matrix profile
    )doc");
}
