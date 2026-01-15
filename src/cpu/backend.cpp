#include "quickmp.hpp"
#include "cpu/internal.hpp"

namespace quickmp {

void init() {
    // No initialization needed for CPU backend
}

void finalize() {
    // No finalization needed for CPU backend
}

void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m) {
    sliding_dot_product_fft(T, Q, QT, n, m);
}

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m) {
    ::compute_mean_std(T, mu, sigma, n, m);
}

void selfjoin(const double *T, double *P, size_t n, size_t m) {
    ::selfjoin(T, P, n, m);
}

void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m) {
    ::abjoin(T1, T2, P, n1, n2, m);
}

} // namespace quickmp
