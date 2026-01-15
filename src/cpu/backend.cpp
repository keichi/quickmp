#include "quickmp.hpp"
#include "cpu/internal.hpp"

#include <unistd.h>

namespace quickmp {

void initialize(int device) {
    // device is ignored for CPU backend
    (void)device;
}

void finalize() {
    // No finalization needed for CPU backend
}

void sliding_dot_product(const double *T, const double *Q, double *QT,
                         size_t n, size_t m, int stream) {
    (void)stream;
    sliding_dot_product_fft(T, Q, QT, n, m);
}

void compute_mean_std(const double *T, double *mu, double *sigma,
                      size_t n, size_t m, int stream) {
    (void)stream;
    ::compute_mean_std(T, mu, sigma, n, m);
}

void selfjoin(const double *T, double *P, size_t n, size_t m, int stream) {
    (void)stream;
    ::selfjoin(T, P, n, m);
}

void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, int stream) {
    (void)stream;
    ::abjoin(T1, T2, P, n1, n2, m);
}

void sleep_us(uint64_t microseconds, int stream) {
    (void)stream;
    usleep(microseconds);
}

} // namespace quickmp
