#include "quickmp.hpp"
#include "cpu/internal.hpp"
#include "stomp.hpp"

#include <stdexcept>

namespace {

bool g_initialized = false;

} // anonymous namespace

namespace quickmp {

void initialize() {
    if (g_initialized) {
        throw std::runtime_error("quickmp already initialized. Call finalize() first.");
    }
    g_initialized = true;
}

void finalize() {
    if (!g_initialized) {
        throw std::runtime_error("quickmp not initialized.");
    }
    g_initialized = false;
}

int get_device_count() {
    // CPU backend always has exactly 1 device
    return 1;
}

void use_device(int device) {
    if (device != 0) {
        throw std::runtime_error("CPU backend only supports device 0.");
    }
    // No-op for CPU backend
}

int get_current_device() {
    // CPU backend always uses device 0
    return 0;
}

void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m) {
    sliding_dot_product_fft(T, Q, QT, n, m);
}

void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m) {
    ::compute_mean_std(T, mu, sigma, n, m);
}

void selfjoin(const double *T, double *P, size_t n, size_t m, bool normalize) {
    stomp::selfjoin(T, P, n, m, normalize);
}

void selfjoin_batch(const double *T, double *P, size_t count, size_t n, size_t m,
                    bool normalize) {
    stomp::selfjoin_batch(T, P, count, n, m, normalize);
}

void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m,
            bool normalize) {
    stomp::abjoin(T1, T2, P, n1, n2, m, normalize);
}

} // namespace quickmp
