#include <complex>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft.hpp"

void sliding_dot_product_fft(const double *T, const double *Q, double *QT, size_t n, size_t m)
{
    std::vector<double> Ta(n * 2), Qra(n * 2);
    std::vector<std::complex<double>> Taf(n + 1), Qraf(n + 1);

    for (size_t i = 0; i < n; i++) {
        Ta[i] = T[i];
    }

    for (size_t i = 0; i < m; i++) {
        Qra[i] = Q[m - i - 1];
    }

    pocketfft::r2c({n * 2}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true, Qra.data(),
                   Qraf.data(), 1.0);

    pocketfft::r2c({n * 2}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true, Ta.data(),
                   Taf.data(), 1.0);

    for (size_t i = 0; i < n + 1; i++) {
        Qraf[i] *= Taf[i];
    }

    pocketfft::c2r({n * 2}, {sizeof(std::complex<double>)}, {sizeof(double)}, 0, false, Qraf.data(),
                   Qra.data(), 1.0 / (n * 2));

    for (size_t i = m - 1; i < n; i++) {
        QT[i - m + 1] = Qra[i];
    }
}

void sliding_dot_product_naive(const double *T, const double *Q, double *QT, size_t n,
                                      size_t m)
{
    for (size_t i = 0; i < n - m + 1; i++) {
        QT[i] = 0.0;
    }

    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < n - m + 1; i++) {
            QT[i] += Q[j] * T[i + j];
        }
    }
}
