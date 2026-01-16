#include <algorithm>
#include <cmath>

#include "cpu/internal.hpp"

void selfjoin(const double *__restrict _T, double *__restrict _P, size_t n, size_t m)
{
    size_t excl_zone = std::ceil(m / 4.0);

    // Note: This is a workaround. icpx ignores __restrict on arugments in some situations.
    const double *__restrict T = _T;
    double *__restrict P = _P;

    double *__restrict QT = new double[n - m + 1];
    double *__restrict QT2 = new double[n - m + 1];
    double *__restrict mu = new double[n - m + 1];
    double *__restrict sigma_inv = new double[n - m + 1];

    compute_mean_std(T, mu, sigma_inv, n, m);

    for (size_t i = 0; i < n - m + 1; i++) {
        sigma_inv[i] = 1.0 / sigma_inv[i];
    }

    // TODO: Use sliding_dot_product_fft if m is large
    sliding_dot_product_naive(T, T, QT, n, m);

    for (size_t j = 0; j < n - m + 1; j++) {
        P[j] = (QT[j] - m * mu[0] * mu[j]) * sigma_inv[0] * sigma_inv[j];
    }

    for (size_t j = 0; j < excl_zone + 1; j++) {
        P[j] = 0.0;
    }

    for (size_t j = excl_zone + 1; j < n - m + 1; j++) {
        P[0] = std::max(P[0], P[j]);
    }

    for (size_t i = 1; i < n - m + 1; i++) {
        double max_pi = P[i];

        for (size_t j = i + excl_zone + 1; j < n - m + 1; j++) {
            // Calculate sliding dot product
            QT2[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1];

            // Calculate distance profile
            double dist = (QT2[j] - m * mu[i] * mu[j]) * sigma_inv[i] * sigma_inv[j];

            // Update matrix profile
            P[j] = std::max(P[j], dist);

            // Note: gcc/clang require -ffast-math to vectorize this reduction.
            max_pi = std::max(max_pi, dist);
        }

        P[i] = max_pi;

        std::swap(QT, QT2);
    }

    for (size_t i = 0; i < n - m + 1; i++) {
        P[i] = std::sqrt(2.0 * m * (1.0 - P[i] / m));
    }

    delete[] QT;
    delete[] QT2;
    delete[] mu;
    delete[] sigma_inv;
}

// For each subsequence in T1, returns its nearest neighbor in T2
void abjoin(const double *__restrict _T1, const double *__restrict _T2, double *__restrict _P,
            size_t n1, size_t n2, size_t m)
{
    // Note: This is a workaround. icpx ignores __restrict on arugments in some situations.
    const double *__restrict T1 = _T1;
    const double *__restrict T2 = _T2;
    double *__restrict P = _P;

    double *__restrict QT = new double[n1 - m + 1];
    double *__restrict QT2 = new double[n1 - m + 1];
    double *__restrict mu1 = new double[n1 - m + 1];
    double *__restrict mu2 = new double[n2 - m + 1];
    double *__restrict sigma_inv1 = new double[n1 - m + 1];
    double *__restrict sigma_inv2 = new double[n2 - m + 1];

    compute_mean_std(T1, mu1, sigma_inv1, n1, m);
    compute_mean_std(T2, mu2, sigma_inv2, n2, m);

    for (size_t i = 0; i < n1 - m + 1; i++) {
        sigma_inv1[i] = 1.0 / sigma_inv1[i];
    }

    for (size_t i = 0; i < n2 - m + 1; i++) {
        sigma_inv2[i] = 1.0 / sigma_inv2[i];
    }

    // TODO: Use sliding_dot_product_fft if m is large
    sliding_dot_product_naive(T1, T2, QT, n1, m);

    for (size_t j = 0; j < n1 - m + 1; j++) {
        P[j] = (QT[j] - m * mu1[j] * mu2[0]) * sigma_inv1[j] * sigma_inv2[0];
    }

    for (size_t i = 1; i < n2 - m + 1; i++) {
        // Compute leftmost element
        sliding_dot_product_naive(T1, T2 + i, QT2, m, m);
        P[0] = std::max(P[0], (QT2[0] - m * mu1[0] * mu2[i]) * sigma_inv1[0] * sigma_inv2[i]);

        for (size_t j = 1; j < n1 - m + 1; j++) {
            // Calculate sliding dot product
            QT2[j] = QT[j - 1] - T1[j - 1] * T2[i - 1] + T1[j + m - 1] * T2[i + m - 1];

            // Calculate distance profile
            double dist = (QT2[j] - m * mu1[j] * mu2[i]) * sigma_inv1[j] * sigma_inv2[i];

            // Update matrix profile
            P[j] = std::max(P[j], dist);
        }

        std::swap(QT, QT2);
    }

    for (size_t i = 0; i < n1 - m + 1; i++) {
        P[i] = std::sqrt(2.0 * m * (1.0 - P[i] / m));
    }

    delete[] QT;
    delete[] QT2;
    delete[] mu1;
    delete[] mu2;
    delete[] sigma_inv1;
    delete[] sigma_inv2;
}

// Non-normalized Euclidean distance version of selfjoin
void selfjoin_ed(const double *__restrict _T, double *__restrict _P, size_t n, size_t m)
{
    size_t excl_zone = std::ceil(m / 4.0);

    const double *__restrict T = _T;
    double *__restrict P = _P;

    double *__restrict QT = new double[n - m + 1];
    double *__restrict QT2 = new double[n - m + 1];
    double *__restrict S = new double[n - m + 1];

    compute_squared_sum(T, S, n, m);

    // TODO: Use sliding_dot_product_fft if m is large
    sliding_dot_product_naive(T, T, QT, n, m);

    // Initialize distance profile (squared distance)
    for (size_t j = 0; j < n - m + 1; j++) {
        P[j] = S[0] + S[j] - 2.0 * QT[j];
    }

    // Set exclusion zone to infinity (minimization problem)
    for (size_t j = 0; j < excl_zone + 1; j++) {
        P[j] = INFINITY;
    }

    // Find minimum for first row
    for (size_t j = excl_zone + 1; j < n - m + 1; j++) {
        P[0] = std::min(P[0], P[j]);
    }

    // STOMP main loop (track minimum)
    for (size_t i = 1; i < n - m + 1; i++) {
        double min_pi = P[i];

        for (size_t j = i + excl_zone + 1; j < n - m + 1; j++) {
            // Calculate sliding dot product
            QT2[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1];

            // Squared Euclidean distance
            double dist_sq = S[i] + S[j] - 2.0 * QT2[j];

            // Update matrix profile (minimum)
            P[j] = std::min(P[j], dist_sq);
            min_pi = std::min(min_pi, dist_sq);
        }

        P[i] = min_pi;

        std::swap(QT, QT2);
    }

    // Convert squared distance to distance
    for (size_t i = 0; i < n - m + 1; i++) {
        P[i] = std::sqrt(P[i]);
    }

    delete[] QT;
    delete[] QT2;
    delete[] S;
}

// Non-normalized Euclidean distance version of abjoin
// For each subsequence in T1, returns its nearest neighbor in T2
void abjoin_ed(const double *__restrict _T1, const double *__restrict _T2, double *__restrict _P,
               size_t n1, size_t n2, size_t m)
{
    const double *__restrict T1 = _T1;
    const double *__restrict T2 = _T2;
    double *__restrict P = _P;

    double *__restrict QT = new double[n1 - m + 1];
    double *__restrict QT2 = new double[n1 - m + 1];
    double *__restrict S1 = new double[n1 - m + 1];
    double *__restrict S2 = new double[n2 - m + 1];

    compute_squared_sum(T1, S1, n1, m);
    compute_squared_sum(T2, S2, n2, m);

    // TODO: Use sliding_dot_product_fft if m is large
    sliding_dot_product_naive(T1, T2, QT, n1, m);

    // Initialize distance profile (squared distance)
    for (size_t j = 0; j < n1 - m + 1; j++) {
        P[j] = S1[j] + S2[0] - 2.0 * QT[j];
    }

    for (size_t i = 1; i < n2 - m + 1; i++) {
        // Compute leftmost element
        sliding_dot_product_naive(T1, T2 + i, QT2, m, m);
        double dist_sq = S1[0] + S2[i] - 2.0 * QT2[0];
        P[0] = std::min(P[0], dist_sq);

        for (size_t j = 1; j < n1 - m + 1; j++) {
            // Calculate sliding dot product
            QT2[j] = QT[j - 1] - T1[j - 1] * T2[i - 1] + T1[j + m - 1] * T2[i + m - 1];

            // Squared Euclidean distance
            dist_sq = S1[j] + S2[i] - 2.0 * QT2[j];

            // Update matrix profile (minimum)
            P[j] = std::min(P[j], dist_sq);
        }

        std::swap(QT, QT2);
    }

    // Convert squared distance to distance
    for (size_t i = 0; i < n1 - m + 1; i++) {
        P[i] = std::sqrt(P[i]);
    }

    delete[] QT;
    delete[] QT2;
    delete[] S1;
    delete[] S2;
}
