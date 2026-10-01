// STOMP implementation shared by the CPU and VE backends.
// The includer must declare compute_mean_std() and compute_squared_sum() beforehand.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stomp {

inline int max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Norm = true: Z-normalized distance. P holds m * Pearson correlation (maximized), a = mean and
// b = 1 / std. Norm = false: Euclidean distance. P holds the squared distance (minimized), a = sum
// of squares and b is unused.
template <bool Norm>
void compute_stats(const double *T, double *a, double *b, size_t n, size_t m)
{
    if constexpr (Norm) {
        compute_mean_std(T, a, b, n, m);
        for (size_t i = 0; i < n - m + 1; i++) {
            b[i] = 1.0 / b[i];
        }
    } else {
        compute_squared_sum(T, a, n, m);
    }
}

// Finite so that it also works with -ffast-math
template <bool Norm> constexpr double worst()
{
    return Norm ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();
}

template <bool Norm> inline double better(double x, double y)
{
    return Norm ? std::max(x, y) : std::min(x, y);
}

template <bool Norm>
inline double dist(double qt, double ai, double bi, double aj, double bj, size_t m)
{
    if constexpr (Norm) {
        return (qt - m * ai * aj) * bi * bj;
    } else {
        return ai + aj - 2.0 * qt;
    }
}

template <bool Norm> inline double finalize(double p, size_t m)
{
    if constexpr (Norm) {
        return std::sqrt(2.0 * m * (1.0 - p / m));
    } else {
        return std::sqrt(p);
    }
}

// Updates P with all pairs (i, j) such that ib <= i < ie and j > i + excl
template <bool Norm>
void selfjoin_rows(const double *__restrict T, const double *__restrict a,
                   const double *__restrict b, double *__restrict P, size_t n, size_t m,
                   size_t excl, size_t ib, size_t ie)
{
    const size_t N = n - m + 1;
    if (ib >= ie) {
        return;
    }

    std::vector<double> buf(2 * N);
    double *__restrict QT = buf.data();
    double *__restrict QT2 = QT + N;

    // First row: compute the dot products directly
    // TODO: Use FFT if m is large
    for (size_t j = ib + excl + 1; j < N; j++) {
        QT[j] = 0.0;
    }
    for (size_t k = 0; k < m; k++) {
        for (size_t j = ib + excl + 1; j < N; j++) {
            QT[j] += T[ib + k] * T[j + k];
        }
    }

    double best = P[ib];
    for (size_t j = ib + excl + 1; j < N; j++) {
        double d = dist<Norm>(QT[j], a[ib], b[ib], a[j], b[j], m);
        P[j] = better<Norm>(P[j], d);
        best = better<Norm>(best, d);
    }
    P[ib] = best;

    for (size_t i = ib + 1; i < ie; i++) {
        best = P[i];

        for (size_t j = i + excl + 1; j < N; j++) {
            // Calculate sliding-window dot product
            QT2[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1];

            double d = dist<Norm>(QT2[j], a[i], b[i], a[j], b[j], m);

            // Update matrix profile (column and row)
            P[j] = better<Norm>(P[j], d);
            best = better<Norm>(best, d);
        }

        P[i] = best;

        std::swap(QT, QT2);
    }
}

// Updates P (indexed by T1 subsequences) with all pairs (j, i) such that ib <= i < ie, where i
// indexes T2 subsequences
template <bool Norm>
void abjoin_rows(const double *__restrict T1, const double *__restrict T2,
                 const double *__restrict a1, const double *__restrict b1,
                 const double *__restrict a2, const double *__restrict b2, double *__restrict P,
                 size_t n1, size_t m, size_t ib, size_t ie)
{
    const size_t N1 = n1 - m + 1;
    if (ib >= ie) {
        return;
    }

    std::vector<double> buf(2 * N1);
    double *__restrict QT = buf.data();
    double *__restrict QT2 = QT + N1;

    // First row: compute the dot products directly
    // TODO: Use FFT if m is large
    for (size_t j = 0; j < N1; j++) {
        QT[j] = 0.0;
    }
    for (size_t k = 0; k < m; k++) {
        for (size_t j = 0; j < N1; j++) {
            QT[j] += T2[ib + k] * T1[j + k];
        }
    }

    for (size_t j = 0; j < N1; j++) {
        P[j] = better<Norm>(P[j], dist<Norm>(QT[j], a1[j], b1[j], a2[ib], b2[ib], m));
    }

    for (size_t i = ib + 1; i < ie; i++) {
        // Compute leftmost element
        double q0 = 0.0;
        for (size_t k = 0; k < m; k++) {
            q0 += T2[i + k] * T1[k];
        }
        QT2[0] = q0;
        P[0] = better<Norm>(P[0], dist<Norm>(q0, a1[0], b1[0], a2[i], b2[i], m));

        for (size_t j = 1; j < N1; j++) {
            // Calculate sliding-window dot product
            QT2[j] = QT[j - 1] - T1[j - 1] * T2[i - 1] + T1[j + m - 1] * T2[i + m - 1];

            double d = dist<Norm>(QT2[j], a1[j], b1[j], a2[i], b2[i], m);

            // Update matrix profile
            P[j] = better<Norm>(P[j], d);
        }

        std::swap(QT, QT2);
    }
}

// Merges per-chunk profiles into P
template <bool Norm>
void merge_profiles(const double *Pc, double *P, size_t N, int nchunks)
{
    for (size_t j = 0; j < N; j++) {
        P[j] = Pc[j];
    }
    for (int c = 1; c < nchunks; c++) {
        for (size_t j = 0; j < N; j++) {
            P[j] = better<Norm>(P[j], Pc[c * N + j]);
        }
    }
}

// Returns the first row of chunk c when rows [0, R) are split into nchunks chunks of equal work,
// where row i has R - i pairs
inline size_t triangle_split(size_t R, int c, int nchunks)
{
    return c >= nchunks ? R
                        : static_cast<size_t>(R - R * std::sqrt(1.0 - static_cast<double>(c) / nchunks));
}

// Converts the profile accumulated by selfjoin_rows to distances
template <bool Norm>
void selfjoin_finalize(double *P, size_t N, size_t m, size_t excl)
{
    // Subsequences in the exclusion zone of the first one start from zero correlation
    if constexpr (Norm) {
        for (size_t j = 0; j < std::min(excl + 1, N); j++) {
            P[j] = std::max(P[j], 0.0);
        }
    }

    // Subsequences without any pair outside the exclusion zone get infinity. Use integer ops
    // since -ffast-math assumes that infinities do not occur.
    if constexpr (!Norm) {
        const double w = worst<Norm>();
        const uint64_t inf_bits = 0x7ff0000000000000;
        for (size_t j = 0; j < N; j++) {
            if (std::memcmp(&P[j], &w, sizeof(double)) == 0) {
                std::memcpy(&P[j], &inf_bits, sizeof(double));
            } else {
                P[j] = finalize<Norm>(P[j], m);
            }
        }
        return;
    }

    for (size_t j = 0; j < N; j++) {
        P[j] = finalize<Norm>(P[j], m);
    }
}

// Computes the matrix profile of T in parallel. Rows are split into one chunk per thread, and
// each chunk accumulates into its own profile.
template <bool Norm> void selfjoin(const double *T, double *P, size_t n, size_t m)
{
    const size_t N = n - m + 1;
    const size_t excl = std::ceil(m / 4.0);
    // Rows that have at least one pair outside the exclusion zone
    const size_t R = N > excl + 1 ? N - excl - 1 : 0;

    std::vector<double> a(N), b(N);
    compute_stats<Norm>(T, a.data(), b.data(), n, m);

    const int nchunks = max_threads();
    std::vector<double> Pc(nchunks * N, worst<Norm>());

#pragma omp parallel for
    for (int c = 0; c < nchunks; c++) {
        selfjoin_rows<Norm>(T, a.data(), b.data(), &Pc[c * N], n, m, excl,
                            triangle_split(R, c, nchunks), triangle_split(R, c + 1, nchunks));
    }

    merge_profiles<Norm>(Pc.data(), P, N, nchunks);
    selfjoin_finalize<Norm>(P, N, m, excl);
}

// Computes the matrix profiles of count time series of length n stored row-major in T in
// parallel, one time series per thread
template <bool Norm>
void selfjoin_batch(const double *T, double *P, size_t count, size_t n, size_t m)
{
    const size_t N = n - m + 1;
    const size_t excl = std::ceil(m / 4.0);
    const size_t R = N > excl + 1 ? N - excl - 1 : 0;

#pragma omp parallel for
    for (size_t k = 0; k < count; k++) {
        const double *Tk = T + k * n;
        double *Pk = P + k * N;

        std::vector<double> a(N), b(N);
        compute_stats<Norm>(Tk, a.data(), b.data(), n, m);

        std::fill(Pk, Pk + N, worst<Norm>());
        selfjoin_rows<Norm>(Tk, a.data(), b.data(), Pk, n, m, excl, 0, R);
        selfjoin_finalize<Norm>(Pk, N, m, excl);
    }
}

// For each subsequence in T1, returns its nearest neighbor in T2. Computed in parallel: rows (T2
// subsequences) are split into one chunk per thread, and each chunk accumulates into its own
// profile.
template <bool Norm>
void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m)
{
    const size_t N1 = n1 - m + 1, N2 = n2 - m + 1;

    std::vector<double> a1(N1), b1(N1), a2(N2), b2(N2);
    compute_stats<Norm>(T1, a1.data(), b1.data(), n1, m);
    compute_stats<Norm>(T2, a2.data(), b2.data(), n2, m);

    const int nchunks = max_threads();
    std::vector<double> Pc(nchunks * N1, worst<Norm>());

#pragma omp parallel for
    for (int c = 0; c < nchunks; c++) {
        abjoin_rows<Norm>(T1, T2, a1.data(), b1.data(), a2.data(), b2.data(), &Pc[c * N1], n1, m,
                          N2 * c / nchunks, N2 * (c + 1) / nchunks);
    }

    merge_profiles<Norm>(Pc.data(), P, N1, nchunks);

    for (size_t j = 0; j < N1; j++) {
        P[j] = finalize<Norm>(P[j], m);
    }
}

inline void selfjoin(const double *T, double *P, size_t n, size_t m, bool normalize)
{
    normalize ? selfjoin<true>(T, P, n, m) : selfjoin<false>(T, P, n, m);
}

inline void selfjoin_batch(const double *T, double *P, size_t count, size_t n, size_t m,
                           bool normalize)
{
    normalize ? selfjoin_batch<true>(T, P, count, n, m) : selfjoin_batch<false>(T, P, count, n, m);
}

inline void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m,
                   bool normalize)
{
    normalize ? abjoin<true>(T1, T2, P, n1, n2, m) : abjoin<false>(T1, T2, P, n1, n2, m);
}

} // namespace stomp
