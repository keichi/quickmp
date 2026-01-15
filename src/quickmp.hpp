#pragma once

#include <cstddef>

namespace quickmp {

// Initialize backend with device number (VE only, ignored for CPU)
void initialize(int device = 0);

// Finalize backend
void finalize();

// Compute sliding dot product between T and Q
// stream: VE stream number (ignored for CPU)
void sliding_dot_product(const double *T, const double *Q, double *QT,
                         size_t n, size_t m, int stream = 0);

// Compute mean and standard deviation of every subsequence
// stream: VE stream number (ignored for CPU)
void compute_mean_std(const double *T, double *mu, double *sigma,
                      size_t n, size_t m, int stream = 0);

// Self-join: compute matrix profile for a single time series
// stream: VE stream number (ignored for CPU)
void selfjoin(const double *T, double *P, size_t n, size_t m, int stream = 0);

// AB-join: compute matrix profile between two time series
// stream: VE stream number (ignored for CPU)
void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, int stream = 0);

} // namespace quickmp
