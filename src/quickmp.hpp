#pragma once

#include <cstddef>

namespace quickmp {

// Initialize backend (called once at module load)
void init();

// Finalize backend (called at module unload)
void finalize();

// Compute sliding dot product between T and Q
void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m);

// Compute mean and standard deviation of every subsequence
void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m);

// Self-join: compute matrix profile for a single time series
void selfjoin(const double *T, double *P, size_t n, size_t m);

// AB-join: compute matrix profile between two time series
void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m);

} // namespace quickmp
