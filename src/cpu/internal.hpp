#pragma once

#include <cstddef>

// Internal implementation functions for CPU backend
void sliding_dot_product_fft(const double *T, const double *Q, double *QT, size_t n, size_t m);
void sliding_dot_product_naive(const double *T, const double *Q, double *QT, size_t n, size_t m);
void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m);
void compute_squared_sum(const double *T, double *sum, size_t n, size_t m);
