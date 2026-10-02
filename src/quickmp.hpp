#pragma once

#include <cstddef>

namespace quickmp {

// Initialize backend (initializes all available devices, selects device 0)
void initialize();

// Finalize backend
void finalize();

// Get number of available devices (VE: number of VE devices, CPU: always 1)
int get_device_count();

// Switch to the specified device for the calling thread (threads that have not called
// use_device() use device 0)
void use_device(int device);

// Get the device ID selected for the calling thread
int get_current_device();

// Compute sliding dot product between T and Q
void sliding_dot_product(const double *T, const double *Q, double *QT, size_t n, size_t m);

// Compute mean and standard deviation of every subsequence
void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m);

// Self-join: compute matrix profile for a single time series (parallelized within the series)
// normalize: if true, use Z-normalized Euclidean distance; otherwise use raw Euclidean distance
void selfjoin(const double *T, double *P, size_t n, size_t m, bool normalize = true);

// Batched self-join: compute matrix profiles of count time series of length n (parallelized
// across the series)
// T: count x n (row-major), P: count x (n - m + 1) (row-major)
// normalize: if true, use Z-normalized Euclidean distance; otherwise use raw Euclidean distance
void selfjoin_batch(const double *T, double *P, size_t count, size_t n, size_t m,
                    bool normalize = true);

// AB-join: compute matrix profile between two time series (parallelized within the series)
// normalize: if true, use Z-normalized Euclidean distance; otherwise use raw Euclidean distance
void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, bool normalize = true);

} // namespace quickmp
