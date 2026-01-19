#pragma once

#include <cstddef>
#include <cstdint>

namespace quickmp {

// Initialize backend (initializes all available devices, selects device 0)
void initialize();

// Finalize backend
void finalize();

// Get number of available devices (VE: number of VE devices, CPU: always 1)
int get_device_count();

// Switch to the specified device
void use_device(int device);

// Get the currently selected device ID
int get_current_device();

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
// normalize: if true, use Z-normalized Euclidean distance; otherwise use raw Euclidean distance
void selfjoin(const double *T, double *P, size_t n, size_t m, int stream = 0,
              bool normalize = true);

// AB-join: compute matrix profile between two time series
// stream: VE stream number (ignored for CPU)
// normalize: if true, use Z-normalized Euclidean distance; otherwise use raw Euclidean distance
void abjoin(const double *T1, const double *T2, double *P,
            size_t n1, size_t n2, size_t m, int stream = 0, bool normalize = true);

// Sleep for specified microseconds on VE (for benchmarking)
// stream: VE stream number (ignored for CPU)
void sleep_us(uint64_t microseconds, int stream = 0);

// Get number of available streams for parallel execution
// CPU backend: returns number of CPU cores
// VE backend: returns number of VE streams for current context
int get_stream_count();

} // namespace quickmp
