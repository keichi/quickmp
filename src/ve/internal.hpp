// vim: set ft=cpp:
#pragma once
#include <cstddef>
#include <veda_device.h>

extern "C" {

void sliding_window_dot_product_fft(const double *T, const double *Q, double *QT, size_t n,
                                    size_t m);
void sliding_window_dot_product_naive(const double *T, const double *Q, double *QT, size_t n,
                                      size_t m);
void compute_mean_std(const double *T, double *mu, double *sigma, size_t n, size_t m);
void compute_squared_sum(const double *T, double *sum, size_t n, size_t m);
void selfjoin(const double *T, double *P, size_t n, size_t m);
void abjoin(const double *T1, const double *T2, double *P, size_t n1, size_t n2, size_t m);

void compute_mean_std_kernel(VEDAdeviceptr T, VEDAdeviceptr mu, VEDAdeviceptr sigma, size_t n,
                             size_t m);
void sliding_dot_product_kernel(VEDAdeviceptr T, VEDAdeviceptr Q, VEDAdeviceptr QT, size_t n,
                                size_t m);
void selfjoin_kernel(VEDAdeviceptr T, VEDAdeviceptr P, size_t n, size_t m);
void abjoin_kernel(VEDAdeviceptr T1, VEDAdeviceptr T2, VEDAdeviceptr P, size_t n1, size_t n2,
                   size_t m);

}
