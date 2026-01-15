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
void selfjoin(VEDAdeviceptr T, VEDAdeviceptr P, size_t n, size_t m);
void abjoin(VEDAdeviceptr T1, VEDAdeviceptr T2, VEDAdeviceptr P, size_t n1, size_t n2, size_t m);

}
