//
// Created by mas on 9/22/22.
//

#ifndef BSLEARN_TESTS_COMMON_H
#define BSLEARN_TESTS_COMMON_H
#include <stddef.h>

int get_rng(double *arr, size_t n);

// A * B + c
int matmul_activate(
        const double *A,
        const double *B,
        const double *c,
        double *output,
        size_t m,
        size_t n,
        size_t k,
        double (*activate)(double));

int print_matrix(const double *a, size_t rows, size_t cols);


#endif //BSLEARN_TESTS_COMMON_H
