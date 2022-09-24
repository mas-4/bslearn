//
// Created by mas on 9/22/22.
//

#ifndef BSLEARN_TESTS_COMMON_H
#define BSLEARN_TESTS_COMMON_H
#include <stddef.h>

int get_rng(double *arr, size_t n);

// A * b + c
int matmul(const double *a, const double *b, const double *c, size_t a_rows, size_t a_cols, size_t b_cols, double *out);
int matmul_activate(
        const double *a, const double *b, const double *c,
        size_t a_rows, size_t a_cols, size_t b_cols,
        double (*activate)(double),
        double *out
);


#endif //BSLEARN_TESTS_COMMON_H
