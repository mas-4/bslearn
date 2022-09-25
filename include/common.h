//
// Created by mas on 9/22/22.
//

#ifndef BSLEARN_TESTS_COMMON_H
#define BSLEARN_TESTS_COMMON_H
#include <stddef.h>

int get_rng(double *arr, size_t n);

// A * b + c
int matvecmul(const double *a, const double *b, const double *c, size_t a_rows, size_t a_cols, double *output);
int matvecmul_activate(
        const double *a, const double *b, const double *c,
        size_t a_rows, size_t a_cols,
        double (*activate)(double), double *out
);
int print_matrix(const double *a, size_t rows, size_t cols);


#endif //BSLEARN_TESTS_COMMON_H
