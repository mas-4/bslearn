//
// Created by mas on 9/22/22.
//

#ifdef USE_MKL
    #define USE_LAPACK
    #include <mkl.h>
#elif USE_ACCELERATE
    #include <Accelerate/Accelerate.h>
    #define USE_LAPACK
#else
    #include <stdlib.h>
    #include <math.h>
#endif

#ifdef USE_LAPACK
#include "constants.h"
#endif

#include <errno.h>

#include "common.h"

#define BS_LOGGER_H_IMPL

#include "logger.h"
#include "bslearn.h"

int get_rng(double *arr, size_t n)
{
    int status;
#ifdef USE_MKL
    VSLStreamStatePtr stream;
    status = vslNewStream(&stream, VSL_BRNG_MT19937, 0);
    if (status != VSL_STATUS_OK) {
        log_warn("MKL RNG error");
        goto mklend;
    }
    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, arr, 0, 1);
    mklend:
    vslDeleteStream(&stream);
    if (status == VSL_STATUS_OK) {
        return status;
    }
#endif
#ifdef USE_LAPACK
    int seed = BSLEARN_SEED;
    int one = 1;
    int n_int = (int)n;
    status = dlarnv_(&one, &seed, &n_int, arr);
    if (status == 0) {
        return status;
    }
#endif
    status = 0;
    for (size_t i = 0; i < n; i++)
    {
        arr[i] = (double)rand() / (double)RAND_MAX; // NOLINT(cert-msc50-cpp)
    }
    if (errno != 0) {
        log_error("RNG error");
        status = errno;
    }
    return status;
}

// <editor-fold desc="matrix">
#ifdef USE_LAPACK
int matmul_activate(
        const double *a, // m x k
        const double *b, // k x n
        const double *c, // n-len, biases
        double *output, // m x n
        size_t m,
        size_t n,
        size_t k,
        double (*activate)(double))
{
    double alpha = 1.0, beta = 1.0;
    // copy C into every column of output
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            output[i * m + j] = c[i];
        }
    }

    cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            (int)m,
            (int)n,
            (int)k,
            alpha,
            a,
            (int)k,
            b,
            (int)n,
            beta,
            output,
            (int)n);

    if (errno != 0) {
        log_error("cblas_dgemm error");
        return -1;
    }
    for (size_t i = 0; i < m * n; i++)
    {
        output[i] = activate(output[i]);
    }
    return 0;
}

#else // USE_LAPACK

int matmul_activate(
        const double *A, // m x k mat
        const double *B, // k x n mat
        const double *c, // n-len vec
        double *output,  // m x n mat
        size_t m,
        size_t n,
        size_t k,
        double (*activate)(double))
{
    for (size_t row = 0; row < m; row++)
    {
        for (size_t col = 0; col < n; col++)
        {
            double sum = 0.0;
            for (size_t i = 0; i < k; i++)
            {
                sum += A[row * k + i] * B[i * n + col];
            }
            double res = activate(sum + c[col]);
            output[row * n + col] = res;
        }
    }
    return 0;
}

#endif // USE_LAPACK
// </editor-fold>

// <editor-fold desc="activations">
double bs_sigmoid(double d)
{
    return 1.0 / (1.0 + exp(-d));
}

double bs_sigmoid_p(double d)
{
    return bs_sigmoid(d) * (1.0 - bs_sigmoid(d));
}

double bs_relu(double d)
{
    return d > 0 ? d : 0;
}
double bs_relu_p(double d)
{
    return d > 0 ? 1 : 0;
}

double bs_leaky_relu(double d)
{
    return d > 0 ? d : 0.01 * d;
}

double bs_leaky_relu_p(double d)
{
    return d > 0 ? 1 : 0.01;
}

double bs_tanh(double d)
{
    return (exp(d) - exp(-d)) / (exp(d) + exp(-d));
}

double bs_tanh_p(double d)
{
    return 1 - pow(tanh(d), 2);
}
// </editor-fold>

// <editor-fold desc="losses">
double bs_mse(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += pow(y[i] - y_hat[i], 2);
    }
    return sum / (double)n;
}

double bs_mae(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += fabs(y[i] - y_hat[i]);
    }
    return sum / (double)n;
}

double bs_crossentropy(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += y[i] * log(y_hat[i]);
    }
    return -sum / (double)n;
}
// </editor-fold>

// <editor-fold desc="miscellaneous">
int print_matrix(const double *arr, size_t rows, size_t cols)
{
    printf("%zu x %zu matrix\n", rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%0.5f ", arr[i * cols + j]);
        }
        printf("\n");
    }
    return 0;
}
// </editor-fold>
