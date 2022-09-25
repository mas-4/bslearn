//
// Created by mas on 9/22/22.
//

#ifdef USE_MKL
#include <mkl.h>
#else
#include <stdlib.h>
#include <math.h>
#endif

#include "common.h"
#include "constants.h"

#define BS_LOGGER_H_IMPL

#include "logger.h"

// <editor-fold desc="randoms">
int get_rng(double *arr, size_t n)
{
#ifdef USE_MKL
    VSLStreamStatePtr stream;
    int status = vslNewStream(&stream, VSL_BRNG_MT19937, BSLEARN_SEED);
    if (status != VSL_STATUS_OK)
    {
        log_error("get_rng: Failed to create stream");
        return status;
    }
    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int)n, arr, 0, 1);
    if (status != VSL_STATUS_OK)
    {
        log_error("get_rng: Failed to generate random numbers");
        return status;
    }
    // free VSLStreamStatePtr
    vslDeleteStream(&stream);
#else // USE_MKL
    double min = 0.0;
    double max = 1.0;
    double range = max - min;
    double div = RAND_MAX / range;
    for (size_t i = 0; i < n; i++)
    {
        arr[i] = min + (rand() / div);
    }
#endif // USE_MKL
    return 0;
}
// </editor-fold>

// <editor-fold desc="matmuls">
int matmul(const double *a, const double *b, const double *c, size_t a_rows, size_t a_cols, size_t b_cols, double *out)
{
    for (size_t i = 0; i < a_rows; i++)
    {
        for (size_t j = 0; j < b_cols; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < a_cols; k++)
            {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = sum + c[j];
        }
    }
    return 0;

}

int matmul_activate(
        const double *a, const double *b, const double *c,
        size_t a_rows, size_t a_cols, size_t b_cols,
        double (*activate)(double), double *out
)
{
    for (size_t i = 0; i < a_rows; i++)
    {
        for (size_t j = 0; j < b_cols; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < a_cols; k++)
            {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = activate(sum + c[j]);
        }
    }
    return 0;

}
// </editor-fold>

// <editor-fold desc="activations">
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime(double d)
{
    return 0;
}

double relu(double d)
{
    return d > 0 ? d : 0;
}
double relu_prime(double d)
{
    return d > 0 ? 1 : 0;
}

double leaky_relu(double d)
{
    return d > 0 ? d : 0.01 * d;
}

double leaky_relu_prime(double d)
{
    return d > 0 ? 1 : 0.01;
}

double tanh(double d)
{
    return (exp(d) - exp(-d)) / (exp(d) + exp(-d));
}

double tanh_prime(double d)
{
    return 1 - pow(tanh(d), 2);
}
// </editor-fold>


// <editor-fold desc="losses">
double mse(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += pow(y[i] - y_hat[i], 2);
    }
    return sum / (double)n;
}

double mae(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += fabs(y[i] - y_hat[i]);
    }
    return sum / (double)n;
}

double binary_crossentropy(const double *y, const double *y_hat, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum += y[i] * log(y_hat[i]);
    }
    return -sum / (double)n;
}
// </editor-fold>