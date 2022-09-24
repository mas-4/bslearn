//
// Created by mas on 9/22/22.
//

#ifdef USE_MKL
#include <mkl.h>
#else
#include <stdlib.h>
#endif

#include "common.h"
#include "constants.h"
#define BS_LOGGER_H_IMPL
#include "logger.h"

#ifdef USE_MKL
int get_rng(double *arr, size_t n)
{
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
    return 0;
}
#else // USE_MKL
int get_rng(double *arr, size_t n)
{
    double min = 0.0;
    double max = 1.0;
    double range = max - min;
    double div = RAND_MAX / range;
    for (size_t i = 0; i < n; i++)
    {
        arr[i] = min + (rand() / div);
    }
    return 0;
}
#endif

double sigmoid_prime(double d)
{
    return 0;
}
