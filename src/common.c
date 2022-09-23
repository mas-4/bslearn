//
// Created by mas on 9/22/22.
//

#include <mkl.h>
#include "common.h"
#include "constants.h"
#include "logger.h"

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

double sigmoid_prime(double d)
{
    return 0;
}
