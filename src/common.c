//
// Created by mas on 9/22/22.
//

#include <mkl.h>
#include "common.h"

int get_rng(double *rng, size_t n)
{
    VSLStreamStatePtr stream;
    int status = vslNewStream(&stream, VSL_BRNG_MT19937, 1);
    if (status != VSL_STATUS_OK)
    {
        return -1;
    }
    status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int)n, rng, 0, 1);
    if (status != VSL_STATUS_OK)
    {
        return -1;
    }
    return 0;
}
