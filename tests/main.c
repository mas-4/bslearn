//
// Created by mas on 9/22/22.
//
#include <stdio.h>
#include "bslearn.h"
#include "logger.h"

#define RED   "\x1B[31m"
#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"

#define CHECK_ERROR(x, msg) if (x != 0) { printf("\t%s[FAIL] %s%s\n", RED, msg, RESET); return 1; }

int test_add_layer()
{
    int success = 0;
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 5), "Failed to initialize network");
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 10), "Failed to add layer to network")
    }
    for (size_t i = 0; i < network.num_layers; i++)
    {
        for (size_t j = 0; j < network.layers[i].nodes; j++)
        {
            if (network.layers[i].weights[j] != 0.0)
            {
                success = 1;
                break;
            }
        }
        if (success == 1)
        {
            break;
        }
    }
    CHECK_ERROR(free_network(&network), "Failed to free network");
    return 0;
}

int test_save_load_network()
{
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 5), "Failed to initialize network");
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 10), "Failed to add layer to network")
    }
    CHECK_ERROR(save_network(&network, "test_save_load_network"), "Failed to save network")
    LayerDenseNetwork network2 = {0};
    CHECK_ERROR(load_network(&network2, "test_save_load_network"), "Failed to load network")
    for (size_t i = 0; i < network.num_layers; i++)
    {
        if (network.layers[i].nodes != network2.layers[i].nodes)
        {
            CHECK_ERROR(network.layers[i].nodes != network2.layers[i].nodes, "Number of nodes does not match.")
        }
        for (size_t j = 0; j < network.layers[i].nodes; j++)
        {
            if (network.layers[i].weights[j] != network2.layers[i].weights[j])
            {
                CHECK_ERROR(network.layers[i].weights[j] != network2.layers[i].weights[j], "Weights mismatch.")
            }
        }
        for (size_t j = 0; j < network.layers[i].nodes; j++)
        {
            if (network.layers[i].biases[j] != network2.layers[i].biases[j])
            {
                if (network.layers[i].biases[j] != network2.layers[i].biases[j])
                {
                    printf("%f %f", network.layers[i].biases[j], network2.layers[i].biases[j]);
                }
                CHECK_ERROR(network.layers[i].biases[j] != network2.layers[i].biases[j], "Biases mismatch.")
            }
        }
    }
    CHECK_ERROR(free_network(&network), "Failed to free network")
    CHECK_ERROR(free_network(&network2), "Failed to free network")
    CHECK_ERROR(remove("test_save_load_network"), "Failed to remove file")
    return 0;
}

int test_evaluate_errors()
{
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 10), "Failed to initialize network");
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 100), "Failed to add layer to network")
    }
    double input[4] = {1, 2, 3, 4};
    double output[5] = {0};
    CHECK_ERROR(evaluate(&network, input, output, relu), "Failed to evaluate network")
    CHECK_ERROR(free_network(&network), "Failed to free network")
    return 0;
}

int run_test(int (*test)(), const char *name)
{
    printf("[TEST] %s...\n", name);
    int result = test();
    if (result == 0)
    {
        printf("\t%s[OKAY]%s\n", GREEN, RESET);
    }
    return result;
}

int main(void)
{
    set_logging_level(BS_LOG_LEVEL_DEBUG);
    int (*tests[]) () = {test_add_layer, test_save_load_network, test_evaluate_errors};
    const char *names[] = {"test_add_layer", "test_save_load_network", "test_evaluate_errors"};t

    int result = 0;
    for (int i = 0; i < sizeof(tests) / sizeof(tests[0]); i++)
    {
        result += run_test(tests[i], names[i]);
    }
    char *color = result == 0 ? GREEN : RED;
    printf("%s[%d TESTS FAILED]%s\n", color, result, RESET);

    return result;
}