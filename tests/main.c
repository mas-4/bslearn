//
// Created by mas on 9/22/22.
//
#include <stdio.h>
#include <string.h>
#include "bslearn.h"
#include "logger.h"
#include "common.h"

#define RED   "\x1B[31m"
#define GREEN "\x1B[32m"
#define RESET "\x1B[0m"

#define CHECK_ERROR(x, msg) if (x != 0) { printf("\t%s[FAIL] %s%s %d\n", RED, msg, RESET, x); return x; }

int test_add_layer()
{
    int success = 0;
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 5, "bs_relu", "bs_mse", 100, 0.1, 100),
                "Failed to initialize network")
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 10), "Failed to add layer to network")
    }
    for (size_t i = 0; i < network.num_layers; i++)
    {
        for (size_t j = 0; j < network.layers[i].n_nodes; j++)
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
    CHECK_ERROR(free_network(&network), "Failed to free network")
    return 0;
}

int test_save_load_network()
{
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 5, "bs_relu", "bs_mse", 100, 0.1, 100),
                "Failed to initialize network")
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 10), "Failed to add layer to network")
    }
    CHECK_ERROR(save_network(&network, "test_save_load_network"), "Failed to save network")
    LayerDenseNetwork network2 = {0};
    CHECK_ERROR(load_network(&network2, "test_save_load_network"), "Failed to load network")
    CHECK_ERROR(strcmp(network.activation, network2.activation) != 0, "Activation functions do not match")
    CHECK_ERROR(strcmp(network.loss, network2.loss) != 0, "Loss functions do not match")
    for (size_t i = 0; i < network.num_layers; i++)
    {
        if (network.layers[i].n_nodes != network2.layers[i].n_nodes)
        {
            CHECK_ERROR(network.layers[i].n_nodes != network2.layers[i].n_nodes, "Number of n_nodes does not match.")
        }
        for (size_t j = 0; j < network.layers[i].n_nodes; j++)
        {
            if (network.layers[i].weights[j] != network2.layers[i].weights[j])
            {
                CHECK_ERROR(network.layers[i].weights[j] != network2.layers[i].weights[j], "Weights mismatch.")
            }
        }
        for (size_t j = 0; j < network.layers[i].n_nodes; j++)
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

int test_matmul()
{
    size_t m = 3;
    size_t n = 3;
    double a[] = {0, 1, 2,
                  3, 4, 5,
                  6, 7, 8};
    double b[] = {1, 0, 0,
                  0, 1, 0,
                  0, 0, 1};
    double c[] = {1, 2, 3};
    double output[9] = {0};
    double answer[9] = {1, 2, 3,
                        5, 6, 7,
                        9, 10, 11};
    CHECK_ERROR(matmul_activate(a, b, c, output, m, n, n, bs_relu), "Failed to matmul")
    for (size_t i = 0; i < 3 * 3; i++)
    {
        if (output[i] != answer[i])
        {
            CHECK_ERROR(output[i] != bs_relu(answer[i]), "Matmul failed")
        }
    }
    return 0;
}

int test_predict_errors()
{
    LayerDenseNetwork network = {0};
    CHECK_ERROR(init_network(&network, 4, 10, "bs_relu", "bs_mse", 100, 0.1, 100), "Failed to initialize network")
    for (int i = 0; i < 4; i++)
    {
        CHECK_ERROR(add_layer(&network, 100), "Failed to add layer to network")
    }
    CHECK_ERROR(add_layer(&network, 10), "Failed to add layer to network")
    double input[4] = {1, 2, 3, 4};
    double output[10] = {0};
    CHECK_ERROR(predict(&network, input, output, 1), "Failed to predict network")
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
    int (*tests[]) () = {
            test_matmul,
            test_add_layer,
            test_save_load_network,
            test_predict_errors
    };
    const char *names[] = {
            "test_matmul",
            "test_add_layer",
            "test_save_load_network",
            "test_predict_errors"
    };

    int result = 0;
    for (int i = 0; i < sizeof(tests) / sizeof(tests[0]); i++)
    {
        result += run_test(tests[i], names[i]);
    }
    char *color = result == 0 ? GREEN : RED;
    printf("%s[%d TESTS FAILED]%s\n", color, result, RESET);

    return result;
}