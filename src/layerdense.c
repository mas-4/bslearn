//
// Created by mas on 9/22/22.
//
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bslearn.h"
#include "constants.h"
#include "common.h"
#include "logger.h"

int alloc_layers(LayerDenseNetwork *network)
{
    if (network->layers == NULL)
    {
        network->layers = malloc(sizeof(Layer) * LAYERDENSE_DEF_LAYERS);
        if (network->layers == NULL)
        {
            log_error("alloc_layers: Failed to allocate memory for layers");
            return 1;
        }
        network->cap_layers = LAYERDENSE_DEF_LAYERS;
    }
    return 0;

}

double (*get_activation_func(char *activation))(double)
{
    if (strcmp(activation, "bs_sigmoid") == 0)
    {
        return bs_sigmoid;
    }
    else if (strcmp(activation, "bs_relu") == 0)
    {
        return bs_relu;
    }
    else if (strcmp(activation, "bs_leaky_relu") == 0)
    {
        return bs_leaky_relu;
    }
    else if (strcmp(activation, "tanh") == 0)
    {
        return bs_tanh;
    }
    else
    {
        log_error("get_activation_func: Unknown activation function");
        return NULL;
    }
}

double (*get_activation_function_prime(char *activation))(double)
{
    if (strcmp(activation, "bs_sigmoid") == 0)
    {
        return bs_sigmoid_p;
    }
    else if (strcmp(activation, "bs_relu") == 0)
    {
        return bs_relu_p;
    }
    else if (strcmp(activation, "bs_leaky_relu") == 0)
    {
        return bs_leaky_relu_p;
    }
    else if (strcmp(activation, "bs_tanh") == 0)
    {
        return bs_tanh_p;
    }
    else
    {
        log_error("get_activation_function_prime: Unknown activation function");
        return NULL;
    }
}

double (*get_loss_func(char *loss))(const double *, const double *, size_t)
{
    if (strcmp(loss, "bs_mse") == 0)
    {
        return bs_mse;
    }
    else if (strcmp(loss, "bs_mae") == 0)
    {
        return bs_mae;
    }
    else if (strcmp(loss, "bs_crossentropy") == 0)
    {
        return bs_crossentropy;
    }
    else
    {
        log_error("get_loss_func: Unknown loss function");
        return NULL;
    }
}

int init_network(
        LayerDenseNetwork *network,
        size_t n_inputs,
        size_t n_nodes,
        char *activation,
        char *loss,
        size_t n_epochs,
        double learning_rate,
        size_t batch_size
)
{
    if (alloc_layers(network) != 0)
    {
        return 1;
    }
    network->loaded = 0; // if loaded from saved file, must free loss/activation
    // set activation function
    network->activation = activation;
    network->activation_func = get_activation_func(activation);
    network->activation_func_prime = get_activation_function_prime(activation);
    // set loss function
    network->loss = loss;
    network->loss_func = get_loss_func(loss);
    network->num_layers = 1;

    // set epochs and learning rate
    network->n_epochs = n_epochs;
    network->learning_rate = learning_rate;
    network->batch_size = batch_size;

    // Allocate memory for the first layer
    network->layers[0].idx = 0;
    network->layers[0].n_nodes = n_nodes;
    network->layers[0].n_inputs = n_inputs;
    network->layers[0].weights = malloc(sizeof(double) * n_inputs * n_nodes);
    network->layers[0].biases = malloc(sizeof(double) * n_nodes);

    network->layers[0].output = NULL;
    network->layers[0].output_sz = 0;

    get_rng(network->layers[0].weights, n_inputs * n_nodes);
    get_rng(network->layers[0].biases, n_nodes);
    return 0;
}

int add_layer(LayerDenseNetwork *network, size_t n_nodes)
{
    if (network->layers == NULL)
    {
        log_error("add_layer: Network not initialized, no first input layer");
        return 1;
    }
    if (network->num_layers >= network->cap_layers)
    {
        network->layers = realloc(network->layers, sizeof(Layer) * network->cap_layers * 2);
        network->cap_layers *= 2;
    }

    // size
    network->layers[network->num_layers].idx = network->num_layers - 1;
    network->layers[network->num_layers].n_nodes = n_nodes;
    size_t prev_nodes = network->layers[network->num_layers - 1].n_nodes;
    network->layers[network->num_layers].n_inputs = prev_nodes;

    // weights
    network->layers[network->num_layers].weights = malloc(sizeof(double) * n_nodes * prev_nodes);
    get_rng(network->layers[network->num_layers].weights, n_nodes * prev_nodes);

    // biases
    network->layers[network->num_layers].biases = malloc(sizeof(double) * n_nodes);
    get_rng(network->layers[network->num_layers].biases, n_nodes);

    // outputs
    network->layers[network->num_layers].output = NULL;
    network->layers[network->num_layers].output_sz = 0;
    network->num_layers++;
    return 0;
}

int save_network(LayerDenseNetwork *network, const char *filename)
{
    if (network == NULL || filename == NULL)
    {
        log_error("save_network: Invalid arguments");
        return 1;
    }
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        log_error("save_network: Failed to open file");
        perror("save_network");
        return 1;
    }

    // layers
    fwrite(&network->num_layers, sizeof(size_t), 1, fp);
    for (size_t i = 0; i < network->num_layers; i++)
    {
        size_t nodes = network->layers[i].n_nodes;
        size_t prev_nodes = network->layers[i].n_inputs;
        fwrite(&network->layers[i].n_nodes, sizeof(size_t), 1, fp);
        fwrite(&network->layers[i].n_inputs, sizeof(size_t), 1, fp);
        fwrite(network->layers[i].weights, sizeof(double), nodes * prev_nodes, fp);
        fwrite(network->layers[i].biases, sizeof(double), nodes, fp);
    }

    // scalars
    fwrite(&network->n_epochs, sizeof(size_t), 1, fp);
    fwrite(&network->learning_rate, sizeof(double), 1, fp);
    fwrite(&network->batch_size, sizeof(size_t), 1, fp);

    // strings
    size_t loss_len = strlen(network->loss);
    size_t activation_len = strlen(network->activation);
    fwrite(&activation_len, sizeof(size_t), 1, fp);
    fwrite(network->activation, sizeof(char), activation_len, fp);
    fwrite(&loss_len, sizeof(size_t), 1, fp);
    fwrite(network->loss, sizeof(char), loss_len, fp);
    fclose(fp);
    return 0;
}

int load_network(LayerDenseNetwork *network, const char *filename)
{
    if (network == NULL || filename == NULL)
    {
        log_error("load_network: Invalid arguments");
        return 1;
    }
    network->loaded = 1;
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        log_error("load_network: Failed to open file");
        perror("load_network");
        return 1;
    }

    // layers
    size_t num_layers;
    fread(&num_layers, sizeof(size_t), 1, fp);
    network->num_layers = num_layers;
    network->layers = malloc(sizeof(Layer) * num_layers);
    network->cap_layers = num_layers;
    for (size_t i = 0; i < num_layers; i++)
    {
        size_t nodes;
        size_t prev_nodes;
        fread(&nodes, sizeof(size_t), 1, fp);
        fread(&prev_nodes, sizeof(size_t), 1, fp);
        network->layers[i].idx = i;
        network->layers[i].n_nodes = nodes;
        network->layers[i].n_inputs = prev_nodes;
        network->layers[i].weights = malloc(sizeof(double) * nodes * prev_nodes);
        network->layers[i].biases = malloc(sizeof(double) * nodes);
        fread(network->layers[i].weights, sizeof(double), nodes * prev_nodes, fp);
        fread(network->layers[i].biases, sizeof(double), nodes, fp);
        network->layers[i].output = NULL;
        network->layers[i].output_sz = 0;
    }
    // scalars
    fread(&network->n_epochs, sizeof(size_t), 1, fp);
    fread(&network->learning_rate, sizeof(double), 1, fp);
    fread(&network->batch_size, sizeof(size_t), 1, fp);

    // strings
    size_t activation_len;
    fread(&activation_len, sizeof(size_t), 1, fp);
    network->activation = malloc(sizeof(char) * (activation_len + 1));
    fread(network->activation, sizeof(char), activation_len, fp);
    network->activation[activation_len] = '\0';
    network->activation_func = get_activation_func(network->activation);
    network->activation_func_prime = get_activation_function_prime(network->activation);
    size_t loss_len;
    fread(&loss_len, sizeof(size_t), 1, fp);
    network->loss = malloc(sizeof(char) * (loss_len + 1));
    fread(network->loss, sizeof(char), loss_len, fp);
    network->loss[loss_len] = '\0';
    fclose(fp);
    return 0;
}

int free_network(LayerDenseNetwork *network)
{
    if (network == NULL)
    {
        log_error("free_network: Invalid arguments");
        return 1;
    }
    for (size_t i = 0; i < network->num_layers; i++)
    {
        free(network->layers[i].weights);
        free(network->layers[i].biases);
        if (network->layers[i].output != NULL)
        {
            free(network->layers[i].output);
        }
    }
    free(network->layers);
    // free activation and loss if allocated
    if (network->loaded == 1)
    {
        free(network->activation);
        free(network->loss);
    }
    return 0;
}

int feed_forward(Layer *layer, const double *inputs, double *outputs, size_t batch_size, double (*activate)(double))
{
    if (layer == NULL || inputs == NULL || outputs == NULL)
    {
        log_error("feed_forward: Invalid arguments");
        return 1;
    }
    // n_nodes  = m
    // inputs = k
    // batch  = n
    matmul_activate(layer->weights, // A  m x k
                    inputs,         // B  k x n
                    layer->biases,  // C  m x 1
                    outputs,        // m x n
                    layer->n_nodes,  // m
                    batch_size,      // n
                    layer->n_inputs, // k
                    activate);

    return 0;
}

int alloc_output(Layer *layer, size_t batch_sz)
{
    if (layer == NULL)
    {
        log_error("alloc_output: Invalid arguments");
        return 1;
    }
    else if (layer->output != NULL)
    {
        free(layer->output);
        layer->output = NULL;
    }
    layer->output = malloc(sizeof(double) * layer->n_nodes * batch_sz);
    memset(layer->output, 0, sizeof(double) * layer->n_nodes * batch_sz); // this is useless, temporary to hide valgrind
    layer->output_sz = batch_sz;
    return 0;
}

int predict(LayerDenseNetwork *network, const double *inputs, double *outputs, size_t batch_sz)
{
    if (network == NULL || inputs == NULL || outputs == NULL || batch_sz == 0)
    {
        log_error("predict: Invalid arguments");
        return 1;
    }
    alloc_output(&network->layers[0], batch_sz);
    feed_forward(&network->layers[0], inputs, network->layers[0].output, batch_sz, network->activation_func);
    for (size_t i = 1; i < network->num_layers; i++)
    {
        alloc_output(&network->layers[i], batch_sz);
        feed_forward(&network->layers[i],
                     network->layers[i - 1].output, // previous layer's output
                     network->layers[i].output,    // current layer's output
                     batch_sz,
                     network->activation_func);
    }
    memcpy(outputs,
           network->layers[network->num_layers - 1].output,
           sizeof(double) * network->layers[network->num_layers-1].output_sz);
    return 0;
}

/*

int backprop(LayerDenseNetwork *network)
{
    if (network == NULL)
    {
        log_error("backpropagation: Invalid arguments");
        return 1;
    }
    return 0;
}

int fit(LayerDenseNetwork *network, const double *x_train, const double *y_train, size_t n_samples)
{
    log_fatal("fit: Not implemented");
    return 1;
    if (network == NULL || x_train == NULL || y_train == NULL)
    {
        log_error("fit: Invalid arguments");
        return 1;
    }
    double *outputs = malloc(sizeof(double) * network->layers[network->num_layers - 1].n_nodes);
    for (size_t i = 0; i < n_samples; i++)
    {
        predict(network, &x_train[i * network->layers[0].n_inputs], outputs);
    }
    backprop(network);
    return 0;
}
*/