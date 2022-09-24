//
// Created by mas on 9/22/22.
//
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
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

int init_network(LayerDenseNetwork *network, size_t n_inputs, size_t n_nodes)
{
    if (alloc_layers(network) != 0)
    {
        return 1;
    }
    network->num_layers = 1;
    network->layers[0].nodes = n_nodes;
    network->layers[0].prev_nodes = n_inputs;
    network->layers[0].weights = malloc(sizeof(double) * n_inputs * n_nodes);
    network->layers[0].biases = malloc(sizeof(double) * n_inputs * n_nodes);
    get_rng(network->layers[0].weights, n_inputs * n_nodes);
    get_rng(network->layers[0].biases, n_inputs * n_nodes);
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
    network->layers[network->num_layers].nodes = n_nodes;
    size_t prev_nodes = network->layers[network->num_layers - 1].nodes;
    network->layers[network->num_layers].prev_nodes = prev_nodes;
    network->layers[network->num_layers].weights = malloc(sizeof(double) * n_nodes * prev_nodes);
    network->layers[network->num_layers].biases = malloc(sizeof(double) * n_nodes * prev_nodes);
    get_rng(network->layers[network->num_layers].weights, n_nodes * prev_nodes);
    get_rng(network->layers[network->num_layers].biases, n_nodes * prev_nodes);
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
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        log_error("save_network: Failed to open file");
        perror("save_network");
        return 1;
    }
    fwrite(&network->num_layers, sizeof(size_t), 1, fp);
    for (size_t i = 0; i < network->num_layers; i++)
    {
        size_t size = network->layers[i].nodes * network->layers[i].prev_nodes;
        fwrite(&network->layers[i].nodes, sizeof(size_t), 1, fp);
        fwrite(&network->layers[i].prev_nodes, sizeof(size_t), 1, fp);
        fwrite(network->layers[i].weights, sizeof(double), size, fp);
        fwrite(network->layers[i].biases, sizeof(double), size, fp);
    }
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
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        log_error("load_network: Failed to open file");
        perror("load_network");
        return 1;
    }
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
        network->layers[i].nodes = nodes;
        network->layers[i].prev_nodes = prev_nodes;
        network->layers[i].weights = malloc(sizeof(double) * nodes * prev_nodes);
        network->layers[i].biases = malloc(sizeof(double) * nodes * prev_nodes);
        fread(network->layers[i].weights, sizeof(double), nodes * prev_nodes, fp);
        fread(network->layers[i].biases, sizeof(double), nodes * prev_nodes, fp);
    }
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
    }
    free(network->layers);
    return 0;
}