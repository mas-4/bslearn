//
// Created by mas on 9/22/22.
//
#include <stddef.h>
#include <malloc.h>
#include "bslearn.h"
#include "constants.h"
#include "common.h"

int add_layer(LayerDenseNetwork *network, size_t n_nodes)
{
    if (network == NULL)
    {
        return 1;
    }
    if (network->layers == NULL)
    {
        network->layers = malloc(sizeof(Layer) * LAYERDENSE_DEF_LAYERS);
        network->cap_layers = LAYERDENSE_DEF_LAYERS;
    }
    if (network->num_layers >= network->cap_layers)
    {
        network->layers = realloc(network->layers, sizeof(Layer) * network->cap_layers * 2);
        network->cap_layers *= 2;
    }
    network->layers[network->num_layers].nodes = n_nodes;
    network->layers[network->num_layers].weights = malloc(sizeof(double) * n_nodes);
    network->layers[network->num_layers].biases = malloc(sizeof(double) * n_nodes);
    get_rng(network->layers[network->num_layers].weights, n_nodes);
    get_rng(network->layers[network->num_layers].biases, n_nodes);
    return 0;
}

int save_network(LayerDenseNetwork *network, const char *filename)
{
    if (network == NULL || filename == NULL)
    {
        return 1;
    }
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        return 1;
    }
    fwrite(&network->num_layers, sizeof(size_t), 1, fp);
    for (size_t i = 0; i < network->num_layers; i++)
    {
        fwrite(&network->layers[i].nodes, sizeof(size_t), 1, fp);
        fwrite(network->layers[i].weights, sizeof(double), network->layers[i].nodes, fp);
        fwrite(network->layers[i].biases, sizeof(double), network->layers[i].nodes, fp);
    }
    fclose(fp);
    return 0;
}

int load_network(LayerDenseNetwork *network, const char *filename)
{
    if (network == NULL || filename == NULL)
    {
        return 1;
    }
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        return 1;
    }
    size_t num_layers;
    fread(&num_layers, sizeof(size_t), 1, fp);
    for (size_t i = 0; i < num_layers; i++)
    {
        size_t nodes;
        fread(&nodes, sizeof(size_t), 1, fp);
        add_layer(network, nodes);
        fread(network->layers[i].weights, sizeof(double), nodes, fp);
        fread(network->layers[i].biases, sizeof(double), nodes, fp);
    }
    fclose(fp);
    return 0;
}

int free_network(LayerDenseNetwork *network)
{
    if (network == NULL)
    {
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