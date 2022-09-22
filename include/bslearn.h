/*
 * All errors are returned as a positive number for testing purposes.
 * Success is zero.
 */
#ifndef BLUESKY_LEARN_LIBRARY_H
#define BLUESKY_LEARN_LIBRARY_H

#include <stddef.h>

#define BLUESKY_LEARN_LIBRARY_VERSION "0.0.1"
#define BLUESKY_LEARN_LIBRARY_VERSION_MAJOR 0
#define BLUESKY_LEARN_LIBRARY_VERSION_MINOR 0
#define BLUESKY_LEARN_LIBRARY_VERSION_PATCH 1


void hello(void);


/******************************************************
 * Layer Dense Neural Networks
 ******************************************************/

typedef struct {
    double *weights;
    double *biases;
    size_t nodes;
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
    size_t cap_layers;
} LayerDenseNetwork;

int add_layer(LayerDenseNetwork *network, size_t n_nodes);
int load_network(LayerDenseNetwork *network, const char *filename);
int save_network(LayerDenseNetwork *network, const char *filename);
int free_network(LayerDenseNetwork *network);

#endif //BLUESKY_LEARN_LIBRARY_H
