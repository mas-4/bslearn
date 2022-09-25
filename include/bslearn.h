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
#define BLUESKY_LEARN_LIBRARY_VERSION_BUILD 0


/******************************************************
 * Layer Dense Neural Networks
 ******************************************************/

typedef struct {
    double *weights;
    double *biases;
    double *output;
    size_t nodes;
    size_t prev_nodes;
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
    size_t cap_layers;
    char *activation;
    char *loss;
    double (*activation_func)(double);
    double (*activation_func_prime)(double);
    double (*loss_func)(double*, double*, size_t);
    int loaded;
} LayerDenseNetwork;

int init_network(LayerDenseNetwork *network, size_t n_inputs, size_t n_nodes, char *activation, char *loss);
int add_layer(LayerDenseNetwork *network, size_t n_nodes);
int load_network(LayerDenseNetwork *network, const char *filename);
int save_network(LayerDenseNetwork *network, const char *filename);
int free_network(LayerDenseNetwork *network);
int predict(LayerDenseNetwork *network, double *inputs, double *outputs);

// sigmoid
double sigmoid(double d);
double sigmoid_prime(double d);
// relu
double relu(double d);
double relu_prime(double d);
// leaky relu
double leaky_relu(double d);
double leaky_relu_prime(double d);
// tanh
double tanh(double d);
double tanh_prime(double d);

// mse
double mse(double *y, double *y_hat, size_t n);
// mae
double mae(double *y, double *y_hat, size_t n);
// binary_crossentropy
double binary_crossentropy(double *y, double *y_hat, size_t n);

#endif //BLUESKY_LEARN_LIBRARY_H
