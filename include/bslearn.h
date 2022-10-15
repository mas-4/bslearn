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
    size_t output_size;
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
    size_t cap_layers;
    char *activation;
    char *loss;
    double (*activation_func)(double);
    double (*activation_func_prime)(double);
    double (*loss_func)(const double*, const double*, size_t);
    int loaded;
    size_t n_epochs;
    double learning_rate;
} LayerDenseNetwork;

int init_network(
        LayerDenseNetwork *network,
        size_t n_inputs, size_t n_nodes,
        char *activation, char *loss,
        size_t n_epochs, double learning_rate);
int add_layer(LayerDenseNetwork *network, size_t n_nodes);
int load_network(LayerDenseNetwork *network, const char *filename);
int save_network(LayerDenseNetwork *network, const char *filename);
int free_network(LayerDenseNetwork *network);
int predict(LayerDenseNetwork *network, const double *inputs, double *outputs);
int fit(LayerDenseNetwork *network, const double *x_train, const double *y_train, size_t n_samples);

// bs_sigmoid
double bs_sigmoid(double d);
double bs_sigmoid_p(double d);
// bs_relu
double bs_relu(double d);
double bs_relu_p(double d);
// leaky bs_relu
double bs_leaky_relu(double d);
double bs_leaky_relu_p(double d);
// tanh
double bs_tanh(double d);
double bs_tanh_p(double d);

// bs_mse
double bs_mse(const double *y, const double *y_hat, size_t n);
// bs_mae
double bs_mae(const double *y, const double *y_hat, size_t n);
// bs_crossentropy
double bs_crossentropy(const double *y, const double *y_hat, size_t n);

#endif //BLUESKY_LEARN_LIBRARY_H
