//
// Created by mas on 9/22/22.
//

#ifndef BSLEARN_TESTS_COMMON_H
#define BSLEARN_TESTS_COMMON_H
#include <stddef.h>

int get_rng(double *arr, size_t n);

// Activation Functions

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


#endif //BSLEARN_TESTS_COMMON_H
