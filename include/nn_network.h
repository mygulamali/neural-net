#pragma once

#include <stdint.h>

#include <gsl/gsl_matrix.h>

typedef struct {
    uintmax_t *layers;
    gsl_matrix **biases;
    gsl_matrix **weights;
} nn_network;

nn_network * nn_network_create(const size_t n, const uintmax_t *layers);
