#pragma once

#include <stdint.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "nn_math.h"
#include "nn_utils.h"

typedef struct {
    size_t size;
    uintmax_t *layers;
    gsl_vector **biases;
    gsl_matrix **weights;
} nn_network;

nn_network * nn_network_create(const size_t n, const uintmax_t *layers);
void nn_network_destroy(nn_network *network);
void nn_network_set_biases(nn_network *network, gsl_vector **biases);
void nn_network_get_biases(nn_network *network, gsl_vector **biases);
void nn_network_set_weights(nn_network *network, gsl_matrix **weights);
void nn_network_get_weights(nn_network *network, gsl_matrix **weights);
gsl_vector * nn_network_ff(nn_network *network, gsl_vector *x);
