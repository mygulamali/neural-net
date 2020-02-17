#include "nn_network.h"

nn_network * nn_network_create(const size_t n, const uintmax_t *layers) {
    static nn_network network;
    const size_t sizeof_matrix = nn_utils_sizeof_gsl_matrix();

    network.size = n;

    network.layers = (uintmax_t *) malloc(n * sizeof(layers[0]));
    for (size_t i = 0; i < n; i++)
	network.layers[i] = layers[i];

    network.biases = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);
    network.weights = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);
    for (size_t i = 0; i < n - 1; i++) {
	const uintmax_t layer_i = layers[i];
	const uintmax_t layer_ip = layers[i + 1];

	network.biases[i] = gsl_matrix_calloc(layer_ip, 1);
	network.weights[i] = gsl_matrix_calloc(layer_ip, layer_i);
    }

    return &network;
}

void nn_network_destroy(nn_network *network) {
    for (size_t i = 0; i < (network->size - 1); i++) {
	gsl_matrix_free(network->biases[i]);
	gsl_matrix_free(network->weights[i]);
    }

    free(network->layers);
    free(network->biases);
    free(network->weights);
}

void nn_network_set_biases(nn_network *network, gsl_matrix **biases) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++)
	gsl_matrix_memcpy(network->biases[i], biases[i]);
}

void nn_network_get_biases(nn_network *network, gsl_matrix **biases) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++) {
	biases[i] = gsl_matrix_alloc(network->biases[i]->size1, 1);
	gsl_matrix_memcpy(biases[i], network->biases[i]);
    }
}
