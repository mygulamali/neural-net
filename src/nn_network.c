#include "nn_network.h"

nn_network * nn_network_create(const size_t n, const uintmax_t *layers) {
    static nn_network network;
    gsl_matrix *matrix = gsl_matrix_alloc(0, 0);

    network.layers = (uintmax_t *) malloc(n * sizeof(layers[0]));
    for (size_t i = 0; i < n; i++)
	network.layers[i] = layers[i];

    network.biases = (gsl_matrix **) malloc(n * sizeof(*matrix));
    network.weights = (gsl_matrix **) malloc(n * sizeof(*matrix));
    for (size_t i = 0; i < n - 1; i++) {
	const uintmax_t layer_i = layers[i];
	const uintmax_t layer_ip = layers[i + 1];

	network.biases[i] = gsl_matrix_calloc(layer_ip, 1);
	network.weights[i] = gsl_matrix_calloc(layer_ip, layer_i);
    }

    gsl_matrix_free(matrix);
    return &network;
}
