#include "nn_network.h"

nn_network * nn_network_create(const size_t n, const uintmax_t *layers) {
    static nn_network network;
    const size_t sizeof_matrix = nn_utils_sizeof_gsl_matrix();
    const size_t sizeof_vector = nn_utils_sizeof_gsl_vector();

    network.size = n;

    network.layers = (uintmax_t *) malloc(n * sizeof(layers[0]));
    for (size_t i = 0; i < n; i++)
	network.layers[i] = layers[i];

    network.biases = (gsl_vector **) malloc((n - 1) * sizeof_vector);
    network.weights = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);
    for (size_t i = 0; i < n - 1; i++) {
	const uintmax_t layer_i = layers[i];
	const uintmax_t layer_ip = layers[i + 1];

	network.biases[i] = gsl_vector_calloc(layer_ip);
	network.weights[i] = gsl_matrix_calloc(layer_ip, layer_i);
    }

    return &network;
}

void nn_network_destroy(nn_network *network) {
    for (size_t i = 0; i < (network->size - 1); i++) {
	gsl_vector_free(network->biases[i]);
	gsl_matrix_free(network->weights[i]);
    }

    free(network->layers);
    free(network->biases);
    free(network->weights);
}

void nn_network_set_biases(nn_network *network, gsl_vector **biases) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++)
	gsl_vector_memcpy(network->biases[i], biases[i]);
}

void nn_network_get_biases(nn_network *network, gsl_vector **biases) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++) {
	biases[i] = gsl_vector_alloc(network->biases[i]->size);
	gsl_vector_memcpy(biases[i], network->biases[i]);
    }
}

void nn_network_set_weights(nn_network *network, gsl_matrix **weights) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++)
	gsl_matrix_memcpy(network->weights[i], weights[i]);
}

void nn_network_get_weights(nn_network *network, gsl_matrix **weights) {
    const size_t n = network->size;

    for (size_t i = 0; i < n - 1; i++) {
	weights[i] = gsl_matrix_alloc(
	    network->weights[i]->size1,
	    network->weights[i]->size2
	);
	gsl_matrix_memcpy(weights[i], network->weights[i]);
    }
}

gsl_vector * nn_network_ff(nn_network *network, gsl_vector *x) {
    const size_t n = network->size;
    gsl_vector *x_tmp;
    gsl_vector *y_tmp;

    x_tmp = gsl_vector_alloc(x->size);
    gsl_vector_memcpy(x_tmp, x);
    for (size_t i = 0; i < n - 1; i++) {
	y_tmp = gsl_vector_alloc(network->biases[i]->size);
	gsl_vector_memcpy(y_tmp, network->biases[i]);

	gsl_blas_dgemv(
	    CblasNoTrans,
	    1.0, network->weights[i], x_tmp,
	    1.0, y_tmp
	);

	gsl_vector_free(x_tmp);
	x_tmp = nn_sigmoid_v(y_tmp);
	gsl_vector_free(y_tmp);
    }

    static gsl_vector *y;
    y = gsl_vector_alloc(x_tmp->size);
    gsl_vector_memcpy(y, x_tmp);
    gsl_vector_free(x_tmp);

    return y;
}
