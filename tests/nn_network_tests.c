#include "nn_network_tests.h"

void test_nn_network_create_destroy(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);

    nn_network *network = nn_network_create(n, layers);

    assert_int_equal(network->size, n);

    for (size_t i = 0; i < n; i++)
	assert_int_equal(network->layers[i], layers[i]);

    gsl_matrix *zeros;

    for (size_t i = 0; i < n - 1; i++) {
	gsl_matrix *bias = network->biases[i];
	assert_int_equal(bias->size1, layers[i + 1]);
	assert_int_equal(bias->size2, 1);

	zeros = gsl_matrix_calloc(bias->size1, 1);
	assert_gsl_matrix_equal(bias, zeros, EPSILON);
	gsl_matrix_free(zeros);

	gsl_matrix *weight = network->weights[i];
	assert_int_equal(weight->size1, layers[i + 1]);
	assert_int_equal(weight->size2, layers[i]);

	zeros = gsl_matrix_calloc(weight->size1, weight->size2);
	assert_gsl_matrix_equal(weight, zeros, EPSILON);
	gsl_matrix_free(zeros);
    }

    nn_network_destroy(network);

    (void) state;
}

void test_nn_network_set_get_biases(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);
    const size_t sizeof_matrix = nn_utils_sizeof_gsl_matrix();
    nn_network *network = nn_network_create(n, layers);

    gsl_matrix **old_biases = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);
    gsl_matrix **new_biases = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);

    for (size_t i = 0; i < n - 1; i++) {
    	old_biases[i] = gsl_matrix_calloc(layers[i + 1], 1);
    	for (size_t j = 0; j < old_biases[i]->size1; j++)
    	    gsl_matrix_set(old_biases[i], j, 0, (1.0 + i) * (1.0 + j));
    }

    nn_network_set_biases(network, old_biases);
    nn_network_get_biases(network, new_biases);

    for (size_t i = 0; i < n - 1; i++)
	assert_gsl_matrix_equal(old_biases[i], new_biases[i], EPSILON);

    for (size_t i = 0; i < n - 1; i++) {
	gsl_matrix_free(old_biases[i]);
	gsl_matrix_free(new_biases[i]);
    }
    free(old_biases);
    free(new_biases);
    nn_network_destroy(network);

    (void) state;
}

void test_nn_network_set_get_weights(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);
    const size_t sizeof_matrix = nn_utils_sizeof_gsl_matrix();
    nn_network *network = nn_network_create(n, layers);

    gsl_matrix **old_weights = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);
    gsl_matrix **new_weights = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);

    for (size_t i = 0; i < n - 1; i++) {
	const uintmax_t layer_i = layers[i];
	const uintmax_t layer_ip = layers[i + 1];

	old_weights[i] = gsl_matrix_calloc(layer_ip, layer_i);
    	for (size_t j = 0; j < old_weights[i]->size1; j++)
	    for (size_t k = 0; k < old_weights[i]->size2; k++)
		gsl_matrix_set(old_weights[i], j, k, (1.0 + j) * (1.0 + k));
    }

    nn_network_set_weights(network, old_weights);
    nn_network_get_weights(network, new_weights);

    for (size_t i = 0; i < n - 1; i++)
	assert_gsl_matrix_equal(old_weights[i], new_weights[i], EPSILON);

    for (size_t i = 0; i < n - 1; i++) {
	gsl_matrix_free(old_weights[i]);
	gsl_matrix_free(new_weights[i]);
    }
    free(old_weights);
    free(new_weights);
    nn_network_destroy(network);
    (void) state;
}
