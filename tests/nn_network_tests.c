#include "nn_network_tests.h"

void test_nn_network_create_destroy(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);

    nn_network *network = nn_network_create(n, layers);

    assert_int_equal(network->size, n);

    for (size_t i = 0; i < n; i++)
	assert_int_equal(network->layers[i], layers[i]);

    gsl_matrix *zeros_m;
    gsl_vector *zeros_v;

    for (size_t i = 0; i < n - 1; i++) {
	gsl_vector *bias = network->biases[i];
	assert_int_equal(bias->size, layers[i + 1]);

	zeros_v = gsl_vector_calloc(bias->size);
	assert_gsl_vector_equal(bias, zeros_v, EPSILON);
	gsl_vector_free(zeros_v);

	gsl_matrix *weight = network->weights[i];
	assert_int_equal(weight->size1, layers[i + 1]);
	assert_int_equal(weight->size2, layers[i]);

	zeros_m = gsl_matrix_calloc(weight->size1, weight->size2);
	assert_gsl_matrix_equal(weight, zeros_m, EPSILON);
	gsl_matrix_free(zeros_m);
    }

    nn_network_destroy(network);

    (void) state;
}

void test_nn_network_set_get_biases(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);
    const size_t sizeof_vector = nn_utils_sizeof_gsl_vector();
    nn_network *network = nn_network_create(n, layers);

    gsl_vector **old_biases = (gsl_vector **) malloc((n - 1) * sizeof_vector);
    gsl_vector **new_biases = (gsl_vector **) malloc((n - 1) * sizeof_vector);

    for (size_t i = 0; i < n - 1; i++) {
    	old_biases[i] = gsl_vector_calloc(layers[i + 1]);
    	for (size_t j = 0; j < old_biases[i]->size; j++)
    	    gsl_vector_set(old_biases[i], j, (1.0 + i) * (1.0 + j));
    }

    nn_network_set_biases(network, old_biases);
    nn_network_get_biases(network, new_biases);

    for (size_t i = 0; i < n - 1; i++)
	assert_gsl_vector_equal(old_biases[i], new_biases[i], EPSILON);

    for (size_t i = 0; i < n - 1; i++) {
	gsl_vector_free(old_biases[i]);
	gsl_vector_free(new_biases[i]);
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

void test_nn_network_ff(void **state) {
    const uintmax_t layers[] = {3, 2, 1};
    const size_t n = sizeof(layers) / sizeof(layers[0]);
    const size_t sizeof_matrix = nn_utils_sizeof_gsl_matrix();
    const size_t sizeof_vector = nn_utils_sizeof_gsl_vector();
    nn_network *network = nn_network_create(n, layers);

    gsl_vector **biases = (gsl_vector **) malloc((n - 1) * sizeof_vector);
    gsl_matrix **weights = (gsl_matrix **) malloc((n - 1) * sizeof_matrix);

    double weight = 0.0;
    for (size_t i = 0; i < n - 1; i++) {
	const uintmax_t layer_i = layers[i];
	const uintmax_t layer_ip = layers[i + 1];

    	biases[i] = gsl_vector_alloc(layer_ip);
	gsl_vector_set_all(biases[i], 0.1);

	weights[i] = gsl_matrix_alloc(layer_ip, layer_i);
    	for (size_t j = 0; j < weights[i]->size1; j++)
	    for (size_t k = 0; k < weights[i]->size2; k++) {
		weight += 0.1;
		gsl_matrix_set(weights[i], j, k, weight);
	    }
    }

    nn_network_set_biases(network, biases);
    nn_network_set_weights(network, weights);

    gsl_vector *x = nn_ones_v(layers[0]);

    gsl_vector *expected_y = gsl_vector_alloc(1);
    /* Value rounded to 16-dp according to Wolfram Alpha */
    gsl_vector_set_all(expected_y, 0.7744036918870783);

    gsl_vector *y = nn_network_ff(network, x);

    assert_gsl_vector_equal(y, expected_y, EPSILON);

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(expected_y);
    for (size_t i = 0; i < n - 1; i++) {
	gsl_vector_free(biases[i]);
	gsl_matrix_free(weights[i]);
    }
    free(biases);
    free(weights);
    nn_network_destroy(network);

    (void) state;
}
