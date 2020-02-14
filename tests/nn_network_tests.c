#include "nn_network_tests.h"

void test_nn_network_create_destroy(void **state) {
    const uintmax_t layers[] = {4, 3, 2};
    const size_t n = sizeof(layers) / sizeof(layers[0]);

    nn_network *network = nn_network_create(n, layers);

    assert_int_equal(network->size, n);

    for (size_t i = 0; i < n; i++)
	assert_int_equal(network->layers[i], layers[i]);

    for (size_t i = 0; i < n - 1; i++) {
	gsl_matrix *bias = network->biases[i];
	assert_int_equal(bias->size1, layers[i + 1]);
	assert_int_equal(bias->size2, 1);

	for (size_t ii = 0; ii < bias->size1; ii++)
	    for (size_t jj = 0; jj < bias->size2; jj++)
		assert_double_equal(
		    gsl_matrix_get(bias, ii, jj),
		    0.0,
		    EPSILON
		);

	gsl_matrix *weight = network->weights[i];
	assert_int_equal(weight->size1, layers[i + 1]);
	assert_int_equal(weight->size2, layers[i]);

	for (size_t ii = 0; ii < weight->size1; ii++)
	    for (size_t jj = 0; jj < weight->size2; jj++)
		assert_double_equal(
		    gsl_matrix_get(weight, ii, jj),
		    0.0,
		    EPSILON
		);
    }

    nn_network_destroy(network);

    (void) state;
}
