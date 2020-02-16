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
