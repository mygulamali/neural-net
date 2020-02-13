#include "nn_math_tests.h"

void test_nn_sigmoid(void **state) {
    for (uintmax_t i = 0; i < 5; i++) {
	const double y = nn_sigmoid(SIGMOID[i][0]);
	assert_double_equal(y, SIGMOID[i][1], EPSILON);
    }

    (void) state;
}

void test_nn_sigmoid_prime(void **state) {
    for (uintmax_t i = 0; i < 5; i++) {
	const double y = nn_sigmoid_prime(SIGMOID_PRIME[i][0]);
	assert_double_equal(y, SIGMOID_PRIME[i][1], EPSILON);
    }

    (void) state;
}

void test_nn_ones_v(void **state) {
    gsl_vector * ones = nn_ones_v(5);

    for (size_t i = 0; i < 5; i++) {
	const double one = gsl_vector_get(ones, i);
	assert_double_equal(one, 1.0, EPSILON);
    }

    gsl_vector_free(ones);
    (void) state;
}

void test_nn_sigmoid_v(void **state) {
    gsl_vector *x = gsl_vector_alloc(5);
    gsl_vector *expected_y = gsl_vector_alloc(5);
    for (intmax_t i = 0; i < 5; i++) {
	gsl_vector_set(x, i, SIGMOID[i][0]);
	gsl_vector_set(expected_y, i, SIGMOID[i][1]);
    }

    gsl_vector *y = nn_sigmoid_v(x);
    assert_gsl_vector_equal(y, expected_y, EPSILON);

    gsl_vector_free(y);
    gsl_vector_free(expected_y);
    gsl_vector_free(x);
    (void) state;
}

void test_nn_sigmoid_prime_v(void **state) {
    gsl_vector *x = gsl_vector_alloc(5);
    gsl_vector *expected_y = gsl_vector_alloc(5);
    for (intmax_t i = 0; i < 5; i++) {
	gsl_vector_set(x, i, SIGMOID_PRIME[i][0]);
	gsl_vector_set(expected_y, i, SIGMOID_PRIME[i][1]);
    }

    gsl_vector *y = nn_sigmoid_prime_v(x);
    assert_gsl_vector_equal(y, expected_y, EPSILON);

    gsl_vector_free(y);
    gsl_vector_free(expected_y);
    gsl_vector_free(x);
    (void) state;
}

void test_nn_ones_m(void **state) {
    gsl_matrix * ones = nn_ones_m(3, 2);

    for (size_t i = 0; i < 3; i++) {
	for (size_t j = 0; j < 2; j++) {
	    const double one = gsl_matrix_get(ones, i, j);
	    assert_double_equal(one, 1.0, EPSILON);
	}
    }

    gsl_matrix_free(ones);
    (void) state;
}

void test_nn_sigmoid_m(void **state) {
    gsl_matrix *x = gsl_matrix_alloc(5, 1);
    gsl_matrix *expected_y = gsl_matrix_alloc(5, 1);
    for (intmax_t i = 0; i < 5; i++) {
    	gsl_matrix_set(x, i, 0, SIGMOID[i][0]);
    	gsl_matrix_set(expected_y, i, 0, SIGMOID[i][1]);
    }

    gsl_matrix *y = nn_sigmoid_m(x);
    assert_gsl_matrix_equal(y, expected_y, EPSILON);

    gsl_matrix_free(y);
    gsl_matrix_free(expected_y);
    gsl_matrix_free(x);
    (void) state;
}

void test_nn_sigmoid_prime_m(void **state) {
    gsl_matrix *x = gsl_matrix_alloc(5, 1);
    gsl_matrix *expected_y = gsl_matrix_alloc(5, 1);
    for (intmax_t i = 0; i < 5; i++) {
	gsl_matrix_set(x, i, 0, SIGMOID_PRIME[i][0]);
	gsl_matrix_set(expected_y, i, 0, SIGMOID_PRIME[i][1]);
    }

    gsl_matrix *y = nn_sigmoid_prime_m(x);
    assert_gsl_matrix_equal(y, expected_y, EPSILON);

    gsl_matrix_free(y);
    gsl_matrix_free(expected_y);
    gsl_matrix_free(x);
    (void) state;
}
