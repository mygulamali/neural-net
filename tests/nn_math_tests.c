#include "nn_math_tests.h"

void test_nn_sigmoid(void **state) {
    /* Values rounded to 16-dp according to Wolfram Alpha */
    double xs[5][2] = {
	{-INFINITY, 0.0},
	{     -1.0, 0.2689414213699951},
	{      0.0, 0.5},
	{      1.0, 0.7310585786300049},
	{ INFINITY, 1.0}
    };

    for (uintmax_t i = 0; i < 5; i++) {
	const double y = nn_sigmoid(xs[i][0]);
	assert_double_equal(y, xs[i][1], EPSILON);
    }

    (void) state;
}

void test_nn_sigmoid_prime(void **state) {
    /* Values rounded to 16-dp according to Wolfram Alpha */
    double xs[5][2] = {
	{-INFINITY, 0.0},
	{     -1.0, 0.1966119332414819},
	{      0.0, 0.25},
	{      1.0, 0.1966119332414819},
	{ INFINITY, 0.0}
    };

    for (uintmax_t i = 0; i < 5; i++) {
	const double y = nn_sigmoid_prime(xs[i][0]);
	assert_double_equal(y, xs[i][1], EPSILON);
    }

    (void) state;
}

void test_nn_ones_v(void **state) {
    gsl_vector * ones = nn_ones_v(5);

    for (size_t i = 0; i < 5; i++) {
	const double one = gsl_vector_get(ones, i);
	assert_int_equal(one, 1.0);
    }

    gsl_vector_free(ones);
    (void) state;
}

void test_nn_sigmoid_v(void **state) {
    /* Values rounded to 16-dp according to Wolfram Alpha */
    double xs[5][2] = {
	{-INFINITY, 0.0},
	{     -1.0, 0.2689414213699951},
	{      0.0, 0.5},
	{      1.0, 0.7310585786300049},
	{ INFINITY, 1.0}
    };

    gsl_vector *x = gsl_vector_alloc(5);
    gsl_vector *expected_y = gsl_vector_alloc(5);
    for (intmax_t i = 0; i < 5; i++) {
	gsl_vector_set(x, i, xs[i][0]);
	gsl_vector_set(expected_y, i, xs[i][1]);
    }

    gsl_vector *y = nn_sigmoid_v(x);
    assert_gsl_vector_equal(y, expected_y, EPSILON);

    gsl_vector_free(y);
    gsl_vector_free(expected_y);
    gsl_vector_free(x);
    (void) state;
}

void test_nn_sigmoid_prime_v(void **state) {
    /* Values rounded to 16-dp according to Wolfram Alpha */
    double xs[5][2] = {
	{-INFINITY, 0.0},
	{     -1.0, 0.1966119332414819},
	{      0.0, 0.25},
	{      1.0, 0.1966119332414819},
	{ INFINITY, 0.0}
    };

    gsl_vector *x = gsl_vector_alloc(5);
    gsl_vector *expected_y = gsl_vector_alloc(5);
    for (intmax_t i = 0; i < 5; i++) {
	gsl_vector_set(x, i, xs[i][0]);
	gsl_vector_set(expected_y, i, xs[i][1]);
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
	    assert_int_equal(one, 1.0);
	}
    }

    gsl_matrix_free(ones);
    (void) state;
}
