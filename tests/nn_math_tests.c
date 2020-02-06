#include "nn_math_tests.h"

void test_sigmoid(void **state) {
    /* Values rounded to 16-dp according to Wolfram Alpha */
    real_t xs[5][2] = {
	{-INFINITY, 0.0},
	{     -1.0, 0.2689414213699951},
	{      0.0, 0.5},
	{      1.0, 0.7310585786300049},
	{ INFINITY, 1.0}
    };

    for (intmax_t i = 0; i < 5; i++) {
	const real_t y = sigmoid(xs[i][0]);
	assert_double_equal(y, xs[i][1], EPSILON);
    }

    (void) state;
}
