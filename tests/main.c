#include "nn_math_tests.h"

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_nn_sigmoid),
        cmocka_unit_test(test_nn_sigmoid_prime),
	cmocka_unit_test(test_nn_ones_v),
        cmocka_unit_test(test_nn_sigmoid_v),
        cmocka_unit_test(test_nn_sigmoid_prime_v),
	cmocka_unit_test(test_nn_ones_m)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
