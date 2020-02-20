#include "nn_math_tests.h"
#include "nn_network_tests.h"

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_nn_sigmoid),
        cmocka_unit_test(test_nn_sigmoid_prime),
	cmocka_unit_test(test_nn_ones_v),
        cmocka_unit_test(test_nn_sigmoid_v),
        cmocka_unit_test(test_nn_sigmoid_prime_v),
	cmocka_unit_test(test_nn_ones_m),
        cmocka_unit_test(test_nn_sigmoid_m),
        cmocka_unit_test(test_nn_sigmoid_prime_m),
        cmocka_unit_test(test_nn_network_create_destroy),
        cmocka_unit_test(test_nn_network_set_get_biases),
        cmocka_unit_test(test_nn_network_set_get_weights),
        cmocka_unit_test(test_nn_network_ff),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
