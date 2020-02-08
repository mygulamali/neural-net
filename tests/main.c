#include "nn_math_tests.h"

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sigmoid),
        cmocka_unit_test(test_sigmoid_prime),
	cmocka_unit_test(test_ones_v),
        cmocka_unit_test(test_sigmoid_v)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
