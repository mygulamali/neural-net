#include "nn_math_tests.h"

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sigmoid)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}