#include "test_helpers.h"

/* GCC have printf type attribute check.  */
#ifdef __GNUC__
#define CMOCKA_PRINTF_ATTRIBUTE(a,b) __attribute__ \
    ((__format__ (__printf__, a, b)))
#else
#define CMOCKA_PRINTF_ATTRIBUTE(a,b)
#endif /* __GNUC__ */

void cm_print_error(const char * const format, ...)
    CMOCKA_PRINTF_ATTRIBUTE(1, 2);

/* Returns 1 if the specified values are equal within tolerance. If the values
 * are not equal, an error is displayed and 0 is returned. */
static int double_values_equal_display_error(const double left,
                                             const double right,
                                             const double tolerance) {
    const int equal = fabs(left - right) < tolerance;
    if (!equal) {
        cm_print_error("%f != %f ± %f\n", left, right, tolerance);
    }
    return equal;
}

/*
 * Returns 1 if the specified values are not equal within tolerance. If the
 * values are equal an error is displayed and 0 is returned. */
static int double_values_not_equal_display_error(const double left,
                                                 const double right,
                                                 const double tolerance) {
    const int not_equal = fabs(left - right) >= tolerance;
    if (!not_equal) {
        cm_print_error("%f == %f ± %f\n", left, right, tolerance);
    }
    return not_equal;
}

/* Returns 1 if the specified values are equal within tolerance. If the values
 * are not equal, an error is displayed and 0 is returned. */
static int gsl_vector_values_equal_display_error(const gsl_vector *left,
						 const gsl_vector *right,
						 const double tolerance) {
    for (size_t i = 0; i < left->size; i++) {
	const double left_i = gsl_vector_get(left, i);
	const double right_i = gsl_vector_get(right, i);

	const int equal = fabs(left_i - right_i) < tolerance;

        if (!equal) {
	    cm_print_error(
		"%f != %f ± %f at index %lo\n", left_i, right_i, tolerance, i
	    );
	    return equal;
	}
    }

    return 1;
}

/* Returns 1 if the specified values are equal within tolerance. If the values
 * are not equal, an error is displayed and 0 is returned. */
static int gsl_matrix_values_equal_display_error(const gsl_matrix *left,
						 const gsl_matrix *right,
						 const double tolerance) {
    for (size_t i = 0; i < left->size1; i++) {
	for (size_t j = 0; j < left->size2; j++) {
	    const double left_ij = gsl_matrix_get(left, i, j);
	    const double right_ij = gsl_matrix_get(right, i, j);

	    const int equal = fabs(left_ij - right_ij) < tolerance;

	    if (!equal) {
		cm_print_error(
		    "%f != %f ± %f at index (%lo, %lo)\n",
		    left_ij, right_ij, tolerance, i, j
		);
		return equal;
	    }
	}
    }

    return 1;
}

void _assert_double_equal(
    const double a, const double b, const double eps,
    const char* const file, const int line
) {
    if (!double_values_equal_display_error(a, b, eps)) {
        _fail(file, line);
    }
}

void _assert_double_not_equal(
    const double a, const double b, const double eps,
    const char* const file, const int line
) {
    if (!double_values_not_equal_display_error(a, b, eps)) {
        _fail(file, line);
    }
}

void _assert_gsl_vector_equal(
    const gsl_vector *a, const gsl_vector *b, const double eps,
    const char* const file, const int line
) {
    if (!gsl_vector_values_equal_display_error(a, b, eps)) {
	_fail(file, line);
    }
}

void _assert_gsl_matrix_equal(
    const gsl_matrix *a, const gsl_matrix *b, const double eps,
    const char* const file, const int line
) {
    if (!gsl_matrix_values_equal_display_error(a, b, eps)) {
	_fail(file, line);
    }
}
