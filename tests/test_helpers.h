#pragma once

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <cmocka.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

/* A very small number */
#define EPSILON 1.0e-16

#ifdef DOXYGEN
/**
 * @brief Assert that the two given doubles are equal to within tolerance.
 *
 * The function prints an error message to standard error and terminates the
 * test by calling fail() if the doubles are not equal to within tolerance.
 *
 * @param[in]  a    The first double to compare.
 *
 * @param[in]  b    The double to compare against the first one.
 *
 * @param[in]  eps  The double tolerance value.
 */
void assert_double_equal(double a, double b, double eps);
#else
#define assert_double_equal(a, b, eps) \
    _assert_double_equal(a, b, eps, __FILE__, __LINE__)
#endif

#ifdef DOXYGEN
/**
 * @brief Assert that the two given doubles are not equal to within tolerance.
 *
 * The function prints an error message to standard error and terminates the
 * test by calling fail() if the doubles are equal to within tolerance.
 *
 * @param[in]  a    The first double to compare.
 *
 * @param[in]  b    The double to compare against the first one.
 *
 * @param[in]  eps  The double tolerance value.
 *
 * @see assert_double_equal()
 */
void assert_double_not_equal(double a, double b, double eps);
#else
#define assert_double_not_equal(a, b, eps) \
    _assert_double_not_equal(a, b, eps, __FILE__, __LINE__)
#endif

#ifdef DOXYGEN
/**
 * @brief Assert that the two given GSL vectors are equal.
 *
 * The function prints an error message to standard error and terminates the
 * test by calling fail() if the vectors are not equal to within tolerance.
 *
 * @param[in]  a    The first GSL vector to compare.
 *
 * @param[in]  b    The GSL vector to compare against the first one.
 *
 * @param[in]  eps  The double tolerance value.
 *
 * @see assert_gsl_vector_equal()
 */
void assert_gsl_vector_equal(gsl_vector *a, gsl_vector *b, double eps);
#else
#define assert_gsl_vector_equal(a, b, eps) \
    _assert_gsl_vector_equal(a, b, eps, __FILE__, __LINE__)
#endif

#ifdef DOXYGEN
/**
 * @brief Assert that the two given GSL matrices are equal.
 *
 * The function prints an error message to standard error and terminates the
 * test by calling fail() if the matrices are not equal to within tolerance.
 *
 * @param[in]  a    The first GSL matrix to compare.
 *
 * @param[in]  b    The GSL matrix to compare against the first one.
 *
 * @param[in]  eps  The double tolerance value.
 *
 * @see assert_gsl_matrix_equal()
 */
void assert_gsl_matrix_equal(gsl_matrix *a, gsl_matrix *b, double eps);
#else
#define assert_gsl_matrix_equal(a, b, eps) \
    _assert_gsl_matrix_equal(a, b, eps, __FILE__, __LINE__)
#endif

void _assert_double_equal(const double a, const double b, const double eps,
                          const char* const file, const int line);

void _assert_double_not_equal(const double a, const double b, const double eps,
                              const char* const file, const int line);

void _assert_gsl_vector_equal(const gsl_vector *a, const gsl_vector *b, double eps,
			      const char* const file, const int line);

void _assert_gsl_matrix_equal(const gsl_matrix *a, const gsl_matrix *b, double eps,
			      const char* const file, const int line);
