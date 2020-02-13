#pragma once

#include <math.h>
#include <stdint.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

double nn_sigmoid(const double x);
double nn_sigmoid_prime(const double x);

gsl_vector * nn_ones_v(const size_t n);
gsl_vector * nn_sigmoid_v(const gsl_vector *x);
gsl_vector * nn_sigmoid_prime_v(const gsl_vector *x);

gsl_matrix * nn_ones_m(const size_t n, const size_t m);
gsl_matrix * nn_sigmoid_m(const gsl_matrix *x);
