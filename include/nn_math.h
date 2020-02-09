#pragma once

#include <math.h>
#include <stdint.h>

#include <gsl/gsl_vector.h>

double nn_sigmoid(const double x);
double nn_sigmoid_prime(const double x);

gsl_vector * nn_ones_v(const size_t n);
gsl_vector * nn_sigmoid_v(const gsl_vector *x);
gsl_vector * nn_sigmoid_prime_v(const gsl_vector *x);
