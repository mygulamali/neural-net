#pragma once

#include <math.h>
#include <stdint.h>

#include <gsl/gsl_vector.h>

double sigmoid(const double x);
double sigmoid_prime(const double x);

gsl_vector * ones_v(const size_t n);
gsl_vector * sigmoid_v(const gsl_vector *x);
gsl_vector * sigmoid_prime_v(const gsl_vector *x);
