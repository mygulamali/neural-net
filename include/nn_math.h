#pragma once

#include <math.h>
#include <stdint.h>

#include <gsl/gsl_vector.h>

#include "nn_types.h"

real_t sigmoid(const real_t x);
real_t sigmoid_prime(const real_t x);

gsl_vector * ones_v(const size_t n);
