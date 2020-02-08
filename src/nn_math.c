#include "nn_math.h"

real_t sigmoid(const real_t x) {
    return 1.0 / (1.0 + exp(-x));
}

real_t sigmoid_prime(const real_t x) {
    const real_t sigma = sigmoid(x);
    return sigma*(1.0 - sigma);
}

gsl_vector * ones_v(const size_t n) {
    gsl_vector *ones = gsl_vector_alloc(n);
    gsl_vector_set_all(ones, 1.0);
    return ones;
}
