#include "nn_math.h"

double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime(const double x) {
    const double sigma = sigmoid(x);
    return sigma*(1.0 - sigma);
}

gsl_vector * ones_v(const size_t n) {
    gsl_vector *ones = gsl_vector_alloc(n);
    gsl_vector_set_all(ones, 1.0);
    return ones;
}
