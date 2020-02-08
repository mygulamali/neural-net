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

gsl_vector * sigmoid_v(const gsl_vector *x) {
    gsl_vector *denom = gsl_vector_alloc(x->size);

    gsl_vector_memcpy(denom, x);
    gsl_vector_scale(denom, -1.0);

    for (size_t i = 0; i < denom->size; i++) {
	const double denom_i = gsl_vector_get(denom, i);
	gsl_vector_set(denom, i, exp(denom_i));
    }

    gsl_vector_add_constant(denom, 1.0);

    gsl_vector *nom = ones_v(x->size);
    gsl_vector_div(nom, denom);

    gsl_vector_free(denom);
    return nom;
}
