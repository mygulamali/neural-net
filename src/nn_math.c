#include "nn_math.h"

double nn_sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

double nn_sigmoid_prime(const double x) {
    const double sigma = nn_sigmoid(x);
    return sigma*(1.0 - sigma);
}

gsl_vector * nn_ones_v(const size_t n) {
    gsl_vector *ones = gsl_vector_alloc(n);
    gsl_vector_set_all(ones, 1.0);
    return ones;
}

gsl_vector * nn_sigmoid_v(const gsl_vector *x) {
    gsl_vector *denom = gsl_vector_alloc(x->size);

    gsl_vector_memcpy(denom, x);
    gsl_vector_scale(denom, -1.0);

    for (size_t i = 0; i < denom->size; i++) {
	const double denom_i = gsl_vector_get(denom, i);
	gsl_vector_set(denom, i, exp(denom_i));
    }

    gsl_vector_add_constant(denom, 1.0);

    gsl_vector *nom = nn_ones_v(x->size);
    gsl_vector_div(nom, denom);

    gsl_vector_free(denom);
    return nom;
}

gsl_vector * nn_sigmoid_prime_v(const gsl_vector *x) {
    gsl_vector *sigma = nn_sigmoid_v(x);
    gsl_vector *result = nn_ones_v(x->size);

    gsl_vector_sub(result, sigma);
    gsl_vector_mul(result, sigma);

    gsl_vector_free(sigma);
    return result;
}
