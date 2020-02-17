#include "nn_utils.h"

size_t nn_utils_sizeof_gsl_matrix(void) {
    gsl_matrix *matrix = gsl_matrix_alloc(0, 0);
    size_t size = sizeof(*matrix);
    gsl_matrix_free(matrix);

    return size;
}
