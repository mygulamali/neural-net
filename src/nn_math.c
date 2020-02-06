#include "nn_math.h"

real_t sigmoid(const real_t x) {
    return 1.0 / (1.0 + exp(-x));
}
