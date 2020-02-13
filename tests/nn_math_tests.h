#pragma once

#include "test_helpers.h"
#include "nn_math.h"

/* Values rounded to 16-dp according to Wolfram Alpha */
static double SIGMOID[5][2] = {
    {-INFINITY, 0.0},
    {     -1.0, 0.2689414213699951},
    {      0.0, 0.5},
    {      1.0, 0.7310585786300049},
    { INFINITY, 1.0}
};

static double SIGMOID_PRIME[5][2] = {
    {-INFINITY, 0.0},
    {     -1.0, 0.1966119332414819},
    {      0.0, 0.25},
    {      1.0, 0.1966119332414819},
    { INFINITY, 0.0}
};

void test_nn_sigmoid(void **state);
void test_nn_sigmoid_prime(void **state);

void test_nn_ones_v(void **state);
void test_nn_sigmoid_v(void **state);
void test_nn_sigmoid_prime_v(void **state);

void test_nn_ones_m(void **state);
void test_nn_sigmoid_m(void **state);
