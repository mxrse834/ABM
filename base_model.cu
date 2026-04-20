#include <math_functions.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>



// Households
#define N_H 1000000 


// Firms
#define N_F 1000
///The division of firms 
/*
Food (cereals, pulses, vegetables etc.) — ~350 firms
Fuel (diesel, LPG, kerosene) — ~100 firms
Manufacturing (clothing, household goods) — ~200 firms
Services (health, education, transport) — ~250 firms
Housing — ~100 firms
*/
#define N_F_food 350
#define N_F_fuel 100
#define N_F_manufacturing 200
#define N_F_services 250
#define N_F_housing 100


#define N_B 1       // banks (scalar for now)
typedef unsigned long ul;

// ── Household state
// Parameters:
float *d_savings_rate;  // s(i) ~ Beta(2,5)
float *d_expect_weight; // lam(i) ~ Uniform(0.3, 0.8)

// State
float *d_income;      // w(i,t)
float *d_wealth;      // a(i,t)
float *d_expectation; // pi(i,t)  (adaptive inflation expectation)
float *d_consumption; // C(i,t)  (computed each tick)

// ── Firm state
// Parameters
float *d_markup_sensitivity; // mu(i) ~ LogNormal(-1.5, 0.4)

// State
float *d_price;           // p(i,t)
float *d_inventory;       // Inv(i,t)
float *d_cost;            // c(i,t) = wage / productivity
float *d_expected_demand; // D(i,t)

// ── Bank state (scalars, pinned host or device scalar)
float h_deposits;
float h_loans_outstanding;
float h_interest_rate_spread;
float h_risk_tolerance;

// ── Central bank (scalars)
float h_policy_rate;
float h_inflation_target;

__device__ void setup_rng(curandState *states, ul seed)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

template <int k>
__device__ float gamma_generator(curandState *states)
{
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < k; i++)
    {
        float l = curand_uniform(states);
        s += -logf(l);
    }
    return s;
}

template <int v1, int v2>
__device__ float beta_generator(curandState *states)
{
    float g1 = gamma_generator<v1>(states);
    float g2 = gamma_generator<v2>(states);
    return g1 / (g1 + g2);
}

template <int v1, int v2>
__device__ void beta_gen_api(curandState *states, float *out, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > N)
        return;
    out[tid] = beta_generator<v1, v2>(states);
}

__device__ void uniform_gen_api(curandState *states, float *out, float a, float b, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > N)
        return;
    out[tid] = a + (b - a) * curand_uniform(states);
}

__device__ void lognormal_gen_api(curandState *states, float *out, float mean, float stddev, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > N)
        return;
    out[tid] = curand_log_normal(states, mean, stddev);
}

__global__ void household_init(curandState *states, float *savings_rate, float *expect_weight, float *income, float *wealth, float *expectation, float *consumption)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > N_H)
        return;
    curandState f = states[tid];
    beta_gen_api<2, 5>(&f, savings_rate, N_H);
    uniform_gen_api(&f, expect_weight, 0.3f, 0.8f, N_H);
    /*income[tid] = 1.0f;       // initial income
    wealth[tid] = 1.0f;       // initial wealth
    expectation[tid] = 0.02f; // initial inflation expectation
    consumption[tid] = 0.0f;  // will be computed each tick
    states[tid] = f;*/
}

int main()
{
    uint32_t *d_savings_rate, *d_expect_weight, *d_income, *d_wealth, *d_expectation, *d_consumption;
    cudaMalloc(&d_savings_rate, N_H * sizeof(float));
    cudaMalloc(&d_expect_weight, N_H * sizeof(float));
    cudaMalloc(&d_income, N_H * sizeof(float));
    cudaMalloc(&d_wealth, N_H * sizeof(float));
    cudaMalloc(&d_expectation, N_H * sizeof(float));
    cudaMalloc(&d_consumption, N_H * sizeof(float));
    return 0;
}