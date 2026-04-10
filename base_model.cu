#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N_H 1000000 // households
#define N_F 1000    // firms
#define N_B 1       // banks (scalar for now)

// ── Household state 
// Parameters: 
float *d_savings_rate;  // s(i) ~ Beta(2,5),   bounded [0,1]
float *d_expect_weight; // λ(i) ~ Uniform(0.3, 0.8), how backward-looking

// State
float *d_income;      // w(i,t)
float *d_wealth;      // a(i,t)
float *d_expectation; // pi(i,t)  (adaptive inflation expectation)
float *d_consumption; // C(i,t)  (computed each tick)

// ── Firm state
// Parameters
float *d_markup_sensitivity; // μ(i) ~ LogNormal(-1.5, 0.4)

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