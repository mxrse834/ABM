"""There are two completely separate phases
Phase 1 — Build & Run        (no real data needed)
Phase 2 — Calibrate          (real data needed)
Right now we're doing Phase 1. You don't need any real data yet.

Phase 1 — what we're doing
You're building a simulation with made-up but reasonable numbers. Like:
"firms start with prices around 1.0"
"households spend about 70% of income"
"central bank targets 4% inflation"
The goal is just to see inflation emerge from agent interactions. You're not trying to match real Indian CPI yet. You're just checking the machine works.
No training. No data. Just simulation logic.

Phase 2 — calibration (later)
This is where real data comes in. Specifically:
Raw data you'd use:
1) RBI DBIE monthly CPI series (free, public) — 33 components from 2011
2) RBI inflation expectations survey — household level
3) PLFS wage data — sector level wages
What you do with it:
You compute summary statistics from the real data — things like:
real mean monthly inflation     = 0.42%
real std of monthly inflation   = 0.38%
real autocorrelation at lag 1   = 0.71
"""

"""
inflation_abm.py — Phase 1 ABM for Indian Inflation
Goal: produce ~5% mean annual inflation with realistic variance
"""
"""
inflation_abm.py — Phase 2 ABM for Indian Inflation
Fixes over Phase 1:
  1. Autocorrelation: expectations track 3-month smoothed inflation
  2. Wage-price spiral: wages feed into firm costs every tick
  3. Monsoon shock: annual kharif shock hits food firms Jun-Sep
  4. Fuel persistence: oil shocks last 2-4 months
Target moments (real Indian CPI):
  mean ~5%, std ~1.5%, autocorrelation lag1 ~0.65
"""

import numpy as np

rng = np.random.default_rng(seed=42)

# ── CONFIG ────────────────────────────────────────────
N_F  = 500
N_H  = 2000
T    = 120
BURN = 24

SECTORS = [
    ("Food",          175, 0.08, 0.04, 0.25),
    ("Fuel",           50, 0.12, 0.05, 0.60),
    ("Manufacturing", 100, 0.06, 0.03, 0.20),
    ("Services",      125, 0.04, 0.02, 0.08),
    ("Housing",        50, 0.02, 0.01, 0.03),
]
assert sum(s[1] for s in SECTORS) == N_F

INFLATION_TARGET = 0.04 / 12
POLICY_RATE_INIT = 0.065 / 12

# Real Indian CPI sector weights
REAL_WEIGHTS = np.array([0.46, 0.07, 0.17, 0.22, 0.08])

# ── INIT FIRMS ────────────────────────────────────────
sector_id        = np.empty(N_F, dtype=int)
markup_sens      = np.empty(N_F)
cost_sensitivity = np.empty(N_F)

idx = 0
for s_id, (name, n, mu, sigma, cs) in enumerate(SECTORS):
    sector_id[idx:idx+n]        = s_id
    markup_sens[idx:idx+n]      = np.clip(rng.normal(mu, sigma, n), 0.001, 0.3)
    cost_sensitivity[idx:idx+n] = np.clip(cs + 0.02*rng.standard_normal(n), 0.01, 0.8)
    idx += n

price           = rng.lognormal(0.0, 0.05, N_F)
expected_demand = np.full(N_F, float(N_H) / N_F)
firm_wage_cost  = price * rng.uniform(0.55, 0.70, N_F)

# wage share by sector: services high, fuel low
wage_shares = np.where(sector_id == 3, 0.60,
              np.where(sector_id == 0, 0.40,
              np.where(sector_id == 1, 0.20, 0.35)))

# ── INIT HOUSEHOLDS ───────────────────────────────────
def beta_sample(a, b, size):
    g1 = rng.gamma(a, 1.0, size)
    g2 = rng.gamma(b, 1.0, size)
    return g1 / (g1 + g2)

savings_rate  = beta_sample(2, 5, N_H)
expect_weight = rng.uniform(0.80, 0.95, N_H)   # high persistence — India-specific
income        = rng.lognormal(np.log(50), 0.4, N_H)
expectation   = np.full(N_H, INFLATION_TARGET)

# ── CPI ───────────────────────────────────────────────
def compute_cpi(prices):
    return sum(REAL_WEIGHTS[s] * prices[sector_id == s].mean()
               for s in range(len(SECTORS)))

# ── STATE ─────────────────────────────────────────────
cpi_history        = []
inflation_history  = []
rate_history       = []
sector_price_index = {s[0]: [] for s in SECTORS}

inf_buffer         = [INFLATION_TARGET] * 3   # 3-month smoother
prev_cpi           = compute_cpi(price)
policy_rate        = POLICY_RATE_INIT
avg_inf_smooth     = INFLATION_TARGET

# persistent fuel shock state
fuel_shock_remaining = 0
fuel_shock_magnitude = 0.0

# monsoon state — set at June each year
monsoon_shock_active = 0.0

print("Running simulation...")
print(f"{'Tick':>5}  {'Inflation (ann%)':>16}  {'Rate (ann%)':>12}  {'Event':>12}")
print("-" * 52)

for t in range(T + BURN):

    month_of_year = t % 12

    # ── MONSOON (draw in June, active Jun-Sep) ────────
    if month_of_year == 5:
        roll = rng.random()
        if roll < 0.25:
            monsoon_shock_active = rng.uniform(0.012, 0.025)   # bad
        elif roll < 0.45:
            monsoon_shock_active = -rng.uniform(0.005, 0.012)  # good
        else:
            monsoon_shock_active = 0.0

    decay = max(0.0, 1.0 - (month_of_year - 5) * 0.25) if month_of_year in [5,6,7,8] else 0.0
    active_monsoon = monsoon_shock_active * decay

    # ── FUEL SHOCK (persistent) ───────────────────────
    if fuel_shock_remaining == 0 and rng.random() < 0.12:
        fuel_shock_remaining = int(rng.integers(2, 5))
        fuel_shock_magnitude = rng.uniform(0.015, 0.05)
        if rng.random() < 0.3:
            fuel_shock_magnitude *= -1
    if fuel_shock_remaining > 0:
        current_fuel_shock = fuel_shock_magnitude
        fuel_shock_remaining -= 1
    else:
        current_fuel_shock = 0.0

    # ── 1. HOUSEHOLDS ─────────────────────────────────
    expectation  = expect_weight * avg_inf_smooth + (1 - expect_weight) * expectation
    avg_expect   = float(expectation.mean())

    real_rate    = policy_rate - avg_expect
    rate_drag    = np.clip(real_rate * 1.5, -0.04, 0.04)
    budgets      = (1 - savings_rate) * income * (1 - rate_drag)
    total_demand = float(budgets.sum())

    # ── 2. GOODS MARKET ───────────────────────────────
    inv_p         = 1.0 / (price + 1e-9)
    shares        = inv_p / inv_p.sum()
    actual_demand = shares * total_demand
    excess_demand = np.clip(
        (actual_demand - expected_demand) / (expected_demand + 1e-9), -0.3, 0.3)

    # ── 3. COST SHOCKS ────────────────────────────────
    base_shock              = rng.normal(0.006, 0.006, N_F)
    base_shock[sector_id==1] += current_fuel_shock
    base_shock[sector_id==0] += active_monsoon

    # wage-price spiral — every tick
    wage_growth    = 0.55 * avg_inf_smooth + 0.002
    firm_wage_cost = firm_wage_cost * (1 + np.clip(wage_growth, 0, 0.012))
    effective_shock = base_shock + wage_shares * wage_growth

    # ── 4. FIRMS: price update ────────────────────────
    price_change = (
          markup_sens * excess_demand
        + cost_sensitivity * effective_shock
        + 0.20 * avg_expect
    )
    price = price * (1 + np.clip(price_change, -0.02, 0.025))
    expected_demand = 0.8 * expected_demand + 0.2 * actual_demand

    # ── 5. CPI ────────────────────────────────────────
    current_cpi = compute_cpi(price)
    inflation   = (current_cpi / prev_cpi) - 1.0
    prev_cpi    = current_cpi

    inf_buffer.append(inflation)
    inf_buffer.pop(0)
    avg_inf_smooth = float(np.mean(inf_buffer))

    # ── 6. CENTRAL BANK ───────────────────────────────
    pi_gap      = avg_inf_smooth - INFLATION_TARGET
    output_gap  = float(excess_demand.mean())
    policy_rate = float(np.clip(
        POLICY_RATE_INIT + 1.5*pi_gap + 0.5*output_gap,
        0.04/12, 0.15/12))

    # ── 7. WAGES ──────────────────────────────────────
    income = income * (1 + np.clip(wage_growth, 0, 0.012))

    # ── RECORD ────────────────────────────────────────
    if t >= BURN:
        cpi_history.append(current_cpi)
        inflation_history.append(inflation)
        rate_history.append(policy_rate)
        for s_id, (name, *_) in enumerate(SECTORS):
            sector_price_index[name].append(price[sector_id == s_id].mean())

        tick = t - BURN
        if tick % 12 == 0:
            ann = ((1+inflation)**12 - 1)*100
            r   = policy_rate*12*100
            ev  = "BAD MON" if active_monsoon > 0.005 else \
                  "GOOD MON" if active_monsoon < -0.002 else \
                  "FUEL↑" if current_fuel_shock > 0.01 else ""
            print(f"{tick:>5}  {ann:>15.2f}%  {r:>11.2f}%  {ev:>12}")

print("\nSimulation complete.")

inf_arr = np.array(inflation_history)
ann_inf = ((1+inf_arr)**12 - 1)*100

print("\n── Summary Statistics ──────────────────────────────")
print(f"  Mean annual inflation      : {ann_inf.mean():.2f}%   (target ~5%)")
print(f"  Std  annual inflation      : {ann_inf.std():.2f}%   (target ~1.5%)")
print(f"  Min / Max                  : {ann_inf.min():.2f}% / {ann_inf.max():.2f}%")
for lag in [1,2,3]:
    ac = np.corrcoef(inf_arr[:-lag], inf_arr[lag:])[0,1]
    print(f"  Autocorrelation (lag {lag})   : {ac:.3f}   (target ~0.65)")
print("────────────────────────────────────────────────────")