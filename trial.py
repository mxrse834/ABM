# ─────────────────────────────────────────────────────────────
# Inflation ABM + Calibration + Forecast (ALL-IN-ONE)
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# LOAD REAL DATA
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("/Users/pnglinkx/Desktop/CPIndex_Jan13-To-Dec25.csv", skiprows=1)
df.columns = df.columns.str.strip()

df['Combined'] = pd.to_numeric(df['Combined'], errors='coerce')
df['Month'] = pd.to_datetime(df['Month'], format='%B').dt.month
df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
df = df.sort_values('Date')

real_cpi = df['Combined'].values
real_inflation = (real_cpi[1:] / real_cpi[:-1]) - 1

# ─────────────────────────────────────────────────────────────
# MODEL FUNCTION
# ─────────────────────────────────────────────────────────────

def run_model(markup_scale=1.0, cost_scale=1.0):

    rng = np.random.default_rng(seed=42)

    N_F, N_H = 500, 2000
    T, BURN = 120, 24

    SECTORS = [
        ("Food",175,0.08,0.04,0.25),
        ("Fuel",50,0.12,0.05,0.60),
        ("Manufacturing",100,0.06,0.03,0.20),
        ("Services",125,0.04,0.02,0.08),
        ("Housing",50,0.02,0.01,0.03),
    ]

    REAL_WEIGHTS = np.array([0.46,0.07,0.17,0.22,0.08])

    sector_id = np.empty(N_F, dtype=int)
    markup_sens = np.empty(N_F)
    cost_sensitivity = np.empty(N_F)

    idx = 0
    for s_id,(_,n,mu,sigma,cs) in enumerate(SECTORS):
        sector_id[idx:idx+n] = s_id
        markup_sens[idx:idx+n] = np.clip(rng.normal(mu,sigma,n),0.001,0.3) * markup_scale
        cost_sensitivity[idx:idx+n] = np.clip(cs+0.02*rng.standard_normal(n),0.01,0.8) * cost_scale
        idx += n

    price = rng.lognormal(0.0,0.05,N_F)
    expected_demand = np.full(N_F, float(N_H)/N_F)

    def compute_cpi(prices):
        return sum(REAL_WEIGHTS[s]*prices[sector_id==s].mean()
                   for s in range(len(SECTORS)))

    cpi_history = []
    prev_cpi = compute_cpi(price)

    for t in range(T+BURN):

        total_demand = N_H * 50

        inv_p = 1.0/(price+1e-9)
        shares = inv_p / inv_p.sum()
        actual_demand = shares * total_demand

        excess = np.clip((actual_demand-expected_demand)/(expected_demand+1e-9),-0.3,0.3)

        shock = rng.normal(0.006,0.006,N_F)

        price_change = markup_sens*excess + cost_sensitivity*shock
        price = price * (1 + np.clip(price_change,-0.02,0.025))

        expected_demand = 0.8*expected_demand + 0.2*actual_demand

        current_cpi = compute_cpi(price)

        if t >= BURN:
            cpi_history.append(current_cpi)

        prev_cpi = current_cpi

    model_cpi = np.array(cpi_history)
    return model_cpi

# ─────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_loss(params):
    model_cpi = run_model(*params)

    scale = real_cpi[0] / model_cpi[0]
    model_cpi *= scale

    model_inf = (model_cpi[1:] / model_cpi[:-1]) - 1

    n = min(len(model_inf), len(real_inflation))
    return np.mean((model_inf[:n] - real_inflation[:n])**2)

# ─────────────────────────────────────────────────────────────
# CALIBRATION (RANDOM SEARCH)
# ─────────────────────────────────────────────────────────────

best_loss = float('inf')
best_params = None

for _ in range(20):
    params = [
        np.random.uniform(0.8,1.2),
        np.random.uniform(0.8,1.2)
    ]

    loss = compute_loss(params)

    if loss < best_loss:
        best_loss = loss
        best_params = params

    print("Trial:", params, "Loss:", loss)

print("\nBEST PARAMS:", best_params)

# ─────────────────────────────────────────────────────────────
# FINAL MODEL RUN
# ─────────────────────────────────────────────────────────────

model_cpi = run_model(*best_params)

scale = real_cpi[0] / model_cpi[0]
model_cpi *= scale

model_inf = (model_cpi[1:] / model_cpi[:-1]) - 1

n = min(len(model_inf), len(real_inflation))
model_inf = model_inf[:n]
real_inf = real_inflation[:n]

# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(10,5))
plt.plot(real_inf, label="Real Inflation")
plt.plot(model_inf, label="Model Inflation")
plt.legend()
plt.title("Model vs Real Inflation")
plt.show()

# ─────────────────────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────────────────────

future_months = 12
forecast = model_inf[-future_months:]

print("\nNext 12 months inflation forecast (%):")
print(forecast * 100)



