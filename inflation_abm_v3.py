"""
inflation_abm_v3.py — Phase 2 refined with real Indian CPI data
================================================================
What changed from v2 and why:

1. SEASONAL SHOCKS — real data shows Jun-Jul inflation is 4x higher than
   Jan-Dec. We now apply month-specific shock multipliers derived directly
   from the real CPI seasonal pattern. This is the biggest missing piece.

2. FOOD VOLATILITY — real monthly std is 8.48% annualized. Food/vegetables
   dominate this. We increase food firm shock variance significantly.

3. CALIBRATION TARGETS — we now target YoY moments (mean=4.91%, std=1.69%,
   AC1=0.898) not MoM moments. YoY is what the RBI uses and what matters
   for policy. MoM is too noisy (dominated by seasonal food swings).

4. TAYLOR RULE — softer response so policy rate doesn't hit ceiling.

Engineering analogy:
  - Seasonal multipliers = time-varying gain on the shock input
  - YoY vs MoM = low-pass filter on the inflation signal
  - Calibration = system identification against real I/O data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(seed=42)

# ── CONFIG ────────────────────────────────────────────────────────────────
N_F  = 500
N_H  = 2000
T    = 120    # months after burn-in
BURN = 24

SECTORS = [
    # name, n_firms, markup_mean, markup_sigma, cost_sens
    ###Understanding these together allows a company to set a profitable markup(markup_mean), understand if they are achieving it (markup_sigma), 
    ###and ensure that the price isn't too high for the customer (cost_sens).
    
    ("Food",          175, 0.08, 0.04, 0.30),
    ("Fuel",           50, 0.12, 0.05, 0.60),
    ("Manufacturing", 100, 0.06, 0.03, 0.20),
    ("Services",      125, 0.04, 0.02, 0.08),
    ("Housing",        50, 0.02, 0.01, 0.03),
]

assert sum(s[1] for s in SECTORS) == N_F

INFLATION_TARGET = 0.04 / 12
POLICY_RATE_INIT = 0.065 / 12

# Real Indian CPI weights (Food 46%, Fuel 7%, Mfg 17%, Services 22%, Housing 8%)
REAL_WEIGHTS = np.array([0.46, 0.07, 0.17, 0.22, 0.08])

# ── SEASONAL MULTIPLIERS ──────────────────────────────────────────────────
# Derived directly from real RBI data (annualized monthly averages)
# Converted to monthly multipliers on the base shock
# Logic: if Jul averages +16.6% ann and mean is +5.4% ann,
#        Jul multiplier = 16.6/5.4 ≈ 3.1x the base shock
# These apply to FOOD sector only (food drives seasonality)
SEASONAL_FOOD_MULT = np.array([
    0.0,   # Jan — prices fall after harvest, net negative → no upward shock
    0.2,   # Feb — low
    0.5,   # Mar — rabi harvest approaching, mild
    1.8,   # Apr — pre-harvest tightness
    1.5,   # May — continued tightness
    2.2,   # Jun — kharif sowing, supply tight
    3.2,   # Jul — peak seasonal pressure
    1.3,   # Aug — monsoon supply uncertainty
    0.6,   # Sep — kharif harvest begins
    1.8,   # Oct — festive demand
    1.0,   # Nov — post-harvest
    0.0,   # Dec — harvest surplus, prices fall
])

# Seasonal base offset — months where prices systematically fall
SEASONAL_FOOD_BASE = np.array([
    -0.008,  # Jan
    +0.001,  # Feb
    +0.003,  # Mar
    +0.006,  # Apr
    +0.005,  # May
    +0.007,  # Jun
    +0.010,  # Jul
    +0.004,  # Aug
    +0.002,  # Sep
    +0.006,  # Oct
    +0.003,  # Nov
    -0.010,  # Dec
])

# ── INIT FIRMS ────────────────────────────────────────────────────────────
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

# wage share by sector
wage_shares = np.where(sector_id == 3, 0.60,
              np.where(sector_id == 0, 0.40,
              np.where(sector_id == 1, 0.20, 0.35)))

# ── INIT HOUSEHOLDS ───────────────────────────────────────────────────────
def beta_sample(a, b, size):
    g1 = rng.gamma(a, 1.0, size)
    g2 = rng.gamma(b, 1.0, size)
    return g1 / (g1 + g2)

savings_rate  = beta_sample(2, 5, N_H)
# Slightly lower persistence than v2 — real MoM AC1=0.37, YoY AC1=0.90
# The YoY persistence comes from the seasonal cycle, not expectation stickiness alone
expect_weight = rng.uniform(0.70, 0.85, N_H)
income        = rng.lognormal(np.log(50), 0.4, N_H)
expectation   = np.full(N_H, INFLATION_TARGET)

# ── CPI ───────────────────────────────────────────────────────────────────
def compute_cpi(prices):
    return float(sum(REAL_WEIGHTS[s] * prices[sector_id == s].mean()
                     for s in range(len(SECTORS))))

# ── STATE ─────────────────────────────────────────────────────────────────
cpi_history        = []
inflation_history  = []
rate_history       = []
sector_price_index = {s[0]: [] for s in SECTORS}

# 3-month smoother for expectations
inf_buffer     = [INFLATION_TARGET] * 3
prev_cpi       = compute_cpi(price)
policy_rate    = POLICY_RATE_INIT
avg_inf_smooth = INFLATION_TARGET

# persistent fuel shock state
fuel_shock_remaining = 0
fuel_shock_magnitude = 0.0

print("Running simulation...")
print(f"{'Tick':>5}  {'YoY%':>8}  {'MoM(ann)%':>10}  {'Rate%':>7}  {'Month':>6}")
print("-" * 48)

for t in range(T + BURN):

    # current calendar month (0=Jan, 11=Dec)
    month_idx = t % 12

    # ── SEASONAL FOOD SHOCK ───────────────────────────
    # This is the key mechanism — food prices follow harvest calendar
    # High variance in Jul (monsoon uncertainty), low in Jan (harvest surplus)
    seasonal_base  = SEASONAL_FOOD_BASE[month_idx]
    seasonal_mult  = SEASONAL_FOOD_MULT[month_idx]
    # food shock = seasonal base + scaled random noise
    # noise variance is higher in monsoon months (Jun-Sep)
    monsoon_var = 0.018 if month_idx in [5,6,7,8] else 0.008
    food_noise  = rng.normal(0, monsoon_var, (sector_id==0).sum())
    food_shock  = seasonal_base + seasonal_mult * food_noise

    # ── PERSISTENT FUEL SHOCK ────────────────────────
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

    # ── 1. HOUSEHOLDS ────────────────────────────────
    expectation  = expect_weight * avg_inf_smooth + (1 - expect_weight) * expectation
    avg_expect   = float(expectation.mean())

    real_rate  = policy_rate - avg_expect
    rate_drag  = np.clip(real_rate * 1.2, -0.03, 0.03)
    budgets    = (1 - savings_rate) * income * (1 - rate_drag)
    total_demand = float(budgets.sum())

    # ── 2. GOODS MARKET ──────────────────────────────
    inv_p         = 1.0 / (price + 1e-9)
    shares        = inv_p / inv_p.sum()
    actual_demand = shares * total_demand
    excess_demand = np.clip(
        (actual_demand - expected_demand) / (expected_demand + 1e-9),
        -0.3, 0.3)

    # ── 3. COST SHOCKS ───────────────────────────────
    # background cost pressure for all firms (wages, inputs)
    base_shock = rng.normal(0.004, 0.004, N_F)

    # sector-specific shocks
    base_shock[sector_id == 0] += food_shock            # food: seasonal
    base_shock[sector_id == 1] += current_fuel_shock    # fuel: persistent shocks

    # wage-price spiral
    wage_growth     = 0.50 * avg_inf_smooth + 0.002
    firm_wage_cost  = firm_wage_cost * (1 + np.clip(wage_growth, 0, 0.010))
    effective_shock = base_shock + wage_shares * wage_growth

    # ── 4. FIRMS: price update ───────────────────────
    price_change = (
          markup_sens * excess_demand          # demand pull
        + cost_sensitivity * effective_shock   # cost push
        + 0.15 * avg_expect                    # expectations
    )
    price = price * (1 + np.clip(price_change, -0.025, 0.030))
    expected_demand = 0.8 * expected_demand + 0.2 * actual_demand

    # ── 5. CPI & INFLATION ───────────────────────────
    current_cpi = compute_cpi(price)
    inflation   = (current_cpi / prev_cpi) - 1.0
    prev_cpi    = current_cpi

    inf_buffer.append(inflation)
    inf_buffer.pop(0)
    avg_inf_smooth = float(np.mean(inf_buffer))

    # ── 6. CENTRAL BANK ──────────────────────────────
    # softer Taylor rule — responds to 6-month smoothed inflation
    # not monthly noise (more realistic — RBI doesn't react to one month)
    pi_gap      = avg_inf_smooth - INFLATION_TARGET
    output_gap  = float(excess_demand.mean())
    policy_rate = float(np.clip(
        POLICY_RATE_INIT + 1.2 * pi_gap + 0.3 * output_gap,
        0.04/12, 0.10/12))   # cap at 10% — more realistic for India

    # ── 7. WAGES ─────────────────────────────────────
    income = income * (1 + np.clip(wage_growth, 0, 0.010))

    # ── RECORD ───────────────────────────────────────
    if t >= BURN:
        cpi_history.append(current_cpi)
        inflation_history.append(inflation)
        rate_history.append(policy_rate)
        for s_id, (name, *_) in enumerate(SECTORS):
            sector_price_index[name].append(price[sector_id == s_id].mean())

        tick = t - BURN
        if tick % 12 == 0:
            ann_mom = ((1+inflation)**12 - 1)*100
            r       = policy_rate*12*100
            mn      = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec'][month_idx]
            # compute YoY if we have enough history
            if len(cpi_history) > 12:
                yoy = (cpi_history[-1]/cpi_history[-13] - 1)*100
            else:
                yoy = ann_mom
            print(f"{tick:>5}  {yoy:>7.2f}%  {ann_mom:>9.2f}%  {r:>6.2f}%  {mn:>6}")

print("\nDone.")

# ── STATS ─────────────────────────────────────────────────────────────────
inf_arr  = np.array(inflation_history)
cpi_arr  = np.array(cpi_history)
ann_mom  = ((1 + inf_arr)**12 - 1) * 100

# YoY from simulated CPI
yoy_sim  = (cpi_arr[12:] / cpi_arr[:-12] - 1) * 100

print("\n── YOUR SIMULATION ─────────────────────────────────────")
print(f"  Mean YoY inflation      : {yoy_sim.mean():.2f}%   real=4.91%")
print(f"  Std  YoY inflation      : {yoy_sim.std():.2f}%   real=1.69%")
print(f"  Mean MoM (ann)          : {ann_mom.mean():.2f}%   real=5.40%")
print(f"  Std  MoM (ann)          : {ann_mom.std():.2f}%   real=8.48%")
for lag in [1,3]:
    ac_sim  = np.corrcoef(yoy_sim[:-lag], yoy_sim[lag:])[0,1]
    print(f"  YoY Autocorr lag {lag}      : {ac_sim:.3f}   real(lag{lag})={'0.898' if lag==1 else '0.669'}")
print("─────────────────────────────────────────────────────────")

# ── REAL DATA for comparison ───────────────────────────────────────────────
import pandas as pd
rdf = pd.read_csv('/Users/pnglinkx/Desktop/CPIndex_Jan13-To-Dec25(Detailed).csv', skiprows=1)
rdf = rdf[['Year','Month','Combined']].dropna()
rdf['Combined'] = pd.to_numeric(rdf['Combined'], errors='coerce')
month_map = {m:i+1 for i,m in enumerate(['January','February','March','April','May',
    'June','July','August','September','October','November','December'])}
rdf['mn'] = rdf['Month'].map(month_map)
rdf = rdf.sort_values(['Year','mn']).reset_index(drop=True)
real_cpi = rdf['Combined'].values
real_yoy = (real_cpi[12:] / real_cpi[:-12] - 1) * 100

# ── PLOT ──────────────────────────────────────────────────────────────────
months_sim  = np.arange(T)
months_real = np.arange(len(real_yoy))

fig = plt.figure(figsize=(15, 12), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,:])

COLORS = {"Food":"#f59e0b","Fuel":"#ef4444",
          "Manufacturing":"#3b82f6","Services":"#10b981","Housing":"#a78bfa"}
TXT="#e5e5e5"; GRID="#222"; ACC="#f59e0b"

def style(ax, title):
    ax.set_facecolor("#161616")
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.set_title(title, color=TXT, fontsize=9, pad=8, fontfamily="monospace")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)

# 1. YoY — sim vs real overlaid
ax1.plot(months_real, real_yoy, color="#60a5fa", lw=1.2, alpha=0.8, label="Real CPI (YoY%)")
ax1.plot(months_sim[:len(yoy_sim)], yoy_sim, color=ACC, lw=1.2, alpha=0.9, label="Simulated (YoY%)")
ax1.axhline(4.0, color="#fff", lw=0.7, ls="--", alpha=0.3, label="4% target")
ax1.axhline(6.0, color="#ef4444", lw=0.5, ls=":", alpha=0.3)
style(ax1, "YoY INFLATION — Simulated vs Real Indian CPI")
ax1.legend(fontsize=7, facecolor="#161616", labelcolor=TXT, framealpha=0.5)
ax1.set_xlabel("Months", fontsize=8)

# 2. MoM distribution: sim vs real
real_mom_ann = ((1 + np.diff(real_cpi)/real_cpi[:-1])**12 - 1)*100
ax2.hist(real_mom_ann, bins=30, color="#60a5fa", alpha=0.6, label="Real", density=True)
ax2.hist(ann_mom,      bins=30, color=ACC,       alpha=0.6, label="Sim",  density=True)
style(ax2, "MoM INFLATION DISTRIBUTION (ann%)")
ax2.legend(fontsize=7, facecolor="#161616", labelcolor=TXT, framealpha=0.5)

# 3. Sector price levels
for name, prices in sector_price_index.items():
    p = np.array(prices)
    ax3.plot(months_sim, p/p[0], color=COLORS[name], lw=1.0, label=name)
style(ax3, "SECTOR PRICE LEVELS — normalized")
ax3.legend(fontsize=7, facecolor="#161616", labelcolor=TXT, framealpha=0.5)

# 4. Seasonal pattern: sim vs real
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
real_seasonal = [real_mom_ann[i::12].mean() for i in range(12)]
sim_seasonal  = [ann_mom[i::12].mean()      for i in range(12)]
x = np.arange(12)
w = 0.35
ax4.bar(x - w/2, real_seasonal, w, color="#60a5fa", alpha=0.8, label="Real")
ax4.bar(x + w/2, sim_seasonal,  w, color=ACC,       alpha=0.8, label="Simulated")
ax4.set_xticks(x)
ax4.set_xticklabels(month_names, fontsize=7)
ax4.axhline(0, color="#fff", lw=0.5, alpha=0.3)
style(ax4, "SEASONAL PATTERN — Real vs Simulated (avg MoM ann%)")
ax4.legend(fontsize=7, facecolor="#161616", labelcolor=TXT, framealpha=0.5)

fig.suptitle("ABM INFLATION  ·  V3  — calibrated to real Indian CPI (Jan 2013–Dec 2025)",
             color=TXT, fontsize=10, fontfamily="monospace", y=0.99)
plt.savefig("abm_v3.png", dpi=150,
            bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print("Chart saved.")
