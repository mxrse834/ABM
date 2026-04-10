# ABM

>>>>>GOAL :
Build an ABM that predicts inflation over time using real India data from MOSPI 
1)Get the data — MOSPI 
2)Build the ABM — 1M+ agents, simulate how prices evolve
3)Output — inflation prediction over time

>>>>>ABM Primitive(base) Model working :

1) Design:
Agents (firms, households, banks, CB(RBI)), their state variables, and behavioral rules.
    How to setup the state variables and behavioral rules(parameters are set at initialization and stay constant while state variables vary per iteration) :
        1.  What decisions does this agent make?
        2.  What info is needed to make that decision?
        3.  What minimal info can be used for this purpose?

2) Initialization
Draw parameters from chosen distributions.

Phase 3 — Tick loop (repeat T times, e.g. T=240 months)
For each tick t:

Firms observe last period excess demand EDi,t−1\text{ED}_{i,t-1}
EDi,t−1​ and update price via the markup equation

Households update inflation expectation π^i,t=λπt−1+(1−λ)π^i,t−1\hat\pi_{i,t} = \lambda \pi_{t-1} + (1-\lambda)\hat\pi_{i,t-1}
π^i,t​=λπt−1​+(1−λ)π^i,t−1​, compute consumption budget

Goods market: households sample k firms, allocate spending, firms record demand Di,tD_{i,t}
Di,t​
Labour market: firms post vacancies, matching function pairs workers to jobs, wages negotiate
Central bank observes πt\pi_t
πt​ and YtY_t
Yt​, sets rtr_t
rt​ via Taylor rule

Compute πt=Pt/Pt−1−1\pi_t = P_t/P_{t-1} - 1
πt​=Pt​/Pt−1​−1 from the Laspeyres price index

Record all macro aggregates

Phase 4 — Calibration via SMM
Define a moment vector mdatam_{\text{data}}
mdata​ from your target dataset (e.g. Indian CPI monthly series). Typical moments: [πˉ,std(π),autocorr(π,1),uˉ,std(u),corr(π,u)][\bar\pi, \text{std}(\pi), \text{autocorr}(\pi,1), \bar u, \text{std}(u), \text{corr}(\pi, u)]
[πˉ,std(π),autocorr(π,1),uˉ,std(u),corr(π,u)]. Your loss is:

L(θ)=(mˉdata−1R∑r=1Rmsim(θ,r))TW(mˉdata−1R∑r=1Rmsim(θ,r))\mathcal{L}(\theta) = \left(\bar m_{\text{data}} - \frac{1}{R}\sum_{r=1}^R m_{\text{sim}}(\theta, r)\right)^T W \left(\bar m_{\text{data}} - \frac{1}{R}\sum_{r=1}^R m_{\text{sim}}(\theta, r)\right)L(θ)=(mˉdata​−R1​r=1∑R​msim​(θ,r))TW(mˉdata​−R1​r=1∑R​msim​(θ,r))
where WW
W is a weighting matrix (identity or inverse-variance). You minimize L(θ)\mathcal{L}(\theta)
L(θ) with a gradient-free optimizer (CMA-ES works well — the objective is noisy due to stochastic simulations). R=500 trajectories per evaluation smooths the noise. This is computationally expensive — it's essentially the same structure as your MCMC/Kalman GPU problem.

Phase 5 — Forecasting
With θ∗\theta^*
θ∗ in hand, run R=1,000 fresh trajectories from the current state of the economy (set agent initial conditions from current data — current unemployment, current firm price dispersion, current interest rate). The forecast distribution is the empirical distribution of πT+h\pi_{T+h}
πT+h​ across trajectories at horizon hh
h. Report the fan chart (10th, 50th, 90th percentile paths).
