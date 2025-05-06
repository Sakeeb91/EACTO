# ENTROPY-ADAPTIVE CYBERNETIC THRESHOLD OPTIMIZER (Project EACTO)

## Core Idea & Philosophy

Project EACTO is built on the premise that financial markets are complex adaptive systems prone to sudden shifts in dynamics (the "node effect", which refers to regime changes or contagion). Traditional risk models with static thresholds often fail during these shifts. EACTO uses:

- **Entropy**: To quantify current market uncertainty and complexity. Higher entropy signals a more unpredictable or volatile regime.
- **Cybernetics**: The MPC framework acts as the cybernetic controller, using feedback (actual vs. predicted outcomes, current entropy) to adjust its control actions (the risk threshold α<sub>t</sub>).
- **Self-Regulation**: The system aims to automatically adjust α<sub>t</sub> to achieve a pre-defined risk objective (e.g., a target rate of VaR exceedances) rather than relying on a fixed percentile like "95% VaR" whose actual dollar value can become meaningless quickly.

## What EACTO Will Predict/Control

EACTO will not primarily "predict" a future price point. Instead, it will:

- Continuously estimate market entropy as a proxy for chaos/regime shifts.
- Dynamically determine an optimal risk threshold (α<sub>t</sub>) for a portfolio or a specific trading strategy.
- Predict the consequences of different risk threshold choices over a short horizon, considering current market entropy and expected dynamics.
- Select the threshold that optimizes a predefined cost function.

The ultimate goal is to maintain a desired risk profile even as market dynamics shift, effectively self-regulating the risk exposure without requiring a fixed percentile calculated on a static basis.

## Data Requirements & Sourcing

- **Primary Data**: High-frequency (e.g., 1-minute or 5-minute) ticker data (OHLCV) for the asset(s) or portfolio components being managed.
- **Source**: Financial data providers (e.g., Alpha Vantage, Polygon.io, IEX Cloud, or Yahoo Finance for prototyping).
- **Duration**: Several years of data, covering different market regimes (bull, bear, crisis periods) for robust model development and backtesting.

## Key Components

### A. Entropy Calculation Module

- **Input**: Recent window of price returns (e.g., log returns).
- **Method**:
  - **Shannon Entropy**: Calculated on the discretized distribution of returns within a rolling window.
  - **Approximate Entropy (ApEn) or Sample Entropy (SampEn)**: Measure regularity and predictability in the time series.
- **Output**: A time series of entropy values (e.g., `H_t`).

### B. System Dynamics Model (Predictive Model within MPC)

- **Purpose**: Predicts the likely evolution of the system over a short future horizon.
- **Inputs**:
  - Recent historical returns
  - Current entropy `H_t`
  - Other relevant features (e.g., realized volatility, trading volume, VIX)
  - Proposed future risk thresholds `α_τ`
- **Outputs**: 
  - Predicted probability of portfolio loss exceeding `α_τ`
  - Or, predicted portfolio return distribution
- **Model Options**:
  - GARCH-family models
  - Vector Autoregression (VAR)
  - Quantile Regression
  - Machine Learning Models (Random Forest, Gradient Boosting)

### C. Model-Predictive Control (MPC) Optimizer

- **Control Variable**: The sequence of risk thresholds `α_τ` 
- **Objective Function**:
  `J = Σ_{τ=t}^{t+N-1} [λ * ε_τ² + γ * Δα_τ²]`
  - `ε_τ`: Error term (difference between actual/predicted breach and target)
  - `Δα_τ`: Change in risk threshold
  - `λ`, `γ`: Weighting factors
- **Constraints**:
  - `α_τ > 0` (Risk threshold must be positive)
  - `α_τ_min ≤ α_τ ≤ α_τ_max` (Practical bounds)
  - Entropy-based constraints: `α_τ ≤ f(H_t)`
- **Output**: The optimal `α_t` for the current period.

### D. Feedback Loop & Self-Regulation

1. **Observe**: Collect new market data, calculate current portfolio loss/return.
2. **Update Entropy**: Calculate current entropy `H_t`.
3. **Evaluate Previous Action**: Check if a breach occurred with the previous threshold.
4. **MPC Cycle**:
   - Feed data into the System Dynamics Model
   - Optimize future thresholds by minimizing the objective function
5. **Implement**: Apply the optimal threshold `α_t` for the next period.
6. **Repeat** for next time step.

## Addressing "Node Effects" (Market Regime Shifts)

- **Entropy as a Detector**: High entropy values signal unusual market conditions.
- **MPC's Predictive Horizon**: Anticipates immediate consequences of different threshold choices.
- **Adaptive Constraints/Penalties**: Entropy can influence constraints or weights in the cost function.
- **System Dynamics Model Re-estimation**: Regularly update the predictive model.

## Evaluation Metrics

- **Kupiec's Proportion of Failures (POF) Test**: Checks if observed breaches match target probability.
- **Christoffersen's Conditional Coverage Test**: Tests for correct breach frequency and independence.
- **Dynamic Quantile Test**: For conditional autoregressive VaR models.
- **Magnitude of Exceedances**: Severity of tail events beyond threshold.
- **Stability of α_t**: Smooth adaptation without erratic changes.
- **Computational Cost**: Ensure optimization is fast enough for operational frequency.
- **Portfolio Performance**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown (if applicable).

## Potential for Innovation

- **True Self-Regulation**: Adjusts the risk threshold based on forward-looking optimization.
- **Principled Integration of Entropy**: Uses entropy as direct input to the MPC's objective function.
- **Handling "Unknown Unknowns"**: Entropy's model-free nature captures emergent market anomalies.
- **Crisis Performance**: Maintain target risk profile more effectively during market turbulence. 