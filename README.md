# PROJECT-1
# Indian Statistical Arbitrage Engine (NSE / NIFTY 500)

A quantitative pairs-trading and statistical arbitrage research framework re-architected for the **Indian Equity Market (NSE / NIFTY 500)** across 10 modular, institutional-grade stages.

The engine operates **100% on real Indian equity market data**, incorporating time-varying dynamic hedge ratios ($\beta_t$), intra-sector fundamental constraints, realistic Indian statutory transaction costs (STT, GST, Stamp Duty, SEBI & NSE charges), and walk-forward cross-validation.

---

## 1. 10-Stage Pipeline Architecture

```
INDIAN STATISTICAL ARBITRAGE
│
├── 01_data_collection
│   ├── nifty500_constituents.py   # NIFTY 50 / 500 scrapers & NSE sector taxonomy (.NS)
│   ├── price_data.py              # Yahoo Finance NSE batched download & local disk caching
│   └── corporate_actions.py       # Stock splits, bonuses & dividend adjustments
│
├── 02_data_cleaning
│   ├── missing_values.py          # NSE calendar alignment & gap filtration
│   ├── outliers.py                # Circuit breaker (5%/10%/20%) spike filtration & winsorization
│   └── adjusted_prices.py         # Unified log-prices & returns matrices
│
├── 03_universe
│   ├── liquidity_filter.py        # ADTV turnover threshold (min INR 5-10 Crore / day)
│   ├── sector_filter.py           # Intra-sector pairing (Private Banks, IT, Metals, etc.)
│   └── survivorship_control.py    # Point-in-time universe snapshots
│
├── 04_pair_selection
│   ├── correlation.py             # Return & price correlation pre-filtering
│   ├── cointegration.py          # Two-step Engle-Granger screening
│   ├── ADF.py                     # Augmented Dickey-Fuller stationarity tests (p < 0.05)
│   └── half_life.py               # Ornstein-Uhlenbeck mean-reversion speed (1 <= t_half <= 60 d)
│
├── 05_model
│   ├── OLS_hedge_ratio.py         # Static OLS beta baseline & regression diagnostics
│   ├── Kalman_filter.py           # 2D State-Space online dynamic beta tracking
│   ├── spread.py                  # Dynamic spread computation
│   └── z_score.py                 # Rolling & adaptive Z-score generation
│
├── 06_strategy
│   ├── entry.py                   # Hysteresis entry triggers (+-2.0 Z) & blowout filters
│   ├── exit.py                    # Mean-reversion exit (|Z| <= 0.5) & stop-loss rules
│   ├── position_sizing.py         # Cash-neutral / Beta-neutral Indian lot sizing
│   └── risk_limits.py             # Gross leverage caps & margin buffer rules
│
├── 07_backtest
│   ├── execution.py               # Vectorized next-day execution (no lookahead bias)
│   ├── transaction_costs.py       # Complete Indian tax schedule (STT, NSE, SEBI, Stamp, GST)
│   ├── slippage.py                # Square-root market impact & slippage modeling
│   └── PnL.py                     # Mark-to-market equity curve & cash flow tracker
│
├── 08_validation
│   ├── walk_forward.py            # Rolling out-of-sample (OOS) cross-validation
│   ├── out_of_sample.py           # Untouched holdout validation & Sharpe retention
│   └── robustness.py              # Parameter sensitivity heatmaps & Monte Carlo tests
│
├── 09_analysis
│   ├── returns.py                 # CAGR, cumulative returns & monthly matrix
│   ├── sharpe.py                  # Sharpe, Sortino, Calmar & Omega ratios
│   ├── drawdown.py                # Max Drawdown & underwater duration analysis
│   ├── trade_statistics.py        # Win rate, profit factor, holding duration
│   └── regime_analysis.py         # Bull/Bear & High India VIX regime decomposition
│
└── 10_final_verdict
    ├── hypothesis_test.py         # Alpha statistical significance (Student's t-test)
    ├── benchmark_comparison.py    # Beta, Jensen's Alpha & Information Ratio vs NIFTY
    └── conclusion.py              # Institutional score (0-100) & Go/No-Go verdict
```

---

## 2. Quantitative Methodology & Mathematical Core

### A. Intra-Sector Pair Selection ([03_universe](file:///c:/Users/Parth/Downloads/statarb-pairs-trading-2-main/statarb-pairs-trading-2-main/03_universe/) & [04_pair_selection](file:///c:/Users/Parth/Downloads/statarb-pairs-trading-2-main/statarb-pairs-trading-2-main/04_pair_selection/))
- **Liquidity Threshold:** Stocks must satisfy $\text{ADTV} \ge \text{INR } 5\text{ Crore}$ to avoid market impact and circuit lockups.
- **Sector Pairing:** Pairs are constructed strictly within homogeneous NSE sectors (e.g. `HDFCBANK.NS` vs `ICICIBANK.NS`, `TCS.NS` vs `INFY.NS`, `HINDALCO.NS` vs `NATIONALUM.NS`) to prevent data-mining bias.
- **Engle-Granger Cointegration & ADF Test:** Regresses $\log(P_{A,t}) = \beta \log(P_{B,t}) + \alpha + \epsilon_t$ and runs the Augmented Dickey-Fuller (ADF) test on the residual $\epsilon_t$ ($p < 0.05$).
- **Ornstein-Uhlenbeck (OU) Half-Life:** Fits $\Delta \epsilon_t = -\lambda \epsilon_{t-1} + \eta_t$ to ensure mean-reversion speed is within a tradeable horizon ($1 \le t_{1/2} = \frac{\ln 2}{\lambda} \le 60\text{ days}$).

### B. Dynamic Kalman Filter ([05_model](file:///c:/Users/Parth/Downloads/statarb-pairs-trading-2-main/statarb-pairs-trading-2-main/05_model/))
Instead of a static OLS beta, an online 2D State-Space Kalman filter continuously tracks time-varying hedge ratios:
$$\text{State Equation:} \quad \theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, W_t)$$
$$\text{Observation Equation:} \quad y_t = [x_t, 1] \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, V_t)$$
$$\text{Dynamic Spread:} \quad s_t = y_t - (\beta_t x_t + \alpha_t)$$

### C. Indian Statutory Transaction Costs ([07_backtest](file:///c:/Users/Parth/Downloads/statarb-pairs-trading-2-main/statarb-pairs-trading-2-main/07_backtest/))
Every backtested return incorporates the complete Indian market fee schedule:
- **Securities Transaction Tax (STT):** $0.025\%$ (intraday equity)
- **NSE Exchange Turnover Charge:** $0.00345\%$
- **SEBI Turnover Charge:** $0.0001\%$
- **Stamp Duty:** $0.003\%$
- **Brokerage:** $0.03\%$
- **GST:** $18\%$ on (Brokerage + Exchange Fees + SEBI charges)
- **All-in One-Way Friction:** $\approx 6.76\text{ bps per trade}$

### D. Walk-Forward Cross-Validation ([08_validation](file:///c:/Users/Parth/Downloads/statarb-pairs-trading-2-main/statarb-pairs-trading-2-main/08_validation/))
The backtest engine rolls sequential `Train` (e.g. 250 days) $\to$ `Test` (e.g. 60 days) windows. Parameters and filter states adapt in the training slice, and performance metrics are reported **exclusively from out-of-sample test periods with 1-day execution lag**.

---

## 3. Quickstart & CLI Usage

### Installation
```powershell
pip install -r requirements.txt
```

### Run Full 10-Stage Pipeline
```powershell
# Run across major NSE sectors from 2021 to present
python run_indian_statarb.py --sectors Private_Banking IT_Services Metals_Mining Automobile FMCG --start-date 2021-01-01
```

### Command-Line Arguments
| Argument | Default | Description |
|---|---|---|
| `--sectors` | `Private_Banking IT_Services ...` | NSE sectors to screen (Banking, IT, Metals, Auto, FMCG, Energy). |
| `--start-date` | `2021-01-01` | Backtest history start date (`YYYY-MM-DD`). |
| `--end-date` | `None` (latest) | Backtest history end date (`YYYY-MM-DD`). |
| `--min-corr` | `0.60` | Minimum correlation pre-filter threshold. |
| `--significance` | `0.05` | ADF cointegration $p$-value threshold ($5\%$). |
| `--min-adtv` | `5.0` | Minimum Average Daily Traded Value in INR Crores. |

---

## 4. Running the Test Suite

Run the automated pytest test suite covering all 10 stages:
```powershell
pytest -v
```

All 11 unit tests validate:
- Constituent & sector mappings (`HDFCBANK.NS`, `TCS.NS`, `HINDALCO.NS`, etc.).
- Calendar alignment & missing data handling.
- Sector and ADTV liquidity filters.
- Engle-Granger & ADF cointegration screening.
- Online Kalman filter state updates.
- Strategy state machine & cash-neutral lot sizing.
- Indian tax & fee calculation.
- Walk-forward cross-validation folds.
- Risk-adjusted ratios (Sharpe, Sortino, Calmar, Omega) & regime analysis.
- Alpha hypothesis tests ($t$-stat, $p$-val) and institutional readiness scorecard.
