"""
tests/test_indian_statarb.py
----------------------------
Pytest test suite validating all 10 stages of the Indian Statistical Arbitrage
Framework on real Indian NSE equity market data.
"""

import sys
import os
import importlib
from pathlib import Path

# Add project root to sys.path for test runner
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import pytest
import numpy as np
import pandas as pd

_data_col = importlib.import_module("01_data_collection")
_data_clean = importlib.import_module("02_data_cleaning")
_univ = importlib.import_module("03_universe")
_pair_sel = importlib.import_module("04_pair_selection")
_model = importlib.import_module("05_model")
_strat = importlib.import_module("06_strategy")
_backtest = importlib.import_module("07_backtest")
_valid = importlib.import_module("08_validation")
_analysis = importlib.import_module("09_analysis")
_verdict = importlib.import_module("10_final_verdict")


@pytest.fixture(scope="module")
def nse_market_data():
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HINDALCO.NS", "NATIONALUM.NS"]
    raw = _data_col.download_nse_prices(tickers, start="2022-01-01")
    clean = _data_clean.clean_missing_data(raw)
    log_p = _data_clean.compute_log_prices(clean)
    sector_map = _data_col.get_sector_mapping()
    return clean, log_p, sector_map


# -----------------------------------------------------------------------------
# STAGE 01 TESTS: Data Collection
# -----------------------------------------------------------------------------
def test_stage01_constituents_and_sectors():
    nifty50 = _data_col.get_nifty50_constituents()
    assert len(nifty50) >= 50
    assert all(t.endswith(".NS") for t in nifty50)

    sector_map = _data_col.get_sector_mapping()
    assert "HDFCBANK.NS" in sector_map
    assert sector_map["HDFCBANK.NS"] == "Private_Banking"
    assert "HINDALCO.NS" in sector_map
    assert sector_map["HINDALCO.NS"] == "Metals_Mining"


def test_stage01_real_nse_prices(nse_market_data):
    clean, log_p, _ = nse_market_data
    assert len(clean) > 200
    assert clean.shape[1] >= 2
    assert (clean > 0).all().all()


# -----------------------------------------------------------------------------
# STAGE 02 TESTS: Data Cleaning
# -----------------------------------------------------------------------------
def test_stage02_cleaning_and_outliers(nse_market_data):
    clean, log_p, _ = nse_market_data
    assert not clean.isna().any().any()
    assert not log_p.isna().any().any()

    rets = _data_clean.compute_returns_matrix(clean)
    assert len(rets) == len(clean) - 1
    winsorized = _data_clean.winsorize_returns(rets)
    assert winsorized.shape == rets.shape


# -----------------------------------------------------------------------------
# STAGE 03 TESTS: Universe Selection
# -----------------------------------------------------------------------------
def test_stage03_liquidity_and_sector_filter(nse_market_data):
    clean, _, sector_map = nse_market_data
    liquid = _univ.filter_by_liquidity(clean, min_adtv_cr=1.0)
    assert liquid.shape[1] > 0

    sector_pairs = _univ.filter_pairs_by_sector(clean.columns.tolist(), sector_map)
    assert len(sector_pairs) > 0
    for a, b in sector_pairs:
        assert sector_map[a] == sector_map[b]


# -----------------------------------------------------------------------------
# STAGE 04 TESTS: Pair Selection & Cointegration
# -----------------------------------------------------------------------------
def test_stage04_cointegration_screening(nse_market_data):
    _, log_p, sector_map = nse_market_data
    candidates = _pair_sel.screen_cointegrated_pairs(
        log_p,
        min_correlation=0.40,
        significance=0.15,
        min_half_life=1.0,
        max_half_life=90.0,
        sector_map=sector_map,
    )
    assert len(candidates) >= 1
    for c in candidates:
        assert 1.0 <= c.half_life <= 90.0


# -----------------------------------------------------------------------------
# STAGE 05 TESTS: Model Estimation (OLS vs Kalman Filter)
# -----------------------------------------------------------------------------
def test_stage05_kalman_filter_and_spread(nse_market_data):
    _, log_p, _ = nse_market_data
    cols = log_p.columns.tolist()
    y = log_p[cols[0]]
    x = log_p[cols[1]]

    ols_res = _model.fit_static_ols(y, x)
    assert hasattr(ols_res, "beta")
    assert hasattr(ols_res, "r_squared")

    kf_df = _model.run_kalman_filter(y, x, delta=1e-4)
    assert "beta" in kf_df.columns
    assert "spread" in kf_df.columns
    assert not kf_df["beta"].isna().any()

    z_scores = _model.compute_rolling_zscore(kf_df["spread"], window=20)
    assert len(z_scores) == len(y)


# -----------------------------------------------------------------------------
# STAGE 06 TESTS: Strategy Logic & Position Sizing
# -----------------------------------------------------------------------------
def test_stage06_strategy_signals_and_weights():
    z = pd.Series([0.0, 1.2, 2.1, 2.3, 0.4, -2.2, -0.3, 0.0])
    beta = pd.Series(1.2, index=z.index)

    positions, _ = _strat.generate_exit_signals(z, entry_z=2.0, exit_rules=_strat.ExitRules(exit_z=0.5))
    assert positions.iloc[2] == -1  # Short spread
    assert positions.iloc[3] == -1  # Hold short spread
    assert positions.iloc[4] == 0   # Take profit exit
    assert positions.iloc[5] == 1   # Long spread

    weights = _strat.calculate_cash_neutral_weights(positions, beta)
    assert "weight_a" in weights.columns
    assert "weight_b" in weights.columns
    active_idx = positions != 0
    gross = weights.loc[active_idx, "weight_a"].abs() + weights.loc[active_idx, "weight_b"].abs()
    assert np.allclose(gross, 1.0)


# -----------------------------------------------------------------------------
# STAGE 07 TESTS: Backtest Engine & Indian Transaction Costs
# -----------------------------------------------------------------------------
def test_stage07_indian_transaction_costs_and_backtest(nse_market_data):
    _, log_p, _ = nse_market_data
    tax = _backtest.IndianTaxCostBreakdown()
    assert 5.0 <= tax.total_bps_one_way <= 10.0

    cols = log_p.columns.tolist()
    cfg = _backtest.BacktestEngineConfig(
        entry_z=2.0,
        exit_z=0.5,
        cost_model=tax,
        base_slippage_bps=2.0,
    )
    res = _backtest.run_pair_backtest(log_p[cols[0]], log_p[cols[1]], config=cfg)
    assert hasattr(res, "sharpe")
    assert hasattr(res, "equity_curve")
    assert len(res.returns) == len(log_p)


# -----------------------------------------------------------------------------
# STAGE 08 TESTS: Validation (Walk-Forward & OOS)
# -----------------------------------------------------------------------------
def test_stage08_walk_forward_validation(nse_market_data):
    _, log_p, _ = nse_market_data
    cols = log_p.columns.tolist()
    wf_res = _valid.run_walk_forward_validation(
        log_p[cols[0]], log_p[cols[1]],
        train_days=150, test_days=40,
    )
    assert wf_res.n_windows >= 1
    assert len(wf_res.oos_returns) > 0

    oos_comp = _valid.evaluate_out_of_sample(log_p[cols[0]], log_p[cols[1]], split_ratio=0.70)
    assert hasattr(oos_comp, "is_sharpe")
    assert hasattr(oos_comp, "oos_sharpe")


# -----------------------------------------------------------------------------
# STAGE 09 TESTS: Risk Ratios, Drawdowns & Regimes
# -----------------------------------------------------------------------------
def test_stage09_analysis_modules(nse_market_data):
    _, log_p, _ = nse_market_data
    cols = log_p.columns.tolist()
    cfg = _backtest.BacktestEngineConfig(entry_z=2.0, exit_z=0.5)
    res = _backtest.run_pair_backtest(log_p[cols[0]], log_p[cols[1]], config=cfg)

    ratios = _analysis.compute_risk_adjusted_ratios(res.returns, risk_free_rate=0.065)
    assert hasattr(ratios, "sortino_ratio")
    assert hasattr(ratios, "calmar_ratio")

    dd_stats = _analysis.analyze_drawdown_periods(res.returns)
    assert dd_stats.max_drawdown_pct <= 0.0

    trade_stats = _analysis.compute_trade_statistics(res.returns, res.positions, res.turnover)
    assert trade_stats.total_trades >= 0

    regimes = _analysis.analyze_market_regimes(res.returns)
    assert hasattr(regimes, "bull_sharpe")
    assert hasattr(regimes, "bear_sharpe")


# -----------------------------------------------------------------------------
# STAGE 10 TESTS: Final Verdict & Hypothesis Testing
# -----------------------------------------------------------------------------
def test_stage10_hypothesis_and_verdict(nse_market_data):
    _, log_p, _ = nse_market_data
    cols = log_p.columns.tolist()
    cfg = _backtest.BacktestEngineConfig(entry_z=2.0, exit_z=0.5)
    res = _backtest.run_pair_backtest(log_p[cols[0]], log_p[cols[1]], config=cfg)

    hyp_res = _verdict.run_alpha_hypothesis_test(res.returns, significance_level=0.05)
    assert hasattr(hyp_res, "t_statistic")

    bmk_comp = _verdict.compare_with_nifty_benchmark(res.returns)
    assert hasattr(bmk_comp, "market_beta")

    tear_sheet = _verdict.generate_final_verdict_report(
        sharpe=1.2,
        max_drawdown_pct=-6.5,
        win_rate_pct=58.0,
        profit_factor=1.6,
        t_stat=2.1,
        market_beta=0.04,
        oos_sharpe_retention=0.75,
    )
    assert tear_sheet.production_score >= 70
    assert "DEPLOY READY" in tear_sheet.production_verdict
