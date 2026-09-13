from __future__ import annotations
from dataclasses import dataclass
import importlib
import numpy as np
import pandas as pd

_kf_module = importlib.import_module("05_model.Kalman_filter")
_z_module = importlib.import_module("05_model.z_score")
_exit_module = importlib.import_module("06_strategy.exit")
_size_module = importlib.import_module("06_strategy.position_sizing")
_risk_module = importlib.import_module("06_strategy.risk_limits")

run_kalman_filter = _kf_module.run_kalman_filter
compute_rolling_zscore = _z_module.compute_rolling_zscore
generate_exit_signals = _exit_module.generate_exit_signals
ExitRules = _exit_module.ExitRules
calculate_cash_neutral_weights = _size_module.calculate_cash_neutral_weights
apply_risk_limits = _risk_module.apply_risk_limits
RiskLimitsConfig = _risk_module.RiskLimitsConfig

from .transaction_costs import calculate_indian_transaction_costs, IndianTaxCostBreakdown
from .slippage import calculate_slippage
from .PnL import compute_portfolio_pnl, compute_performance_metrics, BacktestEngineResult


@dataclass
class BacktestEngineConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float = 3.5
    max_hold_days: int | None = 20
    zscore_window: int = 20
    kalman_delta: float = 1e-4
    kalman_obs_var: float = 1e-3
    base_slippage_bps: float = 2.0
    cost_model: IndianTaxCostBreakdown | None = None
    risk_limits: RiskLimitsConfig | None = None
    annualization_factor: int = 252


def run_pair_backtest(
    log_a: pd.Series,
    log_b: pd.Series,
    config: BacktestEngineConfig | None = None,
) -> BacktestEngineResult:
    if config is None:
        config = BacktestEngineConfig()

    df = pd.concat([log_a, log_b], axis=1).dropna()
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]

    kf_res = run_kalman_filter(
        y, x,
        delta=config.kalman_delta,
        observation_var=config.kalman_obs_var,
    )
    spread = kf_res["spread"]
    beta_series = kf_res["beta"]

    z_scores = compute_rolling_zscore(spread, window=config.zscore_window)

    exit_rules = ExitRules(
        exit_z=config.exit_z,
        stop_loss_z=config.stop_loss_z,
        max_hold_days=config.max_hold_days,
    )
    positions, _ = generate_exit_signals(
        z_scores,
        entry_z=config.entry_z,
        exit_rules=exit_rules,
    )

    weights = calculate_cash_neutral_weights(positions, beta_series)
    if config.risk_limits:
        weights = apply_risk_limits(weights, config.risk_limits)

    ret_a = y.diff().apply(np.expm1).fillna(0.0)
    ret_b = x.diff().apply(np.expm1).fillna(0.0)

    raw_turnover = (weights["weight_a"].diff().abs() + weights["weight_b"].diff().abs()).fillna(0.0)
    costs = calculate_indian_transaction_costs(raw_turnover, config.cost_model)
    slippage = calculate_slippage(raw_turnover, config.base_slippage_bps)

    net_returns, gross_returns, turnover, equity_curve = compute_portfolio_pnl(
        weights, ret_a, ret_b, costs, slippage
    )

    metrics = compute_performance_metrics(
        net_returns, positions, annualization_factor=config.annualization_factor
    )

    return BacktestEngineResult(
        returns=net_returns,
        gross_returns=gross_returns,
        turnover=turnover,
        costs=costs,
        slippage=slippage,
        positions=positions,
        weights=weights,
        equity_curve=equity_curve,
        sharpe=metrics["sharpe"],
        annualized_return=metrics["annualized_return"],
        annualized_vol=metrics["annualized_vol"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        n_trades=metrics["n_trades"],
    )
