from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestEngineResult:
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    slippage: pd.Series
    positions: pd.Series
    weights: pd.DataFrame
    equity_curve: pd.Series
    sharpe: float
    annualized_return: float
    annualized_vol: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def compute_portfolio_pnl(
    weights: pd.DataFrame,
    ret_a: pd.Series,
    ret_b: pd.Series,
    cost_series: pd.Series,
    slippage_series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    w_a_lag = weights["weight_a"].shift(1).fillna(0.0)
    w_b_lag = weights["weight_b"].shift(1).fillna(0.0)

    gross_returns = (w_a_lag * ret_a + w_b_lag * ret_b).fillna(0.0)
    turnover = (weights["weight_a"].diff().abs() + weights["weight_b"].diff().abs()).fillna(0.0)
    net_returns = gross_returns - cost_series - slippage_series
    equity_curve = (1.0 + net_returns).cumprod()

    return net_returns, gross_returns, turnover, equity_curve


def compute_performance_metrics(
    net_returns: pd.Series,
    positions: pd.Series,
    annualization_factor: int = 252,
) -> dict[str, float]:
    clean_rets = net_returns.dropna()
    if len(clean_rets) == 0 or clean_rets.std() == 0:
        return {
            "sharpe": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
        }

    mean_ret = clean_rets.mean()
    vol = clean_rets.std()

    ann_ret = float(mean_ret * annualization_factor)
    ann_vol = float(vol * np.sqrt(annualization_factor))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    equity = (1.0 + clean_rets).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    pos_diff = positions.diff().abs().fillna(0.0)
    n_trades = int((pos_diff > 0).sum() // 2)

    active_rets = clean_rets[positions.shift(1).fillna(0) != 0]
    win_rate = float((active_rets > 0).mean()) if len(active_rets) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
    }
