from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TradeSummaryStats:
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    avg_holding_days: float
    annual_turnover: float


def compute_trade_statistics(
    returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    annualization_factor: int = 252,
) -> TradeSummaryStats:
    clean_rets = returns.dropna()
    clean_pos = positions.reindex(clean_rets.index).fillna(0)

    trade_returns: list[float] = []
    holding_periods: list[int] = []

    in_trade = False
    entry_idx = 0
    trade_ret = 0.0

    pos_vals = clean_pos.values
    ret_vals = clean_rets.values
    n = len(clean_pos)

    for t in range(n):
        curr_p = pos_vals[t]
        if not in_trade and curr_p != 0:
            in_trade = True
            entry_idx = t
            trade_ret = ret_vals[t]
        elif in_trade:
            trade_ret += ret_vals[t]
            if curr_p == 0 or t == n - 1:
                in_trade = False
                trade_returns.append(trade_ret)
                holding_periods.append(t - entry_idx + 1)

    total_trades = len(trade_returns)
    if total_trades == 0:
        return TradeSummaryStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr_rets = np.array(trade_returns)
    wins = arr_rets[arr_rets > 0]
    losses = arr_rets[arr_rets < 0]

    win_rate = (len(wins) / total_trades) * 100.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else (10.0 if len(wins) > 0 else 0.0)

    avg_trade = float(arr_rets.mean()) * 100.0
    avg_win = float(wins.mean() * 100.0) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean() * 100.0) if len(losses) > 0 else 0.0
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0
    avg_holding = float(np.mean(holding_periods)) if holding_periods else 0.0
    ann_turnover = float(turnover.mean() * annualization_factor)

    return TradeSummaryStats(
        total_trades=total_trades,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_trade_return_pct=avg_trade,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        win_loss_ratio=win_loss_ratio,
        avg_holding_days=avg_holding,
        annual_turnover=ann_turnover,
    )
