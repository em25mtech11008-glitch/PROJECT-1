from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class DrawdownStats:
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    clean_rets = returns.dropna()
    equity = (1.0 + clean_rets).cumprod()
    peak = equity.cummax()
    return (equity - peak) / peak


def analyze_drawdown_periods(returns: pd.Series) -> DrawdownStats:
    dd = compute_drawdown_series(returns)
    if dd.empty:
        return DrawdownStats(0.0, 0.0, 0, 0.0)

    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    current_dd = float(dd.iloc[-1])

    is_underwater = dd < 0
    duration_counts = is_underwater.astype(int).groupby((~is_underwater).cumsum()).cumsum()
    max_duration = int(duration_counts.max()) if not duration_counts.empty else 0

    return DrawdownStats(
        max_drawdown_pct=max_dd * 100.0,
        avg_drawdown_pct=avg_dd * 100.0,
        max_drawdown_duration_days=max_duration,
        current_drawdown_pct=current_dd * 100.0,
    )
