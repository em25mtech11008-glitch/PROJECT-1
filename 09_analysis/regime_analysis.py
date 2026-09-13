from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class RegimePerformance:
    bull_sharpe: float
    bear_sharpe: float
    high_vol_sharpe: float
    normal_vol_sharpe: float
    bull_ann_return: float
    bear_ann_return: float


def analyze_market_regimes(
    strategy_returns: pd.Series,
    benchmark_prices: pd.Series | None = None,
    annualization_factor: int = 252,
) -> RegimePerformance:
    clean_rets = strategy_returns.dropna()
    if clean_rets.empty:
        return RegimePerformance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rolling_vol = clean_rets.rolling(20).std()
    vol_threshold = rolling_vol.quantile(0.80)

    high_vol_mask = rolling_vol >= vol_threshold
    norm_vol_mask = rolling_vol < vol_threshold

    high_vol_rets = clean_rets[high_vol_mask].dropna()
    norm_vol_rets = clean_rets[norm_vol_mask].dropna()

    high_vol_sharpe = float((high_vol_rets.mean() / high_vol_rets.std()) * (annualization_factor ** 0.5)) if high_vol_rets.std() > 0 else 0.0
    norm_vol_sharpe = float((norm_vol_rets.mean() / norm_vol_rets.std()) * (annualization_factor ** 0.5)) if norm_vol_rets.std() > 0 else 0.0

    if benchmark_prices is not None and not benchmark_prices.empty:
        bmk_aligned = benchmark_prices.reindex(clean_rets.index).ffill().bfill()
        dma200 = bmk_aligned.rolling(200, min_periods=20).mean()
        bull_mask = bmk_aligned >= dma200
        bear_mask = bmk_aligned < dma200
    else:
        equity = (1.0 + clean_rets).cumprod()
        dma_eq = equity.rolling(100, min_periods=20).mean()
        bull_mask = equity >= dma_eq
        bear_mask = equity < dma_eq

    bull_rets = clean_rets[bull_mask].dropna()
    bear_rets = clean_rets[bear_mask].dropna()

    bull_sharpe = float((bull_rets.mean() / bull_rets.std()) * (annualization_factor ** 0.5)) if bull_rets.std() > 0 else 0.0
    bear_sharpe = float((bear_rets.mean() / bear_rets.std()) * (annualization_factor ** 0.5)) if bear_rets.std() > 0 else 0.0

    bull_ann = float(bull_rets.mean() * annualization_factor) if not bull_rets.empty else 0.0
    bear_ann = float(bear_rets.mean() * annualization_factor) if not bear_rets.empty else 0.0

    return RegimePerformance(
        bull_sharpe=bull_sharpe,
        bear_sharpe=bear_sharpe,
        high_vol_sharpe=high_vol_sharpe,
        normal_vol_sharpe=norm_vol_sharpe,
        bull_ann_return=bull_ann,
        bear_ann_return=bear_ann,
    )
