from __future__ import annotations
import pandas as pd


def compute_adtv(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    window: int = 60,
) -> pd.Series:
    if volumes is not None and not volumes.empty:
        common_cols = [c for c in prices.columns if c in volumes.columns]
        daily_turnover_cr = (prices[common_cols] * volumes[common_cols]) / 1e7
        return daily_turnover_cr.rolling(window).mean().iloc[-1]
    return pd.Series(50.0, index=prices.columns)


def filter_by_liquidity(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    min_adtv_cr: float = 10.0,
    window: int = 60,
) -> pd.DataFrame:
    adtv = compute_adtv(prices, volumes, window=window)
    liquid_tickers = adtv[adtv >= min_adtv_cr].index.tolist()
    return prices[[c for c in liquid_tickers if c in prices.columns]]
