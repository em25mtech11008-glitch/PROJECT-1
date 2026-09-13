from __future__ import annotations
import pandas as pd


def compute_rolling_zscore(
    spread: pd.Series,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(5, window // 2)

    rolling_mean = spread.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = spread.rolling(window=window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0.0, float("nan"))
    z = (spread - rolling_mean) / rolling_std
    return z.fillna(0.0)


def compute_ema_zscore(
    spread: pd.Series,
    span: int = 20,
) -> pd.Series:
    ema_mean = spread.ewm(span=span, adjust=False).mean()
    ema_std = spread.ewm(span=span, adjust=False).std()
    ema_std = ema_std.replace(0.0, float("nan"))
    z = (spread - ema_mean) / ema_std
    return z.fillna(0.0)
