from __future__ import annotations
import pandas as pd


def align_trading_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures datetime index is strictly monotonic and removes duplicate
    timestamps or weekend artifacts.
    """
    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    # Filter only weekdays
    df = df[df.index.dayofweek < 5]
    return df


def clean_missing_data(
    prices: pd.DataFrame,
    min_history_pct: float = 0.90,
    max_consecutive_nans: int = 5,
) -> pd.DataFrame:
    """
    Removes columns with insufficient data and applies forward-filling
    with backward-fill for the remaining valid symbols.

    Parameters
    ----------
    prices : Raw price DataFrame
    min_history_pct : Minimum percentage of non-null observations required (default 90%)
    max_consecutive_nans : Maximum tolerated continuous NaN gap
    """
    df = align_trading_calendar(prices)
    if df.empty:
        return df

    # Drop tickers with too many NaNs overall
    threshold = int(min_history_pct * len(df))
    df = df.dropna(axis=1, thresh=threshold)

    # Check consecutive NaNs
    valid_cols = []
    for col in df.columns:
        series = df[col]
        max_gap = series.isna().astype(int).groupby(series.notna().astype(int).cumsum()).sum().max()
        if max_gap <= max_consecutive_nans:
            valid_cols.append(col)

    df = df[valid_cols]
    # Forward-fill prices (last traded price persists) then backfill early edge
    df = df.ffill().bfill()
    return df
