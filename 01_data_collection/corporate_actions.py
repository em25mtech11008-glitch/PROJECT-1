from __future__ import annotations
import pandas as pd
import yfinance as yf


def fetch_corporate_actions(ticker: str) -> dict[str, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        actions = t.actions
        return {
            "dividends": actions["Dividends"] if "Dividends" in actions.columns else pd.Series(dtype=float),
            "splits": actions["Stock Splits"] if "Stock Splits" in actions.columns else pd.Series(dtype=float),
        }
    except Exception:
        return {"dividends": pd.Series(dtype=float), "splits": pd.Series(dtype=float)}


def apply_corporate_actions(raw_prices: pd.DataFrame, splits_df: pd.DataFrame | None = None) -> pd.DataFrame:
    adjusted = raw_prices.copy()
    if splits_df is not None and not splits_df.empty:
        for ticker in adjusted.columns:
            if ticker in splits_df.columns:
                split_events = splits_df[ticker][splits_df[ticker] > 0]
                for date, factor in split_events.items():
                    adjusted.loc[:date, ticker] /= factor
    return adjusted
