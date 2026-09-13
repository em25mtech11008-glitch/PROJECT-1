from __future__ import annotations
import numpy as np
import pandas as pd


def compute_cagr(returns: pd.Series, annualization_factor: int = 252) -> float:
    clean_rets = returns.dropna()
    n_days = len(clean_rets)
    if n_days == 0:
        return 0.0

    cumulative_wealth = (1.0 + clean_rets).prod()
    years = n_days / annualization_factor
    if years <= 0 or cumulative_wealth <= 0:
        return 0.0

    return float((cumulative_wealth ** (1.0 / years)) - 1.0)


def compute_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    clean_rets = returns.dropna()
    if clean_rets.empty:
        return pd.DataFrame()

    monthly_rets = clean_rets.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)
    df = pd.DataFrame({
        "Year": monthly_rets.index.year,
        "Month": monthly_rets.index.strftime("%b"),
        "Return": monthly_rets.values * 100.0,
    })

    pivot = df.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ordered_cols = [m for m in month_order if m in pivot.columns]
    return pivot[ordered_cols].round(2)
