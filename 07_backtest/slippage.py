from __future__ import annotations
import numpy as np
import pandas as pd


def compute_market_impact(
    order_shares: pd.Series,
    adtv_shares: pd.Series,
    daily_vol: pd.Series,
    impact_constant: float = 0.1,
) -> pd.Series:
    participation_rate = (order_shares / adtv_shares).fillna(0.0).clip(lower=0.0, upper=0.5)
    return impact_constant * daily_vol * np.sqrt(participation_rate)


def calculate_slippage(
    turnover: pd.Series,
    base_slippage_bps: float = 2.0,
) -> pd.Series:
    return turnover * (base_slippage_bps / 10_000.0)
