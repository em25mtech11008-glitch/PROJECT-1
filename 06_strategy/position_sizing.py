from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_cash_neutral_weights(
    positions: pd.Series,
    beta: pd.Series | float,
) -> pd.DataFrame:
    if isinstance(beta, (int, float)):
        beta_series = pd.Series(float(beta), index=positions.index)
    else:
        beta_series = beta

    w_a = positions.astype(float)
    w_b = -positions * beta_series

    gross = w_a.abs() + w_b.abs()
    gross = gross.replace(0.0, np.nan)

    w_a_norm = (w_a / gross).fillna(0.0)
    w_b_norm = (w_b / gross).fillna(0.0)

    return pd.DataFrame(
        {"weight_a": w_a_norm, "weight_b": w_b_norm},
        index=positions.index,
    )


def calculate_beta_neutral_weights(
    positions: pd.Series,
    beta: pd.Series | float,
) -> pd.DataFrame:
    return calculate_cash_neutral_weights(positions, beta)


def calculate_lot_sizes_inr(
    weights: pd.DataFrame,
    price_a: pd.Series,
    price_b: pd.Series,
    portfolio_capital_inr: float = 1_000_000.0,
) -> pd.DataFrame:
    capital_a = weights["weight_a"] * portfolio_capital_inr
    capital_b = weights["weight_b"] * portfolio_capital_inr

    shares_a = (capital_a / price_a).round().astype(int)
    shares_b = (capital_b / price_b).round().astype(int)

    return pd.DataFrame(
        {"shares_a": shares_a, "shares_b": shares_b},
        index=weights.index,
    )
