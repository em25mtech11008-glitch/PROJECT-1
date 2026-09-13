from __future__ import annotations
import numpy as np
import pandas as pd


def compute_log_prices(clean_prices: pd.DataFrame) -> pd.DataFrame:
    assert (clean_prices > 0).all().all(), "Prices must be strictly positive."
    return np.log(clean_prices)


def compute_returns_matrix(
    clean_prices: pd.DataFrame,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    if use_log_returns:
        log_p = compute_log_prices(clean_prices)
        return log_p.diff().dropna()
    return clean_prices.pct_change().dropna()
