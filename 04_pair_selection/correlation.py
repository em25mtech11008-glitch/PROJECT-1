from __future__ import annotations
import pandas as pd


def compute_correlation_matrix(
    log_prices: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Computes pairwise correlation matrix of log-prices or returns.
    """
    return log_prices.corr(method=method)


def filter_correlated_pairs(
    log_prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    min_correlation: float = 0.70,
) -> list[tuple[str, str]]:
    """
    Keeps only pairs whose log-price correlation meets or exceeds min_correlation.
    """
    corr = compute_correlation_matrix(log_prices)
    filtered = []
    for a, b in candidate_pairs:
        if a in corr.columns and b in corr.columns:
            if corr.loc[a, b] >= min_correlation:
                filtered.append((a, b))
    return filtered
