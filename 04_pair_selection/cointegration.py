from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from .ADF import perform_adf_test
from .half_life import calculate_ou_half_life


@dataclass
class PairCandidate:
    asset_a: str
    asset_b: str
    ols_beta: float
    ols_alpha: float
    adf_stat: float
    adf_pvalue: float
    engle_granger_pvalue: float
    half_life: float
    correlation: float
    sector: str = "General"


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    significance: float = 0.05,
) -> tuple[bool, float, float, float, float]:
    """
    Runs the 2-step Engle-Granger procedure:
    Step 1: OLS regression y = beta * x + alpha + eps
    Step 2: ADF test on residual eps
    """
    df = pd.concat([y, x], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    X = add_constant(x_clean.values)
    ols_model = OLS(y_clean.values, X).fit()
    alpha, beta = ols_model.params[0], ols_model.params[1]

    spread = y_clean - (beta * x_clean + alpha)
    adf_res = perform_adf_test(spread, significance=significance)

    # Secondary statsmodels coint check
    _, eg_pvalue, _ = coint(y_clean, x_clean)

    is_coint = adf_res.is_stationary
    return is_coint, float(beta), float(alpha), adf_res.pvalue, float(eg_pvalue)


def screen_cointegrated_pairs(
    log_prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]] | None = None,
    min_correlation: float = 0.65,
    significance: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
    sector_map: dict[str, str] | None = None,
) -> list[PairCandidate]:
    """
    Screens pairs for statistical cointegration.
    """
    tickers = log_prices.columns.tolist()
    corr = log_prices.corr()

    if candidate_pairs is None:
        candidate_pairs = list(combinations(tickers, 2))

    valid_candidates: list[PairCandidate] = []

    for a, b in candidate_pairs:
        if a not in log_prices.columns or b not in log_prices.columns:
            continue

        pair_corr = corr.loc[a, b] if (a in corr.columns and b in corr.columns) else 0.0
        if pair_corr < min_correlation:
            continue

        y = log_prices[a]
        x = log_prices[b]

        is_coint, beta, alpha, adf_pval, eg_pval = engle_granger_test(y, x, significance=significance)
        if not is_coint:
            continue

        spread = y - (beta * x + alpha)
        hl = calculate_ou_half_life(spread)

        if hl < min_half_life or hl > max_half_life:
            continue

        sector = sector_map.get(a, "General") if sector_map else "General"

        valid_candidates.append(PairCandidate(
            asset_a=a,
            asset_b=b,
            ols_beta=beta,
            ols_alpha=alpha,
            adf_stat=float(perform_adf_test(spread).adf_stat),
            adf_pvalue=adf_pval,
            engle_granger_pvalue=eg_pval,
            half_life=hl,
            correlation=float(pair_corr),
            sector=sector,
        ))

    # Sort by ADF p-value ascending (strongest cointegration first)
    valid_candidates.sort(key=lambda c: c.adf_pvalue)
    return valid_candidates
