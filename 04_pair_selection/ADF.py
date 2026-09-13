from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFTestResult:
    adf_stat: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict[str, float]
    is_stationary: bool


def perform_adf_test(
    series: pd.Series,
    significance: float = 0.05,
    autolag: str = "AIC",
) -> ADFTestResult:
    """
    Runs the Augmented Dickey-Fuller unit-root test on a time series.
    Null Hypothesis H0: The series has a unit root (non-stationary).
    Alternative H1: The series is stationary I(0).
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, autolag=autolag)

    adf_stat = float(res[0])
    pvalue = float(res[1])
    usedlag = int(res[2])
    nobs = int(res[3])
    critical_values = {k: float(v) for k, v in res[4].items()}
    is_stationary = bool(pvalue < significance)

    return ADFTestResult(
        adf_stat=adf_stat,
        pvalue=pvalue,
        usedlag=usedlag,
        nobs=nobs,
        critical_values=critical_values,
        is_stationary=is_stationary,
    )


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> bool:
    """
    Quick boolean helper for stationarity verification at level alpha.
    """
    return perform_adf_test(series, significance=alpha).is_stationary
