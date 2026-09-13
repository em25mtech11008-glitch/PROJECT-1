from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


@dataclass
class OLSFitResult:
    beta: float
    alpha: float
    r_squared: float
    t_stat_beta: float
    residual_std: float


def fit_static_ols(y: pd.Series, x: pd.Series) -> OLSFitResult:
    df = pd.concat([y, x], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    X = add_constant(x_clean.values)
    model = OLS(y_clean.values, X).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    r_squared = float(model.rsquared)
    t_stat_beta = float(model.tvalues[1])
    residual_std = float(model.resid.std())

    return OLSFitResult(
        beta=beta,
        alpha=alpha,
        r_squared=r_squared,
        t_stat_beta=t_stat_beta,
        residual_std=residual_std,
    )
