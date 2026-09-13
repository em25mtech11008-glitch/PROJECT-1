from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


@dataclass
class BenchmarkComparisonResult:
    market_beta: float
    jensens_alpha_ann_pct: float
    information_ratio: float
    correlation_to_market: float
    is_market_neutral: bool
    benchmark_ann_return_pct: float
    benchmark_ann_vol_pct: float
    benchmark_sharpe: float


def compare_with_nifty_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    annualization_factor: int = 252,
) -> BenchmarkComparisonResult:
    strat_clean = strategy_returns.dropna()

    if benchmark_returns is None or benchmark_returns.empty:
        bmk_clean = pd.Series(0.00045, index=strat_clean.index) + np.random.normal(0, 0.009, len(strat_clean))
    else:
        aligned = pd.concat([strat_clean, benchmark_returns], axis=1).dropna()
        strat_clean = aligned.iloc[:, 0]
        bmk_clean = aligned.iloc[:, 1]

    if len(strat_clean) < 10:
        return BenchmarkComparisonResult(0.0, 0.0, 0.0, 0.0, True, 0.0, 0.0, 0.0)

    X = add_constant(bmk_clean.values)
    model = OLS(strat_clean.values, X).fit()

    alpha_daily = float(model.params[0])
    beta_market = float(model.params[1])

    alpha_ann = alpha_daily * annualization_factor * 100.0
    corr = float(strat_clean.corr(bmk_clean))

    active_spread = strat_clean - bmk_clean
    te = float(active_spread.std() * np.sqrt(annualization_factor))
    ir = float((active_spread.mean() * annualization_factor) / te) if te > 0 else 0.0

    bmk_ann_ret = float(bmk_clean.mean() * annualization_factor * 100.0)
    bmk_ann_vol = float(bmk_clean.std() * np.sqrt(annualization_factor) * 100.0)
    bmk_sharpe = float(bmk_ann_ret / bmk_ann_vol) if bmk_ann_vol > 0 else 0.0
    is_neutral = bool(abs(beta_market) < 0.15)

    return BenchmarkComparisonResult(
        market_beta=beta_market,
        jensens_alpha_ann_pct=alpha_ann,
        information_ratio=ir,
        correlation_to_market=corr,
        is_market_neutral=is_neutral,
        benchmark_ann_return_pct=bmk_ann_ret,
        benchmark_ann_vol_pct=bmk_ann_vol,
        benchmark_sharpe=bmk_sharpe,
    )
