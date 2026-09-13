from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HypothesisTestResult:
    t_statistic: float
    p_value_one_tailed: float
    is_statistically_significant: bool
    confidence_level_pct: float
    null_rejected: bool
    t_stat_threshold_95: float = 1.645
    t_stat_threshold_99: float = 2.326


def run_alpha_hypothesis_test(
    returns: pd.Series,
    significance_level: float = 0.05,
) -> HypothesisTestResult:
    clean_rets = returns.dropna().values
    n = len(clean_rets)

    if n < 5:
        return HypothesisTestResult(0.0, 1.0, False, 0.0, False)

    mean_val = np.mean(clean_rets)
    std_val = np.std(clean_rets, ddof=1)
    se = std_val / np.sqrt(n)

    t_stat = float(mean_val / se) if se > 0 else 0.0
    p_val = float(1.0 - stats.t.cdf(t_stat, df=n - 1)) if t_stat > 0 else 1.0

    null_rejected = bool(p_val < significance_level and t_stat > 0)
    confidence = float((1.0 - p_val) * 100.0)

    return HypothesisTestResult(
        t_statistic=t_stat,
        p_value_one_tailed=p_val,
        is_statistically_significant=null_rejected,
        confidence_level_pct=confidence,
        null_rejected=null_rejected,
    )
