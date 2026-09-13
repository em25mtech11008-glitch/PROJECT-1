from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .returns import compute_cagr


@dataclass
class RiskRatios:
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    annualized_return: float
    annualized_vol: float
    downside_vol: float


def compute_risk_adjusted_ratios(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    annualization_factor: int = 252,
) -> RiskRatios:
    clean_rets = returns.dropna()
    if len(clean_rets) == 0:
        return RiskRatios(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    daily_rf = risk_free_rate / annualization_factor
    excess_rets = clean_rets - daily_rf

    mean_ret = clean_rets.mean()
    ann_ret = float(mean_ret * annualization_factor)
    ann_vol = float(clean_rets.std() * np.sqrt(annualization_factor))

    sharpe = float(excess_rets.mean() / clean_rets.std() * np.sqrt(annualization_factor)) if clean_rets.std() > 0 else 0.0

    downside_rets = clean_rets[clean_rets < 0]
    downside_std = float(downside_rets.std() * np.sqrt(annualization_factor)) if len(downside_rets) > 0 and downside_rets.std() > 0 else 1e-6
    sortino = float(ann_ret / downside_std)

    equity = (1.0 + clean_rets).cumprod()
    peak = equity.cummax()
    max_dd = float(abs((equity - peak).min() / peak.max())) if peak.max() > 0 else 0.0
    cagr = compute_cagr(clean_rets, annualization_factor=annualization_factor)
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    pos_gains = clean_rets[clean_rets > daily_rf].sum()
    neg_losses = abs(clean_rets[clean_rets < daily_rf].sum())
    omega = float(pos_gains / neg_losses) if neg_losses > 0 else 1.0

    return RiskRatios(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        downside_vol=downside_std,
    )
