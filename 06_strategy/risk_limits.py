from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class RiskLimitsConfig:
    max_gross_leverage: float = 1.0
    max_pair_allocation: float = 0.20
    max_drawdown_stop: float = 0.15
    min_margin_cushion: float = 0.25


def apply_risk_limits(
    weights: pd.DataFrame,
    config: RiskLimitsConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = RiskLimitsConfig()

    scaled_weights = weights.copy()
    gross = scaled_weights["weight_a"].abs() + scaled_weights["weight_b"].abs()

    over_leverage = gross > config.max_gross_leverage
    if over_leverage.any():
        scale = config.max_gross_leverage / gross[over_leverage]
        scaled_weights.loc[over_leverage, "weight_a"] *= scale
        scaled_weights.loc[over_leverage, "weight_b"] *= scale

    return scaled_weights
