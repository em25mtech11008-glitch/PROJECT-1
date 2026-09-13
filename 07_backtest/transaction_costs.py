from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class IndianTaxCostBreakdown:
    stt_rate: float = 0.00025
    exchange_rate: float = 0.0000345
    sebi_rate: float = 0.000001
    stamp_duty_rate: float = 0.00003
    brokerage_rate: float = 0.0003
    gst_rate: float = 0.18

    @property
    def total_bps_one_way(self) -> float:
        fee_base = self.brokerage_rate + self.exchange_rate + self.sebi_rate
        gst_fee = fee_base * self.gst_rate
        total_rate = self.stt_rate + self.stamp_duty_rate + fee_base + gst_fee
        return total_rate * 10_000.0


INDIAN_COST_DEFAULTS = IndianTaxCostBreakdown()


def calculate_indian_transaction_costs(
    turnover: pd.Series,
    cost_model: IndianTaxCostBreakdown | None = None,
) -> pd.Series:
    if cost_model is None:
        cost_model = INDIAN_COST_DEFAULTS

    cost_fraction = cost_model.total_bps_one_way / 10_000.0
    return turnover * cost_fraction
