from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ExitRules:
    exit_z: float = 0.5
    stop_loss_z: float = 3.5
    max_hold_days: int | None = 20


def generate_exit_signals(
    z_scores: pd.Series,
    entry_z: float = 2.0,
    exit_rules: ExitRules | None = None,
) -> tuple[pd.Series, pd.Series]:
    if exit_rules is None:
        exit_rules = ExitRules()

    n = len(z_scores)
    positions = np.zeros(n, dtype=int)
    exit_reasons = pd.Series("", index=z_scores.index, dtype=object)

    current_pos = 0
    hold_days = 0
    z_vals = z_scores.values

    for t in range(n):
        z = z_vals[t]
        if np.isnan(z):
            positions[t] = current_pos
            continue

        if current_pos == 0:
            if z <= -entry_z:
                current_pos = 1
                hold_days = 0
            elif z >= entry_z:
                current_pos = -1
                hold_days = 0
        else:
            hold_days += 1
            if abs(z) <= exit_rules.exit_z:
                current_pos = 0
                exit_reasons.iloc[t] = "take_profit"
            elif exit_rules.stop_loss_z and abs(z) >= exit_rules.stop_loss_z:
                current_pos = 0
                exit_reasons.iloc[t] = "stop_loss_z"
            elif exit_rules.max_hold_days and hold_days >= exit_rules.max_hold_days:
                current_pos = 0
                exit_reasons.iloc[t] = "time_stop"

        positions[t] = current_pos

    return pd.Series(positions, index=z_scores.index, name="position"), exit_reasons
