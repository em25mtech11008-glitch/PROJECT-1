from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class EntryRules:
    entry_z: float = 2.0
    extreme_z_filter: float = 3.5


def generate_entry_signals(
    z_scores: pd.Series,
    rules: EntryRules | None = None,
) -> pd.Series:
    if rules is None:
        rules = EntryRules()

    entries = pd.Series(0, index=z_scores.index, dtype=int)
    long_mask = (z_scores <= -rules.entry_z) & (z_scores >= -rules.extreme_z_filter)
    entries[long_mask] = 1

    short_mask = (z_scores >= rules.entry_z) & (z_scores <= rules.extreme_z_filter)
    entries[short_mask] = -1

    return entries
