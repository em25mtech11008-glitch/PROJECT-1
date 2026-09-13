from __future__ import annotations
from itertools import combinations
import pandas as pd


def group_by_sector(
    tickers: list[str],
    sector_map: dict[str, str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for ticker in tickers:
        sector = sector_map.get(ticker, "Unclassified")
        grouped.setdefault(sector, []).append(ticker)
    return grouped


def filter_pairs_by_sector(
    tickers: list[str],
    sector_map: dict[str, str],
    allow_unclassified: bool = False,
) -> list[tuple[str, str]]:
    grouped = group_by_sector(tickers, sector_map)
    candidate_pairs: list[tuple[str, str]] = []

    for sector, sector_tickers in grouped.items():
        if sector == "Unclassified" and not allow_unclassified:
            continue
        if len(sector_tickers) >= 2:
            candidate_pairs.extend(combinations(sorted(sector_tickers), 2))

    return candidate_pairs
