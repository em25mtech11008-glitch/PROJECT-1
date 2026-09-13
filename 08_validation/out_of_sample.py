from __future__ import annotations
from dataclasses import dataclass
import importlib
import pandas as pd

_exec_module = importlib.import_module("07_backtest.execution")
run_pair_backtest = _exec_module.run_pair_backtest
BacktestEngineConfig = _exec_module.BacktestEngineConfig


@dataclass
class OOSComparisonResult:
    is_sharpe: float
    oos_sharpe: float
    is_ann_return: float
    oos_ann_return: float
    is_max_dd: float
    oos_max_dd: float
    sharpe_retention_ratio: float


def evaluate_out_of_sample(
    log_a: pd.Series,
    log_b: pd.Series,
    split_ratio: float = 0.70,
    config: BacktestEngineConfig | None = None,
) -> OOSComparisonResult:
    if config is None:
        config = BacktestEngineConfig()

    df = pd.concat([log_a, log_b], axis=1).dropna()
    split_idx = int(len(df) * split_ratio)

    is_a, is_b = df.iloc[:split_idx, 0], df.iloc[:split_idx, 1]
    oos_a, oos_b = df.iloc[split_idx:, 0], df.iloc[split_idx:, 1]

    is_res = run_pair_backtest(is_a, is_b, config=config)
    oos_res = run_pair_backtest(oos_a, oos_b, config=config)

    retention = (oos_res.sharpe / is_res.sharpe) if is_res.sharpe > 0 else 0.0

    return OOSComparisonResult(
        is_sharpe=is_res.sharpe,
        oos_sharpe=oos_res.sharpe,
        is_ann_return=is_res.annualized_return,
        oos_ann_return=oos_res.annualized_return,
        is_max_dd=is_res.max_drawdown,
        oos_max_dd=oos_res.max_drawdown,
        sharpe_retention_ratio=retention,
    )
