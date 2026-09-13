from __future__ import annotations
from dataclasses import dataclass
import importlib
import numpy as np
import pandas as pd

_exec_module = importlib.import_module("07_backtest.execution")
_pnl_module = importlib.import_module("07_backtest.PnL")

run_pair_backtest = _exec_module.run_pair_backtest
BacktestEngineConfig = _exec_module.BacktestEngineConfig
compute_performance_metrics = _pnl_module.compute_performance_metrics


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    oos_equity_curve: pd.Series
    sharpe: float
    annualized_return: float
    annualized_vol: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    n_windows: int
    window_sharpes: list[float]


def run_walk_forward_validation(
    log_a: pd.Series,
    log_b: pd.Series,
    train_days: int = 250,
    test_days: int = 60,
    config: BacktestEngineConfig | None = None,
) -> WalkForwardResult:
    if config is None:
        config = BacktestEngineConfig()

    df = pd.concat([log_a, log_b], axis=1).dropna()
    n_total = len(df)
    window_len = train_days + test_days

    oos_return_chunks: list[pd.Series] = []
    oos_position_chunks: list[pd.Series] = []
    window_sharpes: list[float] = []

    start = 0
    while start + window_len <= n_total:
        sub_a = df.iloc[start : start + window_len, 0]
        sub_b = df.iloc[start : start + window_len, 1]

        res = run_pair_backtest(sub_a, sub_b, config=config)

        oos_rets = res.returns.iloc[train_days:]
        oos_pos = res.positions.iloc[train_days:]

        oos_return_chunks.append(oos_rets)
        oos_position_chunks.append(oos_pos)

        w_sharpe = float((oos_rets.mean() / oos_rets.std()) * np.sqrt(config.annualization_factor)) if oos_rets.std() > 0 else 0.0
        window_sharpes.append(w_sharpe)

        start += test_days

    if not oos_return_chunks:
        res = run_pair_backtest(df.iloc[:, 0], df.iloc[:, 1], config=config)
        return WalkForwardResult(
            oos_returns=res.returns,
            oos_equity_curve=res.equity_curve,
            sharpe=res.sharpe,
            annualized_return=res.annualized_return,
            annualized_vol=res.annualized_vol,
            max_drawdown=res.max_drawdown,
            win_rate=res.win_rate,
            n_trades=res.n_trades,
            n_windows=1,
            window_sharpes=[res.sharpe],
        )

    all_oos_returns = pd.concat(oos_return_chunks)
    all_oos_positions = pd.concat(oos_position_chunks)
    all_oos_returns = all_oos_returns[~all_oos_returns.index.duplicated(keep="first")].sort_index()
    all_oos_positions = all_oos_positions[~all_oos_positions.index.duplicated(keep="first")].sort_index()

    oos_equity = (1.0 + all_oos_returns).cumprod()
    metrics = compute_performance_metrics(
        all_oos_returns, all_oos_positions, annualization_factor=config.annualization_factor
    )

    return WalkForwardResult(
        oos_returns=all_oos_returns,
        oos_equity_curve=oos_equity,
        sharpe=metrics["sharpe"],
        annualized_return=metrics["annualized_return"],
        annualized_vol=metrics["annualized_vol"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        n_trades=metrics["n_trades"],
        n_windows=len(window_sharpes),
        window_sharpes=window_sharpes,
    )
