from __future__ import annotations
from dataclasses import dataclass
import itertools
import importlib
import numpy as np
import pandas as pd

_exec_module = importlib.import_module("07_backtest.execution")
run_pair_backtest = _exec_module.run_pair_backtest
BacktestEngineConfig = _exec_module.BacktestEngineConfig


@dataclass
class MonteCarloRobustnessResult:
    mean_sharpe: float
    std_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    prob_positive_sharpe: float


def run_sensitivity_grid(
    log_a: pd.Series,
    log_b: pd.Series,
    entry_z_list: list[float] | None = None,
    exit_z_list: list[float] | None = None,
    z_windows: list[int] | None = None,
) -> pd.DataFrame:
    if entry_z_list is None:
        entry_z_list = [1.5, 2.0, 2.5]
    if exit_z_list is None:
        exit_z_list = [0.2, 0.5, 0.8]
    if z_windows is None:
        z_windows = [15, 20, 30]

    rows: list[dict] = []
    for ez, xz, zw in itertools.product(entry_z_list, exit_z_list, z_windows):
        cfg = BacktestEngineConfig(
            entry_z=ez,
            exit_z=xz,
            zscore_window=zw,
        )
        res = run_pair_backtest(log_a, log_b, config=cfg)
        rows.append({
            "entry_z": ez,
            "exit_z": xz,
            "zscore_window": zw,
            "sharpe": res.sharpe,
            "ann_return": res.annualized_return,
            "max_dd": res.max_drawdown,
            "n_trades": res.n_trades,
        })

    return pd.DataFrame(rows)


def run_monte_carlo_permutation(
    returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
    annualization_factor: int = 252,
) -> MonteCarloRobustnessResult:
    rng = np.random.default_rng(seed)
    clean_rets = returns.dropna().values
    n = len(clean_rets)

    if n < 10:
        return MonteCarloRobustnessResult(0.0, 0.0, 0.0, 0.0, 0.0)

    sim_sharpes = np.zeros(n_simulations)
    for i in range(n_simulations):
        sampled = rng.choice(clean_rets, size=n, replace=True)
        std_val = sampled.std()
        if std_val > 0:
            sim_sharpes[i] = (sampled.mean() / std_val) * np.sqrt(annualization_factor)
        else:
            sim_sharpes[i] = 0.0

    return MonteCarloRobustnessResult(
        mean_sharpe=float(np.mean(sim_sharpes)),
        std_sharpe=float(np.std(sim_sharpes)),
        p5_sharpe=float(np.percentile(sim_sharpes, 5)),
        p95_sharpe=float(np.percentile(sim_sharpes, 95)),
        prob_positive_sharpe=float(np.mean(sim_sharpes > 0)),
    )
