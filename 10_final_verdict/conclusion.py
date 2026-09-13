from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ExecutiveTearSheet:
    strategy_name: str
    production_verdict: str
    production_score: int
    key_strengths: list[str]
    risk_warnings: list[str]
    verdict_summary: str


def generate_final_verdict_report(
    sharpe: float,
    max_drawdown_pct: float,
    win_rate_pct: float,
    profit_factor: float,
    t_stat: float,
    market_beta: float,
    oos_sharpe_retention: float,
) -> ExecutiveTearSheet:
    score = 0
    strengths: list[str] = []
    risks: list[str] = []

    if sharpe >= 1.5:
        score += 25
        strengths.append(f"Exceptional Risk-Adjusted Return: Sharpe {sharpe:.2f}")
    elif sharpe >= 0.75:
        score += 15
        strengths.append(f"Acceptable Risk-Adjusted Return: Sharpe {sharpe:.2f}")
    elif sharpe > 0:
        score += 5
        risks.append(f"Sub-optimal Sharpe ({sharpe:.2f}); vulnerable to transaction cost drag.")
    else:
        risks.append(f"Negative Sharpe ({sharpe:.2f}); strategy lost capital net of costs.")

    if abs(max_drawdown_pct) <= 8.0:
        score += 20
        strengths.append(f"Tight Drawdown Control: Max DD {max_drawdown_pct:.2f}%")
    elif abs(max_drawdown_pct) <= 15.0:
        score += 10
    else:
        risks.append(f"High Tail Risk: Max DD reached {max_drawdown_pct:.2f}%")

    if abs(market_beta) <= 0.10:
        score += 20
        strengths.append(f"Strict Market Neutrality: Market Beta = {market_beta:.3f}")
    elif abs(market_beta) <= 0.25:
        score += 10
    else:
        risks.append(f"High Market Correlation: Beta = {market_beta:.3f}")

    if t_stat >= 2.0:
        score += 20
        strengths.append(f"Statistically Significant Alpha: t-stat = {t_stat:.2f} (p < 0.05)")
    elif t_stat > 1.0:
        score += 10
    else:
        risks.append(f"Alpha not statistically distinguishable from random chance (t-stat {t_stat:.2f})")

    if oos_sharpe_retention >= 0.60:
        score += 15
        strengths.append(f"Strong Out-of-Sample Retention: {oos_sharpe_retention*100:.1f}%")
    else:
        risks.append(f"In-sample Overfitting / Degradation in OOS: retention {oos_sharpe_retention*100:.1f}%")

    if score >= 75:
        verdict = "DEPLOY READY (GO)"
    elif score >= 50:
        verdict = "CONDITIONAL / PAPER TRADING (CAUTION)"
    else:
        verdict = "REJECT / RE-OPTIMIZE (NO-GO)"

    summary = (
        f"Indian StatArb Strategy achieved an institutional readiness score of {score}/100. "
        f"Verdict: {verdict}."
    )

    return ExecutiveTearSheet(
        strategy_name="Indian Equities Cointegrated Pairs Trading Engine",
        production_verdict=verdict,
        production_score=score,
        key_strengths=strengths,
        risk_warnings=risks,
        verdict_summary=summary,
    )
