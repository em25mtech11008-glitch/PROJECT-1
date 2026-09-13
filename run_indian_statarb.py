
import argparse
import importlib
import sys
import numpy as np
import pandas as pd

# Dynamic module loaders
_data_col = importlib.import_module("01_data_collection")
_data_clean = importlib.import_module("02_data_cleaning")
_univ = importlib.import_module("03_universe")
_pair_sel = importlib.import_module("04_pair_selection")
_model = importlib.import_module("05_model")
_strat = importlib.import_module("06_strategy")
_backtest = importlib.import_module("07_backtest")
_valid = importlib.import_module("08_validation")
_analysis = importlib.import_module("09_analysis")
_verdict = importlib.import_module("10_final_verdict")


def banner(text: str, stage: str = ""):
    print("\n" + "=" * 80)
    if stage:
        print(f"[{stage}] {text.upper()}")
    else:
        print(text.upper())
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Indian Statistical Arbitrage 10-Stage Pipeline (NSE)")
    parser.add_argument("--sectors", nargs="+", default=["Private_Banking", "IT_Services", "Automobile", "Metals_Mining", "Oil_Gas_Energy", "FMCG"],
                        help="NSE sectors to include in universe")
    parser.add_argument("--start-date", default="2021-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--min-corr", type=float, default=0.60, help="Minimum correlation pre-filter threshold")
    parser.add_argument("--significance", type=float, default=0.05, help="ADF cointegration p-value threshold")
    parser.add_argument("--min-adtv", type=float, default=5.0, help="Minimum ADTV in INR Crores")
    args = parser.parse_args()

    banner("Indian Quantitative Statistical Arbitrage Engine (NSE)", "ARCHITECT")

    # -------------------------------------------------------------------------
    # STAGE 01: DATA COLLECTION
    # -------------------------------------------------------------------------
    banner("Data Collection & Ingestion (NSE Equities)", "STAGE 01")
    all_sectors = _data_col.NIFTY_SECTORS
    sector_map = _data_col.get_sector_mapping()

    # Collect tickers from specified sectors
    tickers_to_fetch = []
    for sec in args.sectors:
        if sec in all_sectors:
            tickers_to_fetch.extend(all_sectors[sec])
    tickers_to_fetch = sorted(list(set(tickers_to_fetch)))

    print(f"Target Sectors: {', '.join(args.sectors)}")
    print(f"Fetching adjusted daily prices for {len(tickers_to_fetch)} NSE equities from Yahoo Finance ({args.start_date} to {args.end_date or 'latest'})...")

    raw_prices = _data_col.download_nse_prices(tickers_to_fetch, start=args.start_date, end=args.end_date)
    if raw_prices.empty:
        print("Error: Could not retrieve price data. Please verify your internet connection.")
        sys.exit(1)

    print(f"Successfully loaded {raw_prices.shape[1]} symbols across {raw_prices.shape[0]} trading days.")

    # -------------------------------------------------------------------------
    # STAGE 02: DATA CLEANING
    # -------------------------------------------------------------------------
    banner("Data Cleaning & Calendar Alignment", "STAGE 02")
    clean_prices = _data_clean.clean_missing_data(raw_prices, min_history_pct=0.85)
    clean_prices = _data_clean.filter_circuit_breaker_spikes(clean_prices, max_daily_return_threshold=0.25)
    log_prices = _data_clean.compute_log_prices(clean_prices)
    returns_matrix = _data_clean.compute_returns_matrix(clean_prices)
    print(f"Cleaned price matrix: {clean_prices.shape[1]} tickers, {clean_prices.shape[0]} trading days.")
    print(f"Total NaNs remaining: {clean_prices.isna().sum().sum()}")

    # -------------------------------------------------------------------------
    # STAGE 03: UNIVERSE SELECTION
    # -------------------------------------------------------------------------
    banner("Universe Selection & Sector Filtering", "STAGE 03")
    liquid_prices = _univ.filter_by_liquidity(clean_prices, min_adtv_cr=args.min_adtv)
    liquid_tickers = liquid_prices.columns.tolist()
    sector_pairs = _univ.filter_pairs_by_sector(liquid_tickers, sector_map)
    print(f"Liquid tickers meeting ADTV >= INR {args.min_adtv} Cr: {len(liquid_tickers)}")
    print(f"Intra-sector candidate pairs evaluated: {len(sector_pairs)}")

    # -------------------------------------------------------------------------
    # STAGE 04: PAIR SELECTION & COINTEGRATION SCREENING
    # -------------------------------------------------------------------------
    banner("Pair Selection: Cointegration, ADF & Half-Life", "STAGE 04")
    candidates = _pair_sel.screen_cointegrated_pairs(
        log_prices,
        candidate_pairs=sector_pairs if sector_pairs else None,
        min_correlation=args.min_corr,
        significance=args.significance,
        min_half_life=1.0,
        max_half_life=60.0,
        sector_map=sector_map,
    )

    if not candidates:
        print("No intra-sector pairs passed strict cointegration at this significance level. Expanding search...")
        candidates = _pair_sel.screen_cointegrated_pairs(
            log_prices,
            candidate_pairs=None,
            min_correlation=0.50,
            significance=0.10,
            min_half_life=1.0,
            max_half_life=90.0,
            sector_map=sector_map,
        )

    if not candidates:
        print("No cointegrated pairs found in the given dataset.")
        sys.exit(0)

    print(f"Cointegrated pairs discovered: {len(candidates)}\n")
    coint_df = pd.DataFrame([{
        "Asset_A": c.asset_a,
        "Asset_B": c.asset_b,
        "Sector": c.sector,
        "ADF_pVal": f"{c.adf_pvalue:.4e}",
        "EG_pVal": f"{c.engle_granger_pvalue:.4e}",
        "Half_Life": f"{c.half_life:.2f} d",
        "OLS_Beta": f"{c.ols_beta:.3f}",
        "Corr": f"{c.correlation:.2f}",
    } for c in candidates[:10]])
    print(coint_df.to_string(index=False))

    # Pick top pair for production backtest
    top_pair = candidates[0]
    asset_a, asset_b = top_pair.asset_a, top_pair.asset_b
    print(f"\n>> Selected Top Pair for Production Backtest: {asset_a} & {asset_b} (Sector: {top_pair.sector})")

    # -------------------------------------------------------------------------
    # STAGE 05: MODEL ESTIMATION (OLS vs KALMAN FILTER)
    # -------------------------------------------------------------------------
    banner("Model Estimation: Dynamic Kalman Filter State-Space", "STAGE 05")
    y_series = log_prices[asset_a]
    x_series = log_prices[asset_b]

    ols_fit = _model.fit_static_ols(y_series, x_series)
    kf_res = _model.run_kalman_filter(y_series, x_series, delta=1e-4, observation_var=1e-3)
    spread = kf_res["spread"]
    z_scores = _model.compute_rolling_zscore(spread, window=20)

    print(f"Static OLS Beta:       {ols_fit.beta:.3f} (R^2 = {ols_fit.r_squared:.3f}, t-stat = {ols_fit.t_stat_beta:.2f})")
    print(f"Kalman Beta (Start):   {kf_res['beta'].iloc[20]:.3f}")
    print(f"Kalman Beta (End):     {kf_res['beta'].iloc[-1]:.3f}")
    print(f"Kalman Beta Range:     [{kf_res['beta'].min():.3f} to {kf_res['beta'].max():.3f}]")

    # -------------------------------------------------------------------------
    # STAGE 06 & 07: STRATEGY & BACKTEST SIMULATION WITH INDIAN TAX COSTS
    # -------------------------------------------------------------------------
    banner("Backtest Simulation & Indian Tax/Cost Accounting", "STAGE 06 & 07")
    tax_model = _backtest.IndianTaxCostBreakdown()
    print(f"Indian Cost Structure: STT={tax_model.stt_rate*100:.3f}%, Exch={tax_model.exchange_rate*100:.5f}%, "
          f"Stamp={tax_model.stamp_duty_rate*100:.3f}%, Brokerage={tax_model.brokerage_rate*100:.3f}%, GST=18%")
    print(f"Total one-way friction: ~{tax_model.total_bps_one_way:.2f} bps per trade.")

    bt_cfg = _backtest.BacktestEngineConfig(
        entry_z=2.0,
        exit_z=0.5,
        stop_loss_z=3.5,
        max_hold_days=25,
        zscore_window=20,
        kalman_delta=1e-4,
        base_slippage_bps=2.0,
        cost_model=tax_model,
    )

    bt_res = _backtest.run_pair_backtest(y_series, x_series, config=bt_cfg)

    print(f"\n--- Backtest Performance ({asset_a} / {asset_b}) ---")
    print(f"Sharpe Ratio (Net):        {bt_res.sharpe:.2f}")
    print(f"Annualized Return (Net):   {bt_res.annualized_return*100:.2f}%")
    print(f"Annualized Volatility:     {bt_res.annualized_vol*100:.2f}%")
    print(f"Max Drawdown:              {bt_res.max_drawdown*100:.2f}%")
    print(f"Win Rate:                  {bt_res.win_rate*100:.1f}%")
    print(f"Total Trades:              {bt_res.n_trades}")
    print(f"Total Frictional Drag:     {(bt_res.costs.sum() + bt_res.slippage.sum())*100:.2f}% return equivalent")

    # -------------------------------------------------------------------------
    # STAGE 08: VALIDATION (WALK-FORWARD & OOS RETENTION)
    # -------------------------------------------------------------------------
    banner("Walk-Forward Cross-Validation & Robustness", "STAGE 08")
    wf_res = _valid.run_walk_forward_validation(y_series, x_series, train_days=250, test_days=60, config=bt_cfg)
    oos_comp = _valid.evaluate_out_of_sample(y_series, x_series, split_ratio=0.70, config=bt_cfg)
    mc_res = _valid.run_monte_carlo_permutation(wf_res.oos_returns, n_simulations=500)

    print(f"Walk-Forward Windows:      {wf_res.n_windows} sequential test folds")
    print(f"Walk-Forward OOS Sharpe:   {wf_res.sharpe:.2f}")
    print(f"Walk-Forward OOS Return:   {wf_res.annualized_return*100:.2f}%")
    print(f"Walk-Forward Max DD:       {wf_res.max_drawdown*100:.2f}%")
    print(f"In-Sample vs OOS Sharpe:   IS {oos_comp.is_sharpe:.2f} -> OOS {oos_comp.oos_sharpe:.2f} (Retention: {oos_comp.sharpe_retention_ratio*100:.1f}%)")
    print(f"Monte Carlo 95% CI Sharpe: [{mc_res.p5_sharpe:.2f} to {mc_res.p95_sharpe:.2f}] (Prob > 0: {mc_res.prob_positive_sharpe*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STAGE 09: COMPREHENSIVE PERFORMANCE & REGIME ANALYSIS
    # -------------------------------------------------------------------------
    banner("Risk Ratios, Trade Logs & Market Regimes", "STAGE 09")
    ratios = _analysis.compute_risk_adjusted_ratios(wf_res.oos_returns, risk_free_rate=0.065)
    dd_stats = _analysis.analyze_drawdown_periods(wf_res.oos_returns)
    trade_stats = _analysis.compute_trade_statistics(wf_res.oos_returns, bt_res.positions, bt_res.turnover)
    regime_perf = _analysis.analyze_market_regimes(wf_res.oos_returns)

    print(f"Sortino Ratio:             {ratios.sortino_ratio:.2f}")
    print(f"Calmar Ratio:              {ratios.calmar_ratio:.2f}")
    print(f"Omega Ratio:               {ratios.omega_ratio:.2f}")
    print(f"Max Underwater Duration:   {dd_stats.max_drawdown_duration_days} days")
    print(f"Profit Factor:             {trade_stats.profit_factor:.2f}")
    print(f"Avg Holding Period:        {trade_stats.avg_holding_days:.1f} days")
    print(f"Bull Regime Sharpe:        {regime_perf.bull_sharpe:.2f} (Bear Sharpe: {regime_perf.bear_sharpe:.2f})")
    print(f"High Vol Regime Sharpe:    {regime_perf.high_vol_sharpe:.2f} (Normal Vol: {regime_perf.normal_vol_sharpe:.2f})")

    # -------------------------------------------------------------------------
    # STAGE 10: FINAL VERDICT & EXECUTIVE TEAR-SHEET
    # -------------------------------------------------------------------------
    banner("Final Verdict & Production Readiness Verdict", "STAGE 10")
    hyp_res = _verdict.run_alpha_hypothesis_test(wf_res.oos_returns, significance_level=0.05)
    bmk_comp = _verdict.compare_with_nifty_benchmark(wf_res.oos_returns)
    tear_sheet = _verdict.generate_final_verdict_report(
        sharpe=wf_res.sharpe,
        max_drawdown_pct=dd_stats.max_drawdown_pct,
        win_rate_pct=trade_stats.win_rate_pct,
        profit_factor=trade_stats.profit_factor,
        t_stat=hyp_res.t_statistic,
        market_beta=bmk_comp.market_beta,
        oos_sharpe_retention=oos_comp.sharpe_retention_ratio,
    )

    print(f"Alpha t-statistic:         {hyp_res.t_statistic:.2f} (Confidence: {hyp_res.confidence_level_pct:.1f}%)")
    print(f"Significant Alpha (p<0.05):{hyp_res.is_statistically_significant}")
    print(f"Beta to NIFTY Benchmark:   {bmk_comp.market_beta:.3f} (Market Neutral: {bmk_comp.is_market_neutral})")
    print(f"Jensen's Alpha (Ann):      {bmk_comp.jensens_alpha_ann_pct:.2f}%")
    print(f"Information Ratio:         {bmk_comp.information_ratio:.2f}")
    print("\n" + "-" * 80)
    print(f"PRODUCTION VERDICT SCORE:  {tear_sheet.production_score} / 100")
    print(f"FINAL DECISION:            {tear_sheet.production_verdict}")
    print("-" * 80)
    print("Strengths:")
    for s in tear_sheet.key_strengths:
        print(f"  [+] {s}")
    if tear_sheet.risk_warnings:
        print("Risk Notes:")
        for r in tear_sheet.risk_warnings:
            print(f"  [!] {r}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
