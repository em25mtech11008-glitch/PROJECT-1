from __future__ import annotations
import io
import pandas as pd
import requests

# Curated benchmark mapping of major liquid Indian stocks across sectors with Yahoo ticker format (.NS)
NIFTY_SECTORS: dict[str, list[str]] = {
    "Private_Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS"
    ],
    "PSU_Banking": [
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS"
    ],
    "IT_Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS"
    ],
    "Automobile": [
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "BHARATFORG.NS"
    ],
    "Oil_Gas_Energy": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HPCL.NS", "GAIL.NS", "OIL.NS", "NTPC.NS", "POWERGRID.NS"
    ],
    "Metals_Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "JINDALSTEL.NS", "VEDL.NS", "NMDC.NS", "NATIONALUM.NS", "COALINDIA.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS"
    ],
    "Pharma_Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS"
    ],
    "Financial_Services_NBFC": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS"
    ],
    "Cement_Construction": [
        "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "LT.NS"
    ],
    "Chemicals_Fertilizers": [
        "PIDILITIND.NS", "SRF.NS", "AARTIIND.NS", "DEEPAKNTR.NS", "TATACHEM.NS", "UPL.NS"
    ],
    "Telecom_Media": [
        "BHARTIARTL.NS", "INDUSINDBK.NS", "ZEEL.NS", "SUNTV.NS", "PVRINOX.NS"
    ]
}


def get_sector_mapping() -> dict[str, str]:
    
    mapping = {}
    for sector, tickers in NIFTY_SECTORS.items():
        for t in tickers:
            mapping[t] = sector
    return mapping


def get_nifty50_constituents() -> list[str]:
  
    nifty50_base = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
        "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
        "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
        "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
        "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
    ]
    return [f"{t}.NS" for t in nifty50_base]


def get_nifty500_constituents(use_web_fallback: bool = True) -> list[str]:
    """
    Fetches official NIFTY 500 constituent list from NSE CSV endpoint,
    or falls back to the curated multi-sector universe.
    """
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

    if use_web_fallback:
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                if "Symbol" in df.columns:
                    tickers = df["Symbol"].dropna().unique().tolist()
                    return [f"{t}.NS" for t in tickers]
        except Exception:
            pass

    # Fallback to sector universe list
    all_tickers = []
    for tickers in NIFTY_SECTORS.values():
        all_tickers.extend(tickers)
    return sorted(list(set(all_tickers)))
