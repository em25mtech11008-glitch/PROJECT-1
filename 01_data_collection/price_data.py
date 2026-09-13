from __future__ import annotations
import os
import time
import hashlib
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")


def download_nse_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    batch_size: int = 25,
    cache: bool = True,
) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tickers_hash = hashlib.md5("".join(sorted(tickers)).encode()).hexdigest()[:8]
    cache_key = f"nse_prices_{start}_{end or 'latest'}_{len(tickers)}_{tickers_hash}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            time.sleep(2)
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

        if len(batch) == 1:
            if "Close" in data.columns:
                closes = data[["Close"]].rename(columns={"Close": batch[0]})
            else:
                closes = pd.DataFrame(index=data.index)
        else:
            if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.levels[0]:
                closes = data["Close"]
            else:
                closes = pd.DataFrame(index=data.index)
        frames.append(closes)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    if cache and not prices.empty:
        prices.to_parquet(cache_path)

    return prices
