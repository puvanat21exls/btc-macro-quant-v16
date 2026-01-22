"""
var_setup_v16.py — Final Full Version
=====================================

This script automatically downloads or loads Bitcoin (BTC-USD),
M2 money supply (M2SL), and Federal Funds Rate (FEDFUNDS) data.
It handles inconsistent FRED column names and text issues in CSVs.

After running once, it saves:
    btcusd_m.csv
    M2SL.csv
    FEDFUNDS.csv
    var_data_cache_v16.pkl

Usage:
    python var_setup_v16.py
"""

import os
import pickle
import pandas as pd
import numpy as np

# Try importing required libs
try:
    import yfinance as yf
    from pandas_datareader import data as web
except ImportError:
    raise ImportError(
        "Missing dependencies. Please install them first:\n"
        "pip install yfinance pandas_datareader"
    )

# --------------------------------------------------------------------
# Downloader helpers
# --------------------------------------------------------------------

def download_btc_data(filename: str = "btcusd_m.csv"):
    print("📥 Downloading BTC-USD data from Yahoo Finance...")
    btc = yf.download("BTC-USD", start="2010-01-01", progress=False)
    btc = btc.resample("M").last().reset_index()
    btc = btc[["Date", "Open", "High", "Low", "Close"]]
    btc.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with {len(btc)} monthly rows.")
    return btc


def download_fred_data(series_id: str, filename: str):
    print(f"📥 Downloading {series_id} from FRED...")
    df = web.DataReader(series_id, "fred", start="2010-01-01")
    df = df.reset_index().rename(columns={series_id: series_id})
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with {len(df)} rows.")
    return df


# --------------------------------------------------------------------
# Main loader
# --------------------------------------------------------------------

def load_data(refresh: bool = False) -> pd.DataFrame:
    cache_path = "var_data_cache_v16.pkl"
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            print("💾 Loaded cached dataset var_data_cache_v16.pkl")
            return pickle.load(fh)

    cwd = os.getcwd()

    # -------------------- BTC data --------------------
    btc_path = os.path.join(cwd, "btcusd_m.csv")
    if not os.path.exists(btc_path):
        btc = download_btc_data(btc_path)
    else:
        btc = pd.read_csv(btc_path, parse_dates=["Date"])
        print(f"📄 Loaded existing {btc_path}")

    btc.rename(columns={"Date": "date", "Close": "btc_price"}, inplace=True)
    btc = btc.set_index("date").sort_index()

    # 🔧 Clean numeric BTC prices
    btc["btc_price"] = (
        btc["btc_price"]
        .astype(str)
        .replace("-", np.nan)
        .replace(r"[^\d\.]", "", regex=True)
    )
    btc["btc_price"] = pd.to_numeric(btc["btc_price"], errors="coerce")
    btc["btc_ret"] = btc["btc_price"].pct_change(fill_method=None)

    # -------------------- M2 data --------------------
    m2_path = os.path.join(cwd, "M2SL.csv")
    if not os.path.exists(m2_path):
        m2 = download_fred_data("M2SL", m2_path)
    else:
        m2 = pd.read_csv(m2_path)
        print(f"📄 Loaded existing {m2_path}")

    # Normalize column names
    m2.columns = [c.lower() for c in m2.columns]
    if "observation_date" in m2.columns:
        m2.rename(columns={"observation_date": "date"}, inplace=True)
    elif "date" not in m2.columns and "DATE" in m2.columns:
        m2.rename(columns={"DATE": "date"}, inplace=True)
    if "m2sl" in m2.columns:
        m2.rename(columns={"m2sl": "m2"}, inplace=True)

    m2["date"] = pd.to_datetime(m2["date"], errors="coerce")
    m2 = m2.set_index("date").sort_index()
    m2["m2"] = pd.to_numeric(m2["m2"], errors="coerce")
    m2["m2_ret"] = m2["m2"].pct_change()

    # -------------------- Fed Funds data --------------------
    fed_path = os.path.join(cwd, "FEDFUNDS.csv")
    if not os.path.exists(fed_path):
        fed = download_fred_data("FEDFUNDS", fed_path)
    else:
        fed = pd.read_csv(fed_path)
        print(f"📄 Loaded existing {fed_path}")

    fed.columns = [c.lower() for c in fed.columns]
    if "observation_date" in fed.columns:
        fed.rename(columns={"observation_date": "date"}, inplace=True)
    elif "date" not in fed.columns and "DATE" in fed.columns:
        fed.rename(columns={"DATE": "date"}, inplace=True)
    if "fedfunds" in fed.columns:
        fed.rename(columns={"fedfunds": "fed_rate"}, inplace=True)

    fed["date"] = pd.to_datetime(fed["date"], errors="coerce")
    fed = fed.set_index("date").sort_index()
    fed["fed_rate"] = pd.to_numeric(fed["fed_rate"], errors="coerce").fillna(method="ffill")
    fed["delta_fed"] = fed["fed_rate"].diff()
    fed["real_fed"] = fed["fed_rate"] - 2.0  # assume 2% inflation

    # -------------------- Merge everything --------------------
    # Merge and limit date range (start 2008 for consistency)
    df = pd.concat([btc[["btc_price", "btc_ret"]], m2[["m2", "m2_ret"]], fed], axis=1)
    df = df.resample("M").last().ffill().dropna()
    df = df.loc["2010-01-01":]  # restrict window

    # -------------------- Cache --------------------
    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh)

    print(f"\n✅ BTC, M2, and FED datasets ready — total rows: {len(df)}")
    print(f"📅 Data range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"💾 Cached merged dataset to {cache_path}")

    # Summary of missing values
    na_summary = df.isna().sum()
    print("\n🔍 Missing values per column:")
    print(na_summary[na_summary > 0] if na_summary.sum() > 0 else "None — all clean ✅")

    return df


# --------------------------------------------------------------------
# Utility functions used by other modules
# --------------------------------------------------------------------

def prepare_exog(df: pd.DataFrame, exog_type: str, lagged: bool) -> pd.DataFrame:
    if exog_type == "level":
        base = df[["fed_rate"]].copy()
    elif exog_type == "change":
        base = df[["delta_fed"]].copy()
    elif exog_type == "level+change":
        base = df[["fed_rate", "delta_fed"]].copy()
    elif exog_type == "real":
        base = df[["real_fed"]].copy()
    else:
        raise ValueError(f"Unknown exog_type: {exog_type}")

    if lagged:
        lagged_cols = {f"{c}_lag1": base[c].shift(1) for c in base.columns}
        base = pd.concat([base, pd.DataFrame(lagged_cols, index=base.index)], axis=1)
    return base


def compute_price_path(base_price: float, returns: np.ndarray) -> np.ndarray:
    prices = []
    current_price = base_price
    for r in returns:
        current_price = current_price * (1 + r)
        prices.append(current_price)
    return np.array(prices)


if __name__ == "__main__":
    df = load_data(refresh=True)
    print("\n✅ Preview of merged dataset:")
    print(df.tail())

