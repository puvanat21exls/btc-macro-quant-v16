#!/usr/bin/env python3
"""
Fetch BTC daily history (CryptoCompare) and aggregate to monthly close prices.

- Outputs the last daily close price for each month.
- Excludes the current (incomplete) month (UTC-based).
- Prices rounded to 2 decimal places.
- Saves to: btc_monthly.csv
"""

import sys
import time
import requests
import pandas as pd

URL = "https://min-api.cryptocompare.com/data/histoday"
DEFAULT_PARAMS = {
    "fsym": "BTC",
    "tsym": "USD",
    "allData": "true",  # full daily history
}

def fetch_with_retries(url, params, headers=None, retries=3, backoff=2.0):
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = backoff * attempt
            print(f"[WARN] HTTP {r.status_code}. Retrying in {wait:.1f}s... (attempt {attempt}/{retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"[ERROR] HTTP {r.status_code}: {r.text}", file=sys.stderr)
        r.raise_for_status()
    r.raise_for_status()
    return r

def load_daily():
    resp = fetch_with_retries(URL, DEFAULT_PARAMS, headers={"User-Agent": "btc-monthly-fetch/1.0"})
    js = resp.json()
    if "Data" not in js:
        print("Unexpected response payload:", js, file=sys.stderr)
        raise SystemExit(1)
    df = pd.DataFrame(js["Data"])
    for col in ["time", "close"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df = df.set_index("date").sort_index()
    df = df[df["close"].notna()]

    # Exclude the current (incomplete) month (UTC)
    current_month = pd.Timestamp.utcnow().to_period("M")
    df = df[df.index.to_period("M") != current_month]

    return df

def monthly_last_close(df):
    s = df["close"].resample("M").last().round(2)
    return s.reset_index().rename(columns={"close": "btc_price"})

def main():
    print("[INFO] Fetching BTC daily history from CryptoCompare...")
    df = load_daily()
    if len(df) == 0:
        print("[WARN] No data after filtering (API returned only current month?).", file=sys.stderr)
    else:
        print(f"[INFO] Got {len(df):,} daily rows from {df.index.min().date()} to {df.index.max().date()} (current month excluded)")

    out_df = monthly_last_close(df)
    out_df.to_csv("btc_monthly.csv", index=False, float_format="%.2f")
    print(f"[INFO] Wrote btc_monthly.csv ({len(out_df):,} rows)")
    print(out_df.tail(3).to_string(index=False))  # preview last 3 months

if __name__ == "__main__":
    main()