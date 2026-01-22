"""
var_forecast_v16.py
===================

This script generates a 12‑month Bitcoin price forecast using the final
model specification selected during the walk‑forward backtest.  It loads
the merged dataset via :mod:`var_setup_v16` and reads the backtest
results produced by :mod:`var_walkforward_v16` to determine the most
recently chosen VAR lag and Federal Reserve representation.  It then
fits a VAR on the full history and projects forward 12 months,
producing median (P50) and 10th/90th percentile (P10/P90) price
trajectories via bootstrap resampling of residuals.  The forecast
table is saved as ``var_forecast_v16.csv`` and a corresponding chart
``var_forecast_v16.png``.  A horizon‑quality summary of backtest
errors is written to ``var_horizon_quality_v16.csv``.

This script should be run after ``var_walkforward_v16.py`` has been
executed, as it relies on the presence of ``var_walkforward_v16.csv``.
Nevertheless, it can also compute horizon‑level statistics by reusing the
cached backtest from that file.
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from var_setup_v16 import load_data, prepare_exog, compute_price_path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def run_forecast(df: pd.DataFrame, walk_res: pd.DataFrame) -> None:
    """Generate a 12‑month forecast using the most recent backtest spec.

    Parameters
    ----------
    df : pd.DataFrame
        Merged macro/crypto dataset returned from :func:`var_setup_v16.load_data`.
    walk_res : pd.DataFrame
        Walk‑forward results dataframe created by :mod:`var_walkforward_v16`.
        Must include columns ``chosen_m2_spec`` and ``chosen_fed_spec``.
    """
    # Identify last valid spec
    last_row = walk_res.dropna(subset=["var_fed"]).iloc[-1]
    p_opt = last_row["chosen_m2_spec"]
    fed_spec = last_row["chosen_fed_spec"]
    if pd.isna(p_opt) or pd.isna(fed_spec):
        print("No valid specification found in backtest; cannot produce forecast.")
        return
    # Decode exogenous spec
    if "_lag01" in fed_spec:
        exog_type = fed_spec.replace("_lag01", "")
        lagged = True
    else:
        exog_type = fed_spec.replace("_lag0", "")
        lagged = False

    # Fit final VAR on full history
    endog_full = df[["btc_ret", "m2_ret"]].dropna()
    exog_full = prepare_exog(df, exog_type, lagged).loc[endog_full.index]
    combined_full = pd.concat([endog_full, exog_full], axis=1).dropna()
    endog_aligned = combined_full[["btc_ret", "m2_ret"]]
    exog_aligned = combined_full.drop(columns=["btc_ret", "m2_ret"])
    var_final = VAR(endog_aligned, exog=exog_aligned).fit(int(p_opt))

    horizon = 12
    last_y = endog_aligned.values[-var_final.k_ar:]
    forecast_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="M")
    # Construct exog_future for each horizon step
    exog_future = []
    exog_all = prepare_exog(df, exog_type, lagged)
    for dt in forecast_dates:
        if dt in exog_all.index:
            row = exog_all.loc[dt]
        else:
            row = exog_all.iloc[-1]
        exog_future.append(row.values.astype(float))
    exog_future = np.vstack(exog_future)
    fc_ret = var_final.forecast(y=last_y, steps=horizon, exog_future=exog_future)
    last_price = df.iloc[-1]["btc_price"]
    price_path = compute_price_path(last_price, fc_ret[:, 0])

    # Bootstrap prediction intervals
    resid = var_final.resid["btc_ret"].dropna().values
    B = 200
    boot_prices = np.zeros((B, horizon))
    rng = np.random.default_rng(42)
    for b in range(B):
        curr_price = last_price
        for h in range(horizon):
            shock = rng.choice(resid)
            r_pred = fc_ret[h, 0] + shock
            curr_price = curr_price * (1 + r_pred)
            boot_prices[b, h] = curr_price
    p10 = np.nanpercentile(boot_prices, 10, axis=0)
    p50 = np.nanpercentile(boot_prices, 50, axis=0)
    p90 = np.nanpercentile(boot_prices, 90, axis=0)

    # Save forecast table
    forecast_table = pd.DataFrame({
        "month": [d.strftime("%Y-%m") for d in forecast_dates],
        "price_p50": p50,
        "p10": p10,
        "p90": p90,
    })
    forecast_table.to_csv("var_forecast_v16.csv", index=False, float_format="%.2f")

    # Plot forecast
    plt.figure(figsize=(10, 5))
    hist_cut = df[-24:]
    plt.plot(hist_cut.index, hist_cut["btc_price"], label="History (last 24M)")
    plt.plot(forecast_dates, p50, label="Forecast (median)")
    plt.fill_between(forecast_dates, p10, p90, color="orange", alpha=0.3, label="P10–P90")
    plt.title("BTC Price Forecast with P10/P90 Bands (v16)")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("var_forecast_v16.png")
    plt.close()

    # Horizon confidence analysis
    horizon_metrics = []
    horizons = range(1, horizon + 1)
    for h in horizons:
        errs = []
        dirs = []
        # Each row in walk_res corresponds to a one‑step ahead forecast for date idx
        for idx, row in walk_res.dropna(subset=["var_fed"]).iterrows():
            p_h = row["chosen_m2_spec"]
            fed_h = row["chosen_fed_spec"]
            if pd.isna(p_h) or pd.isna(fed_h):
                continue
            # Decode spec
            if "_lag01" in fed_h:
                exog_type_h = fed_h.replace("_lag01", "")
                lagged_h = True
            else:
                exog_type_h = fed_h.replace("_lag0", "")
                lagged_h = False
            end_date = idx - pd.DateOffset(months=1)
            endog_tmp = df.loc[:end_date][["btc_ret", "m2_ret"]].dropna()
            exog_tmp = prepare_exog(df, exog_type_h, lagged_h).loc[endog_tmp.index]
            combined_tmp = pd.concat([endog_tmp, exog_tmp], axis=1).dropna()
            endog_tmp_aligned = combined_tmp[["btc_ret", "m2_ret"]]
            exog_tmp_aligned = combined_tmp.drop(columns=["btc_ret", "m2_ret"])
            if len(endog_tmp_aligned) <= p_h:
                continue
            try:
                var_tmp = VAR(endog_tmp_aligned, exog=exog_tmp_aligned).fit(int(p_h))
            except Exception:
                continue
            last_y_tmp = endog_tmp_aligned.values[-var_tmp.k_ar:]
            f_dates = pd.date_range(end_date + pd.DateOffset(months=1), periods=h, freq="M")
            exog_future_h = []
            exog_all_h = prepare_exog(df, exog_type_h, lagged_h)
            for dt in f_dates:
                if dt in exog_all_h.index:
                    row_ex = exog_all_h.loc[dt]
                else:
                    row_ex = exog_all_h.iloc[-1]
                exog_future_h.append(row_ex.values.astype(float))
            exog_future_h = np.vstack(exog_future_h)
            try:
                fc_ret_h = var_tmp.forecast(y=last_y_tmp, steps=h, exog_future=exog_future_h)
            except Exception:
                continue
            price_base = df["btc_price"].asof(end_date)
            pred_price_path = compute_price_path(price_base, fc_ret_h[:, 0])
            target_date = end_date + pd.DateOffset(months=h)
            if target_date not in df.index:
                continue
            true_price = df["btc_price"].asof(target_date)
            dir_true = np.sign(true_price - price_base)
            dir_pred = np.sign(pred_price_path[-1] - price_base)
            dirs.append(dir_true == dir_pred)
            errs.append(abs(pred_price_path[-1] - true_price) / (true_price if true_price != 0 else 1e-9))
        if len(errs) > 0:
            mape_h = float(np.mean(errs))
            dir_acc_h = float(np.mean(dirs))
        else:
            mape_h = np.nan
            dir_acc_h = np.nan
        price_range = float(np.median(p90[h - 1] - p10[h - 1]))
        if np.isnan(mape_h):
            conf = "NA"
        elif mape_h < 0.10:
            conf = "Green"
        elif mape_h < 0.15:
            conf = "Yellow"
        else:
            conf = "Red"
        horizon_metrics.append({
            "Horizon (mo)": h,
            "MAPE": mape_h,
            "DirAcc": dir_acc_h,
            "Price Range (USD)": price_range,
            "Confidence": conf,
        })
    horizon_df = pd.DataFrame(horizon_metrics)
    horizon_df.to_csv("var_horizon_quality_v16.csv", index=False)

    usable = horizon_df[horizon_df["Confidence"].isin(["Green", "Yellow"])]
    if not usable.empty:
        max_usable = int(usable["Horizon (mo)"].max())
        print(f"Effective horizon: 1–{max_usable} months (strong/usable).  Horizons beyond that show weak confidence.")
    else:
        print("All horizons exhibit weak confidence.")

    # Optional: impulse response function
    try:
        irf = var_final.irf(horizon)
        fig = irf.plot(orth=False)
        plt.suptitle("Impulse Response Functions (v16)")
        plt.savefig("var_irf_v16.png")
        plt.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC VAR forecast (v16)")
    parser.add_argument("--refresh", action="store_true", help="Refresh data cache (still requires backtest file)")
    args = parser.parse_args()
    df = load_data(refresh=args.refresh)
    # Load backtest results from CSV
    if not os.path.exists("var_walkforward_v16.csv"):
        raise FileNotFoundError(
            "var_walkforward_v16.csv not found.  Run var_walkforward_v16.py first to produce the backtest results."
        )
    walk_res = pd.read_csv("var_walkforward_v16.csv", parse_dates=["date"], index_col="date")
    run_forecast(df, walk_res)


if __name__ == "__main__":
    main()