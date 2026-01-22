"""
var_walkforward_v16.py
======================

This script performs the expanding window walk‑forward evaluation for the
Bitcoin VAR forecasting system (v16).  It imports data loading and utility
functions from :mod:`var_setup_v16` and loops over evaluation dates from
January 2020 onward.  For each date the script selects the best model
specification from a small menu of VAR orders and Federal Reserve
representations, fits the model on all data up to three months before the
evaluation date, uses a three‑month validation window to choose the
specification, and then produces a one‑step ahead price forecast.

The results are written to ``var_walkforward_v16.csv`` and a simple line
plot ``var_walkforward_v16.png``.  Basic evaluation metrics (MAPE, RMSE,
directional accuracy) are printed to the console for both the baseline
VAR and the Fed‑augmented VAR.  Cached results are stored in
``var_walkforward_cache_v16.pkl`` to accelerate repeated runs; pass the
``--refresh`` flag to force recomputation.

Running this script should be done after ``var_setup_v16.py`` has been run
or imported at least once, ensuring that the data cache exists.  The
modular design allows the backtest and forecast to be executed separately.
"""

import argparse
import math
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from var_setup_v16 import load_data, prepare_exog, compute_price_path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def weighted_mape(actual: np.ndarray, forecast: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean absolute percentage error.

    Parameters
    ----------
    actual : np.ndarray
        True values of shape ``(n,)``.  Must be strictly positive to avoid
        division by zero.
    forecast : np.ndarray
        Forecast values of shape ``(n,)``.
    weights : np.ndarray
        Non‑negative weights for each element.  Should sum to anything; the
        function normalises internally.

    Returns
    -------
    float
        The weighted mean absolute percentage error, i.e. :math:`\sum_i
        w_i \cdot |a_i - f_i| / a_i / \sum(w)`.
    """
    assert len(actual) == len(forecast) == len(weights)
    ape = np.abs(actual - forecast) / np.where(actual == 0, 1e-9, actual)
    w = weights / weights.sum()
    return float((ape * w).sum())


def run_walk_forward(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Perform the walk‑forward backtest and return a DataFrame of results.

    The resulting DataFrame has columns:

    - ``date``: the end of month date for which the one‑step ahead forecast is made
    - ``true``: the realised BTC price at that date
    - ``var_base``: prediction from a baseline VAR (no Fed exogenous) with order
      chosen from {3, 6, 9, 12}
    - ``var_fed``: prediction from the VAR including the selected Fed specification
    - ``picked_model``: textual description of the chosen Fed specification
    - ``chosen_m2_spec``: the selected VAR lag order
    - ``chosen_fed_spec``: the selected Fed representation with lag information
    - ``val_mape``: the weighted MAPE on the validation window for the chosen spec

    The function caches results to ``var_walkforward_cache_v16.pkl`` and
    writes ``var_walkforward_v16.csv`` and ``var_walkforward_v16.png``.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataset returned from :func:`var_setup_v16.load_data`.
    refresh : bool, optional
        If ``True``, ignore any cached backtest results and recompute.

    Returns
    -------
    pd.DataFrame
        The walk‑forward results indexed by forecast date.
    """
    res_cache = "var_walkforward_cache_v16.pkl"
    if not refresh and os.path.exists(res_cache):
        with open(res_cache, "rb") as fh:
            return pickle.load(fh)

    eval_start = pd.Timestamp("2020-01-31")
    eval_dates = df.index[df.index >= eval_start]

    m2_lags: List[int] = [3, 6, 9, 12]
    fed_reps: List[Tuple[str, bool]] = [
        ("level", False),
        ("level", True),
        ("change", False),
        ("change", True),
        ("level+change", False),
        ("level+change", True),
        ("real", False),
        ("real", True),
    ]

    results = []

    for eval_date in eval_dates:
        # Define training and validation windows
        train_end = eval_date - pd.DateOffset(months=3)
        validation_start = eval_date - pd.DateOffset(months=2)
        if train_end < df.index[12]:
            continue

        train_df = df.loc[:train_end]
        val_df = df.loc[validation_start:eval_date]
        val_dates = list(val_df.index)
        weights = np.array([2.0, 1.0, 1.0])
        actual_prices = df.loc[val_dates, "btc_price"].values

        # Baseline model selection
        base_best_p = None
        base_best_mape = np.inf
        base_best_aic = np.inf
        base_pred_price = np.nan
        for p in m2_lags:
            endog_train = train_df[["btc_ret", "m2_ret"]].dropna()
            if len(endog_train) <= p:
                continue
            try:
                model = VAR(endog_train)
                res = model.fit(p)
            except Exception:
                continue
            last_y = endog_train.values[-res.k_ar:]
            fc_ret = res.forecast(y=last_y, steps=3)
            price0 = df["btc_price"].asof(train_end)
            pred_prices = compute_price_path(price0, fc_ret[:, 0])
            mape = weighted_mape(actual_prices, pred_prices, weights)
            aic = res.aic if hasattr(res, "aic") else np.inf
            if (mape < base_best_mape) or (math.isclose(mape, base_best_mape) and aic < base_best_aic):
                base_best_mape = mape
                base_best_aic = aic
                base_best_p = p
        if base_best_p is not None:
            full_endog = df.loc[:eval_date][["btc_ret", "m2_ret"]].dropna()
            if len(full_endog) > base_best_p:
                try:
                    base_model = VAR(full_endog).fit(base_best_p)
                    last_y_full = full_endog.values[-base_model.k_ar:]
                    fc = base_model.forecast(last_y_full, steps=1)
                    base_pred_price = df["btc_price"].asof(eval_date) * (1 + fc[0][0])
                except Exception:
                    base_pred_price = float("nan")

        # Fed‑augmented model selection
        best_mape = np.inf
        best_aic = np.inf
        best_spec = None
        best_pred_price = np.nan
        best_desc = ""
        for p in m2_lags:
            endog_train = train_df[["btc_ret", "m2_ret"]].dropna()
            if len(endog_train) <= p:
                continue
            for exog_type, lagged in fed_reps:
                exog_train = prepare_exog(df, exog_type, lagged).loc[endog_train.index]
                combined = pd.concat([endog_train, exog_train], axis=1).dropna()
                if len(combined) <= p:
                    continue
                endog_aligned = combined[["btc_ret", "m2_ret"]]
                exog_aligned = combined.drop(columns=["btc_ret", "m2_ret"])
                try:
                    var_model = VAR(endog_aligned, exog=exog_aligned)
                    var_res = var_model.fit(p)
                except Exception:
                    continue
                last_y = endog_aligned.values[-var_res.k_ar:]
                exog_future = []
                for d in val_dates:
                    exog_all = prepare_exog(df, exog_type, lagged)
                    row = exog_all.loc[d]
                    exog_future.append(row.values.astype(float))
                exog_future = np.vstack(exog_future)
                try:
                    fc_ret = var_res.forecast(y=last_y, steps=3, exog_future=exog_future)
                except Exception:
                    continue
                price0 = df["btc_price"].asof(train_end)
                pred_prices = compute_price_path(price0, fc_ret[:, 0])
                mape = weighted_mape(actual_prices, pred_prices, weights)
                aic = var_res.aic if hasattr(var_res, "aic") else np.inf
                if (mape < best_mape) or (math.isclose(mape, best_mape) and aic < best_aic):
                    best_mape = mape
                    best_aic = aic
                    best_spec = (p, exog_type, lagged)

        if best_spec is not None:
            p_opt, exog_opt, lagged_opt = best_spec
            endog_full = df.loc[:eval_date][["btc_ret", "m2_ret"]].dropna()
            exog_full = prepare_exog(df, exog_opt, lagged_opt).loc[endog_full.index]
            combined_full = pd.concat([endog_full, exog_full], axis=1).dropna()
            endog_full_aligned = combined_full[["btc_ret", "m2_ret"]]
            exog_full_aligned = combined_full.drop(columns=["btc_ret", "m2_ret"])
            try:
                var_final = VAR(endog_full_aligned, exog=exog_full_aligned).fit(p_opt)
                last_y_full = endog_full_aligned.values[-var_final.k_ar:]
                next_date = eval_date + pd.DateOffset(months=1)
                exog_next_all = prepare_exog(df, exog_opt, lagged_opt)
                if next_date in exog_next_all.index:
                    exog_next = exog_next_all.loc[next_date]
                else:
                    exog_next = exog_next_all.iloc[-1]
                exog_next_arr = exog_next.values.reshape(1, -1).astype(float)
                fc_one = var_final.forecast(y=last_y_full, steps=1, exog_future=exog_next_arr)
                best_pred_price = df["btc_price"].asof(eval_date) * (1 + fc_one[0][0])
                best_desc = f"{exog_opt}{'_lag01' if lagged_opt else '_lag0'}"
            except Exception:
                best_pred_price = float("nan")

        next_month = eval_date + pd.DateOffset(months=1)
        true_next_price = df["btc_price"].asof(next_month)
        results.append({
            "date": next_month,
            "true": true_next_price,
            "var_base": base_pred_price,
            "var_fed": best_pred_price,
            "picked_model": best_desc,
            "chosen_m2_spec": base_best_p if base_best_p is not None else np.nan,
            "chosen_fed_spec": best_desc if best_desc else np.nan,
            "val_mape": best_mape,
        })

    walk_res = pd.DataFrame(results).set_index("date")
    with open(res_cache, "wb") as fh:
        pickle.dump(walk_res, fh)

    walk_res.to_csv("var_walkforward_v16.csv", float_format="%.6f")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(walk_res.index, walk_res["true"], label="True", color="black")
    plt.plot(walk_res.index, walk_res["var_base"], label="VAR baseline", linestyle="--")
    plt.plot(walk_res.index, walk_res["var_fed"], label="VAR+Fed", linestyle="-")
    plt.legend()
    plt.title("Walk‑forward Backtest (v16)")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.tight_layout()
    plt.savefig("var_walkforward_v16.png")
    plt.close()

    # Print simple evaluation metrics
    def rmse(true: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((pred - true) ** 2)))

    def mape(true: np.ndarray, pred: np.ndarray) -> float:
        return float(np.nanmean(np.abs(pred - true) / np.where(true == 0, 1e-9, true)))

    def dir_acc(true: np.ndarray, pred: np.ndarray) -> float:
        truth_dir = np.sign(np.diff(true))
        pred_dir = np.sign(np.diff(pred))
        return float((truth_dir == pred_dir).mean())

    for model_name in ["var_base", "var_fed"]:
        true_vals = walk_res["true"].values
        pred_vals = walk_res[model_name].values
        print(f"Backtest metrics for {model_name}:")
        print(f"  MAPE: {mape(true_vals, pred_vals):.2%}")
        print(f"  RMSE: {rmse(true_vals, pred_vals):.2f}")
        print(f"  DirAcc: {dir_acc(true_vals, pred_vals):.2%}")
        slices = {
            "2020-2021": (walk_res.index.year <= 2021),
            "2022": (walk_res.index.year == 2022),
            "2023+": (walk_res.index.year >= 2023),
        }
        for name, mask in slices.items():
            tv = true_vals[mask]
            pv = pred_vals[mask]
            if len(tv) > 0:
                print(f"    {name} MAPE: {mape(tv, pv):.2%}, DirAcc: {dir_acc(tv, pv):.2%}")

    return walk_res


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC VAR walk‑forward (v16)")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached data and recompute backtest")
    args = parser.parse_args()
    df = load_data(refresh=args.refresh)
    run_walk_forward(df, refresh=args.refresh)


if __name__ == "__main__":
    main()