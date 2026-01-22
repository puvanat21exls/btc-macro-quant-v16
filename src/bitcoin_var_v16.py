"""
bitcoin_var_v16.py
====================

This module implements a walk‑forward evaluation and production forecast for a
Vector Autoregression (VAR) model designed specifically for forecasting the
Bitcoin (BTC) price.  The design follows the ``v16`` operating plan outlined
by the user.  Key features include:

* **Expanding training window** – Each backtest step trains on all data from
  the start of the sample up to three months before the evaluation date.
* **Spec selection** – A small menu of VAR orders (lags on the endogenous
  variables) and exogenous Federal Reserve representations are tried for
  every evaluation date.  Specifications are scored on a tiny validation
  window (the three months immediately preceding the evaluation date) with
  heavier weight on the most immediate horizon.  The winning spec is chosen
  via a weighted mean absolute percentage error (MAPE) with an AIC tie‑break.
* **Forecast horizon** – Once the best spec is selected, the model is re‑fit
  on all data through the evaluation date and used to generate a one month
  ahead price forecast.  In the production forecast the model projects
  12 months ahead and also produces 10th/50th/90th percentile bands via
  bootstrap sampling of historical residuals.
* **Horizon confidence** – After the walk‑forward backtest, performance is
  summarised by horizon.  Horizons with MAPE < 10 % are labelled "Green",
  10–15 % as "Yellow", and >15 % as "Red".

The code is organised into three numbered sections to mirror the user’s
instructions:

1. **Setup/Patching** – data loading, feature construction, caching and
   utility functions.
2. **Walk‑forward Evaluation** – looping over evaluation dates, spec
   selection, and saving backtest results.
3. **Production Forecast** – fitting the selected spec on the full sample
   and producing a 12‑month forecast with quantile bands.

To accelerate repeated runs, intermediate data (the merged macro/crypto
dataset and the backtest results) can be cached to disk.  A ``--refresh``
flag forces the script to rebuild these caches.

This script depends on ``pandas``, ``numpy`` and ``statsmodels``.  No
external network calls are made – all input data are expected to reside in
the repository (BTC price, M2 money stock and Federal Funds Rate).  If
consumer price index (CPI) data are unavailable, a constant 2 % year‑over‑
year inflation rate is assumed when computing the "real Fed" representation.
"""

import argparse
import datetime
import json
import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


# ---------------------------------------------------------------------------
# 1. Setup / Patching
#
# This section loads the various data sources, performs the necessary
# transformations (percentage changes, lags, etc.) and prepares the merged
# dataset used by the VAR.  A simple caching mechanism is included so that
# repeated executions of the script do not need to re‑read and merge large
# CSV files.  To refresh all data from scratch, pass the ``--refresh`` flag
# when invoking this script from the command line.

def load_data(refresh: bool = False) -> pd.DataFrame:
    """Load and merge BTC, M2 and Fed funds rate data.

    Parameters
    ----------
    refresh : bool, optional
        If True the cached merged dataframe is ignored and rebuilt.

    Returns
    -------
    pd.DataFrame
        A monthly DataFrame indexed by period end date with columns:
        'btc_price', 'btc_ret', 'm2', 'm2_ret', 'fed_rate',
        'delta_fed', 'real_fed'.
    """
    cache_path = "var_data_cache_v16.pkl"
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # Load BTC monthly OHLC data.  We use the 'Close' column as the
    # representative month‑end price.  The file btcusd_m.csv ships with this
    # repository and covers Jul‑2010 through Oct‑2025.
    btc = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "btcusd_m.csv"),
        parse_dates=["Date"],
    )
    btc.rename(columns={"Date": "date", "Close": "btc_price"}, inplace=True)
    btc = btc.set_index("date").sort_index()

    # Calculate percentage change in BTC price (returns).  We use simple
    # percentage change rather than log returns for ease of interpretation.
    btc["btc_ret"] = btc["btc_price"].pct_change()

    # Load M2 money stock (seasonally adjusted) from a local CSV.  The file
    # M2SL.csv should contain two columns: observation_date, M2SL.  M2SL is
    # expressed in billions of dollars.  We align dates to month end and
    # compute percentage changes.
    m2 = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "M2SL.csv"),
        parse_dates=["observation_date"],
    )
    m2.rename(columns={"observation_date": "date", "M2SL": "m2"}, inplace=True)
    m2 = m2.set_index("date").sort_index()
    m2["m2_ret"] = m2["m2"].pct_change()

    # Load Federal Funds Rate from local CSV.  The file FEDFUNDS.csv should
    # contain two columns: observation_date, FEDFUNDS.  Values are in percent.
    fed = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "FEDFUNDS.csv"),
        parse_dates=["observation_date"],
    )
    fed.rename(columns={"observation_date": "date", "FEDFUNDS": "fed_rate"}, inplace=True)
    fed = fed.set_index("date").sort_index()
    # Forward fill missing values and compute month‑to‑month change.
    fed["fed_rate"] = fed["fed_rate"].astype(float)
    fed["fed_rate"] = fed["fed_rate"].fillna(method="ffill")
    fed["delta_fed"] = fed["fed_rate"].diff()

    # Compute a placeholder real Fed rate.  In the absence of CPI data, we
    # assume a constant 2 % year‑over‑year inflation rate.  This assumption
    # allows the specification menu to include a "real" rate while keeping
    # the implementation self contained.
    annual_inflation = 2.0  # percent
    fed["real_fed"] = fed["fed_rate"] - annual_inflation

    # Merge BTC, M2 and Fed on a monthly index.  We construct an index that
    # spans the full date range of all series and then forward fill missing
    # values to ensure alignment.
    df = pd.concat([btc[["btc_price", "btc_ret"]], m2[["m2", "m2_ret"]], fed], axis=1)
    # Intersect to monthly period end frequencies by taking the last value of
    # each calendar month.  This ensures all data series line up properly.
    df = df.resample("M").last().ffill()
    df.dropna(inplace=True)

    # Save to cache
    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh)
    return df


def prepare_exog(df: pd.DataFrame, exog_type: str, lagged: bool) -> pd.DataFrame:
    """Construct exogenous regressors for a given Fed representation.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataframe returned from ``load_data``.  Must contain
        columns 'fed_rate', 'delta_fed' and 'real_fed'.
    exog_type : {'level', 'change', 'level+change', 'real'}
        Which Fed representation to use.  'level+change' produces two
        columns, others produce one column.
    lagged : bool
        Whether to include an additional 1‑month lagged column for each
        selected representation.  When True, the returned exogenous
        dataframe will have one (for level or change or real) or two
        (for level+change) additional columns representing the t−1 values.

    Returns
    -------
    pd.DataFrame
        A dataframe of exogenous regressors aligned to ``df``.  The index
        matches ``df.index``.  Columns are named based on ``exog_type`` and
        lag status, e.g. 'fed_rate' and 'fed_rate_lag1'.  No rows are
        dropped; missing values introduced by shifting are retained for
        alignment and will be dropped when fitting the VAR.
    """
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
        # Append a 1‑month lag for each column
        lagged_cols = {}
        for col in base.columns:
            lagged_cols[f"{col}_lag1"] = base[col].shift(1)
        lagged_df = pd.DataFrame(lagged_cols)
        exog = pd.concat([base, lagged_df], axis=1)
    else:
        exog = base

    return exog


def weighted_mape(actual: np.ndarray, forecast: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean absolute percentage error.

    Parameters
    ----------
    actual : np.ndarray
        True values of shape (n,).  Must be strictly positive to avoid
        division by zero.
    forecast : np.ndarray
        Forecast values of shape (n,).
    weights : np.ndarray
        Non‑negative weights for each element.  Should sum to anything; the
        function will normalise internally.

    Returns
    -------
    float
        The weighted mean absolute percentage error, i.e. sum_i w_i
        * |a_i − f_i| / a_i / sum(w).
    """
    assert len(actual) == len(forecast) == len(weights)
    ape = np.abs(actual - forecast) / np.where(actual == 0, 1e-9, actual)
    w = weights / weights.sum()
    return float((ape * w).sum())


def compute_price_path(base_price: float, returns: np.ndarray) -> np.ndarray:
    """Given an initial price and a sequence of returns, compute the price path.

    Returns
    -------
    np.ndarray
        An array where the i‑th element is the cumulative price after
        applying the first i+1 returns.
    """
    prices = []
    current_price = base_price
    for r in returns:
        current_price = current_price * (1 + r)
        prices.append(current_price)
    return np.array(prices)


# ---------------------------------------------------------------------------
# 2. Walk‑forward Evaluation
#
# This function performs the expanding window backtest described in the v16
# specification.  For each evaluation month in the range 2020 onward it
# selects the best model specification from a small menu, fits the VAR and
# produces a one‑month ahead price forecast.  The results are saved to a CSV
# file and plotted.

def run_walk_forward(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Perform walk‑forward evaluation and return a DataFrame of results.

    The DataFrame returned has the following columns:
    - date: the end of month date for which the one‑step ahead forecast is made
    - true: the realised BTC price at that date
    - var_base: prediction from a baseline VAR (no Fed exogenous) with order
      chosen from {3, 6, 9, 12} via the same validation procedure
    - var_fed: prediction from the VAR including the selected Fed specification
    - picked_model: textual description of the chosen specification for the
      Fed‑augmented VAR
    - chosen_m2_spec: the chosen VAR lag order
    - chosen_fed_spec: the chosen Fed representation (e.g. 'level', 'change',
      'level+change', 'real') with lag information (e.g. 'lag0', 'lag01')
    - val_mape: the weighted MAPE on the validation window for the chosen spec

    The function also saves ``var_walkforward_v16.csv`` and
    ``var_walkforward_v16.png`` into the current working directory.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataset produced by ``load_data``.
    refresh : bool, optional
        If True, ignore any cached backtest results and recompute.

    Returns
    -------
    pd.DataFrame
        The walk‑forward results indexed by forecast date.
    """
    res_cache = "var_walkforward_cache_v16.pkl"
    if not refresh and os.path.exists(res_cache):
        with open(res_cache, "rb") as fh:
            walk_res = pickle.load(fh)
        return walk_res

    # Only consider evaluation dates from Jan‑2020 onward (modern regime).
    eval_start = pd.Timestamp("2020-01-31")
    eval_dates = df.index[df.index >= eval_start]

    # List of candidate VAR orders (lags on endogenous variables).  These
    # correspond to M2 lag specifications in the v16 plan.
    m2_lags = [3, 6, 9, 12]
    # List of Fed representations and whether to include the lagged term.
    fed_reps = [
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

    # Precompute the baseline orders (no Fed exogenous).  We'll scan the
    # candidate m2_lags once per evaluation date for the baseline VAR.
    for eval_date in eval_dates:
        # Skip the first few months to ensure we have enough data for large
        # lag orders and validation.  We require at least 24 months of data
        # prior to the evaluation window (train_end) and three months of
        # validation data (t‑2, t‑1, t).  If the data are insufficient we
        # simply continue to the next date.
        train_end = eval_date - pd.DateOffset(months=3)
        validation_start = eval_date - pd.DateOffset(months=2)
        # Ensure there is at least one year (12 months) of training data.
        if train_end < df.index[12]:
            continue

        # Extract training and validation data.
        train_df = df.loc[:train_end]
        val_df = df.loc[validation_start:eval_date]
        # Validation dates: three months [t-2, t-1, t]
        val_dates = list(val_df.index)

        # Precompute weights for the 3‑step validation horizon.  The first
        # horizon (one month ahead) gets weight 2, the remaining get weight 1.
        weights = np.array([2.0, 1.0, 1.0])

        # Prepare actual validation prices for BTC.
        actual_prices = df.loc[val_dates, "btc_price"].values

        # Baseline VAR (no Fed exogenous).  We scan p in m2_lags and pick
        # whichever minimises the weighted MAPE.  We also record its AIC to
        # break ties.  For the baseline, endog variables are BTC and M2
        # returns.
        base_best = None
        base_best_mape = np.inf
        base_best_aic = np.inf
        base_best_p = None
        for p in m2_lags:
            # Align endogenous data: we drop the first max(p) rows to allow
            # for lag creation.
            endog_train = train_df[["btc_ret", "m2_ret"]].dropna()
            # If there are fewer observations than required lags, skip.
            if len(endog_train) <= p:
                continue
            try:
                model = VAR(endog_train)
                res = model.fit(p)
            except Exception:
                continue
            # Forecast the next 3 months.  We use the last p observations of
            # endog_train.  Since there is no exog, exog_future is None.
            last_y = endog_train.values[-res.k_ar:]
            fc_ret = res.forecast(y=last_y, steps=3)
            # Construct price path from the last known price (train_end).
            # Retrieve the last available price on or before train_end.
            price0 = df["btc_price"].asof(train_end)
            pred_prices = compute_price_path(price0, fc_ret[:, 0])
            mape = weighted_mape(actual_prices, pred_prices, weights)
            aic = res.aic if hasattr(res, "aic") else np.inf
            if (mape < base_best_mape) or (math.isclose(mape, base_best_mape) and aic < base_best_aic):
                base_best_mape = mape
                base_best_aic = aic
                base_best_p = p
        # Fit baseline model on full data up to eval_date for the one step
        # ahead prediction.
        base_pred_price = np.nan
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

        # Fed‑augmented VAR.  Scan over all combinations of m2 lag and Fed spec.
        best_spec = None
        best_mape = np.inf
        best_aic = np.inf
        best_pred_price = np.nan
        best_desc = ""
        best_p = None
        best_fed_spec = None
        for p in m2_lags:
            # Endogenous training data
            endog_train = train_df[["btc_ret", "m2_ret"]].dropna()
            # If not enough data for this lag, skip
            if len(endog_train) <= p:
                continue
            for exog_type, lagged in fed_reps:
                exog_train = prepare_exog(df, exog_type, lagged).loc[endog_train.index]
                # Align exogenous and endogenous by dropping rows with NaNs
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
                # Forecast the next 3 months using this spec.  Build
                # exogenous future for the validation period (t-2, t-1, t).
                last_y = endog_aligned.values[-var_res.k_ar:]
                # Build exog_future matrix
                exog_future = []
                for i, d in enumerate(val_dates):
                    # Determine exog variables at date d.  We call
                    # prepare_exog on the entire df and then select rows.
                    exog_all = prepare_exog(df, exog_type, lagged)
                    row = exog_all.loc[d]
                    # Ensure row is array (1D)
                    exog_future.append(row.values.astype(float))
                exog_future = np.vstack(exog_future)
                # Forecast returns
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
                    # One step ahead prediction below

        # If no spec was found (very unlikely), skip forecasting for this date
        if best_spec is not None:
            p_opt, exog_opt, lagged_opt = best_spec
            # Fit on full data through eval_date
            endog_full = df.loc[:eval_date][["btc_ret", "m2_ret"]].dropna()
            exog_full = prepare_exog(df, exog_opt, lagged_opt).loc[endog_full.index]
            combined_full = pd.concat([endog_full, exog_full], axis=1).dropna()
            endog_full_aligned = combined_full[["btc_ret", "m2_ret"]]
            exog_full_aligned = combined_full.drop(columns=["btc_ret", "m2_ret"])
            try:
                var_final = VAR(endog_full_aligned, exog=exog_full_aligned).fit(p_opt)
                last_y_full = endog_full_aligned.values[-var_final.k_ar:]
                # Exog value for next month (eval_date + 1M)
                next_date = eval_date + pd.DateOffset(months=1)
                exog_next_all = prepare_exog(df, exog_opt, lagged_opt)
                # If next_date not in index, use last available exog row
                if next_date in exog_next_all.index:
                    exog_next = exog_next_all.loc[next_date]
                else:
                    exog_next = exog_next_all.iloc[-1]
                exog_next_arr = exog_next.values.reshape(1, -1).astype(float)
                fc_one = var_final.forecast(y=last_y_full, steps=1, exog_future=exog_next_arr)
                best_pred_price = df["btc_price"].asof(eval_date) * (1 + fc_one[0][0])
                best_desc = f"{exog_opt}{'_lag01' if lagged_opt else '_lag0'}"
                best_p = p_opt
                best_fed_spec = best_desc
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
            "chosen_m2_spec": best_p,
            "chosen_fed_spec": best_fed_spec,
            "val_mape": best_mape,
        })

    walk_res = pd.DataFrame(results).set_index("date")
    # Save cache
    with open(res_cache, "wb") as fh:
        pickle.dump(walk_res, fh)

    # Save CSV
    walk_res.to_csv("var_walkforward_v16.csv", float_format="%.6f")

    # Plot backtest results
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

    # Compute and print evaluation metrics
    # MAPE, RMSE, Directional accuracy overall and by regime slices
    def rmse(true, pred):
        return float(np.sqrt(np.nanmean((pred - true) ** 2)))

    def mape(true, pred):
        return float(np.nanmean(np.abs(pred - true) / np.where(true == 0, 1e-9, true)))

    def dir_acc(true, pred):
        # Direction of price changes relative to previous month
        truth_dir = np.sign(np.diff(true))
        pred_dir = np.sign(np.diff(pred))
        return float((truth_dir == pred_dir).mean())

    # Compute metrics for baseline and fed models
    for model_name in ["var_base", "var_fed"]:
        true_vals = walk_res["true"].values
        pred_vals = walk_res[model_name].values
        print(f"Backtest metrics for {model_name}:")
        print(f"  MAPE: {mape(true_vals, pred_vals):.2%}")
        print(f"  RMSE: {rmse(true_vals, pred_vals):.2f}")
        print(f"  DirAcc: {dir_acc(true_vals, pred_vals):.2%}")
        # Regime slices
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


# ---------------------------------------------------------------------------
# 3. Production Forecast
#
# After backtesting, the final step fits the best performing model on the
# complete dataset and produces a 12‑month ahead forecast along with P10/P50/P90
# bands.  It also summarises horizon level confidence based on backtest
# performance.

def run_forecast(df: pd.DataFrame, walk_res: pd.DataFrame) -> None:
    """Generate a 12‑month forecast using the final date’s chosen spec.

    This function fits the last selected specification (the one used for the
    most recent backtest date) on all available data and forecasts 12 months
    ahead.  It also computes prediction intervals via a simple bootstrap on
    residuals and writes the forecast table to ``var_forecast_v16.csv`` and
    ``var_forecast_v16.png``.  Horizon level statistics (MAPE, directional
    accuracy and typical price range) are summarised into
    ``var_horizon_quality_v16.csv``.
    """
    # Determine the last spec used in backtest
    last_row = walk_res.dropna(subset=["var_fed"]).iloc[-1]
    p_opt = last_row["chosen_m2_spec"]
    fed_spec = last_row["chosen_fed_spec"]
    if pd.isna(p_opt) or pd.isna(fed_spec):
        print("No valid spec found in backtest; skipping forecast generation.")
        return
    # Parse fed_spec string back to exog_type and lag flag
    if "_lag01" in fed_spec:
        exog_type = fed_spec.replace("_lag01", "")
        lagged = True
    else:
        exog_type = fed_spec.replace("_lag0", "")
        lagged = False

    # Fit final VAR on all data
    endog_full = df[["btc_ret", "m2_ret"]].dropna()
    exog_full = prepare_exog(df, exog_type, lagged).loc[endog_full.index]
    combined_full = pd.concat([endog_full, exog_full], axis=1).dropna()
    endog_aligned = combined_full[["btc_ret", "m2_ret"]]
    exog_aligned = combined_full.drop(columns=["btc_ret", "m2_ret"])
    var_final = VAR(endog_aligned, exog=exog_aligned).fit(int(p_opt))

    # Forecast 12 months ahead
    horizon = 12
    last_y = endog_aligned.values[-var_final.k_ar:]
    forecast_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="M")
    # Build exog_future for each horizon step
    exog_future = []
    exog_all = prepare_exog(df, exog_type, lagged)
    for dt in forecast_dates:
        # For exogenous variables beyond the existing index, use the last known
        # value (assume no further change).  This is a common pragmatic
        # assumption when future macro drivers are unknown.
        if dt in exog_all.index:
            row = exog_all.loc[dt]
        else:
            row = exog_all.iloc[-1]
        exog_future.append(row.values.astype(float))
    exog_future = np.vstack(exog_future)
    fc_ret = var_final.forecast(y=last_y, steps=horizon, exog_future=exog_future)
    # Convert return forecasts to price path
    last_price = df.iloc[-1]["btc_price"]
    price_path = compute_price_path(last_price, fc_ret[:, 0])

    # Bootstrap prediction intervals.  We sample residuals of the BTC return
    # equation with replacement and add them to the forecasted returns.  A
    # modest number of bootstrap draws (e.g. 200) provides stable P10/P90
    # estimates without excessive runtime.
    resid = var_final.resid["btc_ret"].dropna().values
    B = 200
    boot_prices = np.zeros((B, horizon))
    rng = np.random.default_rng(42)
    for b in range(B):
        curr_price = last_price
        for h in range(horizon):
            # Forecasted return plus bootstrap shock
            shock = rng.choice(resid)
            r_pred = fc_ret[h, 0] + shock
            curr_price = curr_price * (1 + r_pred)
            boot_prices[b, h] = curr_price
    p10 = np.nanpercentile(boot_prices, 10, axis=0)
    p50 = np.nanpercentile(boot_prices, 50, axis=0)
    p90 = np.nanpercentile(boot_prices, 90, axis=0)

    # Build forecast table and save
    forecast_table = pd.DataFrame({
        "month": [d.strftime("%Y-%m") for d in forecast_dates],
        "price_p50": p50,
        "p10": p10,
        "p90": p90,
    })
    forecast_table.to_csv("var_forecast_v16.csv", index=False, float_format="%.2f")

    # Plot last 24 months of history plus forecast with bands
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

    # Horizon confidence analysis based on backtest errors
    # For each horizon 1–12 months we compute MAPE and directional accuracy
    # across all overlapping forecast paths in the backtest.  Because the
    # walk‑forward backtest produced only one‑step ahead forecasts, we need
    # to simulate longer horizons by re‑forecasting at each evaluation date
    # using the then‑selected spec.  To avoid an exponential blow up of
    # models we restrict to the horizon of interest using the same residuals
    # bootstrap approach.  For each evaluation date we forecast h steps ahead
    # and compare with actual prices.
    horizons = range(1, horizon + 1)
    horizon_metrics = []
    for h in horizons:
        errs = []
        dirs = []
        ranges = []
        for idx, row in walk_res.dropna(subset=["var_fed"]).iterrows():
            # Determine spec for this evaluation date
            p_h = row["chosen_m2_spec"]
            fed_h = row["chosen_fed_spec"]
            if pd.isna(p_h) or pd.isna(fed_h):
                continue
            # Parse fed spec
            if "_lag01" in fed_h:
                exog_type_h = fed_h.replace("_lag01", "")
                lagged_h = True
            else:
                exog_type_h = fed_h.replace("_lag0", "")
                lagged_h = False
            # Fit model on data up to this eval date minus 1 month
            end_date = idx - pd.DateOffset(months=1)
            endog_tmp = df.loc[:end_date][["btc_ret", "m2_ret"]].dropna()
            exog_tmp = prepare_exog(df, exog_type_h, lagged_h).loc[endog_tmp.index]
            combined_tmp = pd.concat([endog_tmp, exog_tmp], axis=1).dropna()
            endog_tmp_aligned = combined_tmp[["btc_ret", "m2_ret"]]
            exog_tmp_aligned = combined_tmp.drop(columns=["btc_ret", "m2_ret"])
            # Check length
            if len(endog_tmp_aligned) <= p_h:
                continue
            try:
                var_tmp = VAR(endog_tmp_aligned, exog=exog_tmp_aligned).fit(int(p_h))
            except Exception:
                continue
            last_y_tmp = endog_tmp_aligned.values[-var_tmp.k_ar:]
            # Build exog_future for h steps
            exog_future_h = []
            exog_all_h = prepare_exog(df, exog_type_h, lagged_h)
            # Forecast horizon from next month after end_date
            f_dates = pd.date_range(end_date + pd.DateOffset(months=1), periods=h, freq="M")
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
            # Price path
            price_base = df["btc_price"].asof(end_date)
            pred_price_path = compute_price_path(price_base, fc_ret_h[:, 0])
            # Actual price at end_date + h months
            target_date = end_date + pd.DateOffset(months=h)
            if target_date not in df.index:
                continue
            true_price = df["btc_price"].asof(target_date)
            # Directional accuracy: compare change from base
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
        # Typical price range: median p90-p10 from bootstrap
        price_range = float(np.median(p90[h-1] - p10[h-1]))
        # Confidence colouring
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

    # Print a one‑liner summarising the effective horizon
    usable = horizon_df[horizon_df["Confidence"].isin(["Green", "Yellow"])]
    if not usable.empty:
        max_usable = int(usable["Horizon (mo)"].max())
        print(f"Effective horizon: 1–{max_usable} months (strong/usable). Horizons beyond that show weak confidence.")
    else:
        print("All horizons exhibit weak confidence.")

    # Optional: impulse response function for +1σ shocks to M2 and Fed change
    try:
        irf = var_final.irf(horizon)
        fig = irf.plot(orth=False)
        plt.suptitle("Impulse Response Functions (v16)")
        plt.savefig("var_irf_v16.png")
        plt.close()
    except Exception:
        pass

    # Granger causality tests
    try:
        # Test whether M2 returns Granger cause BTC returns (lags up to 12)
        print("\nGranger causality tests:")
        print("M2 → BTC_ret")
        gc_m2 = grangercausalitytests(df[["btc_ret", "m2_ret"]].dropna(), maxlag=12, verbose=False)
        # Extract p‑values of F‑tests for each lag
        for lag, res in gc_m2.items():
            f_test = res[0]["ssr_ftest"]
            pval = f_test[1]
            print(f"  Lag {lag}: p = {pval:.3f}")
        # Fed → BTC_ret using delta_fed
        print("Fed → BTC_ret")
        # Create a temporary DataFrame with btc_ret and delta_fed
        tmp = df[["btc_ret", "delta_fed"]].dropna()
        gc_fed = grangercausalitytests(tmp[["btc_ret", "delta_fed"]], maxlag=12, verbose=False)
        for lag, res in gc_fed.items():
            f_test = res[0]["ssr_ftest"]
            pval = f_test[1]
            print(f"  Lag {lag}: p = {pval:.3f}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="BTC VAR v16")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached data and recompute backtest")
    args = parser.parse_args()
    print(f"[v16] Execution started at {datetime.datetime.now()}")
    df = load_data(refresh=args.refresh)
    walk_res = run_walk_forward(df, refresh=args.refresh)
    run_forecast(df, walk_res)
    print(f"[v16] Execution finished at {datetime.datetime.now()}")


if __name__ == "__main__":
    main()