🚀 Bitcoin Macro-Forecasting System (VAR v16)
A quantitative framework exploring global liquidity regimes and Bitcoin price action.

🧩 Investment Thesis
This project tests the hypothesis that Bitcoin is a "liquidity barometer" sensitive to Federal Reserve policy. By integrating M2 Money Stock and Fed Funds Rates as exogenous drivers, the model identifies specific liquidity regimes that precede high-volatility events.

🧬 The "Bio-Quant" Bridge
As a Biomolecular Engineering & Bioinformatics graduate (UCSC '24), I translate high-precision data techniques from genomics—such as RNA-seq normalization and stochastic modeling—into the financial domain. I treat market noise with the same scientific rigor required for gene sequence analysis.

🛠️ Tech Stack & Methodology

Language: Python (Pandas, NumPy, Statsmodels).

Architecture: Vector Autoregression (VAR) with expanding-window walk-forward validation to eliminate look-ahead bias.

Machine Learning: Random Forest for feature selection and LSTM for non-linear residual analysis.

Robustness: Bootstrap Residual Resampling is used to generate probabilistic price bands (P10/P50/P90) rather than fixed-point guesses.

📈 Results & Evaluation (The "Proof of Work")
The following outputs in the /results folder validate the model's performance:

var_forecast_v16.csv (Probabilistic Forecasting): Provides a 12-month trajectory using quantile bands. This demonstrates a professional approach to Risk Management.

var_horizon_quality_v16.csv (Model Reliability): Automatically labels horizons as Green, Yellow, or Red based on MAPE. This "honesty report" shows when the model is actionable versus when the market is too noisy.

var_walkforward_v16.csv (Strategic Alpha): This backtest proves that adding M2 Money Supply significantly improves prediction accuracy over a standard univariate baseline.
