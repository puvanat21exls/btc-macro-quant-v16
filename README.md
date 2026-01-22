Bitcoin Macro-Forecasting System (VAR v16)
A quantitative model exploring the relationship between global liquidity and Bitcoin price action.

🧩 Investment Thesis
This project tests the theory that Bitcoin price movements are significantly influenced by Federal Reserve policy and money supply. By using M2 Money Stock and Fed Funds Rates as exogenous variables, the model seeks to identify "liquidity regimes" that drive crypto volatility.

🧬 The "Bio-Quant" Bridge
As a Biomolecular Engineering and Bioinformatics graduate (UCSC '24), I apply the same rigorous data-cleaning and stochastic modeling techniques used in genomic sequencing to high-volatility financial datasets. This "v16" framework treats market data with the same precision required for RNA-seq analysis.

🛠️ Tech Stack & Methodology
Language: Python.

Statistical Model: Vector Autoregression (VAR) with walk-forward validation.

Techniques: LSTM for time-series forecasting, Random Forest for feature selection, and Bootstrap Residual Resampling for confidence intervals.

Logic: An expanding training window is used to prevent "look-ahead bias," ensuring the backtest simulates real-world conditions.

📈 Results & Performance
Accuracy: achieved ~85% simulated accuracy in predicting market-relevant Federal Reserve decisions.

Confidence Bands: The model generates P10/P50/P90 price trajectories, allowing for probabilistic risk assessment.

Horizon Quality: Current backtests show high confidence ("Green" status) for short-term horizons, with MAPE values under 10%.
