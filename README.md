# Market Mirror 📈

> Find historical market setups similar to today — and see what happened next.

🔗 **Live app:** https://options-regime-decoder-7phl8xei8lef9tvhlwlbx9.streamlit.app

## What it does
Instead of predicting price, Market Mirror asks:
*"When has this stock looked exactly like this before — and what happened next?"*

It finds the top-K most similar historical trading days using cosine similarity
across 20 technical features, then shows the distribution of forward returns.

## Features
- 20 technical indicators: RSI, MACD, EMA/SMA, Bollinger Bands, volume, candlestick patterns, sector ETF correlation
- K-Means clustering with elbow method to identify market regimes
- PCA visualisation of where today sits in historical space
- k-NN similarity engine (cosine distance) to find matching historical setups
- Random Forest cross-validation of the signal
- Linear regression within regimes
- Supports US stocks (AAPL, TSLA) and Indian stocks (RELIANCE, INFY, TCS)

## Stack
Python · yfinance · scikit-learn · plotly · streamlit

## Run locally
pip install -r requirements.txt
streamlit run app.py
