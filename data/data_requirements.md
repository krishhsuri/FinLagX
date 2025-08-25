# FinLagX – Data Requirements

This document outlines all the datasets required for the FinLagX project, their sources, update frequencies, and preprocessing steps.

---

## 📂 Data Layers Overview

| **Data Layer**            | **What We Collect**                                | **Source / API**                                       | **Frequency**      | **Processing Notes**                                                                 |
|---------------------------|---------------------------------------------------|-------------------------------------------------------|-------------------|------------------------------------------------------------------------------------|
| **Market Prices (Core)**  | OHLCV (Open, High, Low, Close, Volume) for ~20–30 global assets (indices, commodities, FX, crypto) | Yahoo Finance (`yfinance`), Binance API               | Daily / Intraday  | Compute log returns, normalize, create lag features.                              |
| **Finance News Headlines**| Title + Summary of financial news mapped to tickers | Yahoo Finance RSS, Reuters, Bloomberg RSS, `newspaper3k` | Daily             | Clean text, deduplicate, run **FinBERT** for sentiment embeddings.                |
| **Inflation (CPI)**       | CPI Index, % change YoY                           | FRED API (`CPIAUCSL`)                                 | Monthly           | Forward-fill monthly values to daily for model alignment.                         |
| **Interest Rates**        | Fed Funds Rate, 10Y Treasury Yield                | FRED API (`FEDFUNDS`, `DGS10`)                        | Daily/Monthly     | Compute yield curve slope (10Y-2Y).                                               |
| **GDP Growth**            | Quarterly GDP growth rates                        | IMF API, World Bank Data                              | Quarterly         | Interpolate quarterly values; derive YoY.                                         |
| **Unemployment**          | National unemployment rate                        | FRED API (`UNRATE`)                                   | Monthly           | Forward-fill and smooth.                                                          |
| **VIX**                   | Volatility Index (Fear Index)                     | Yahoo Finance (`^VIX`)                                | Daily             | Direct input feature; aligns with sentiment signals.                              |
| **Social Media Sentiment**| Posts/tweets about key assets/markets             | Reddit Pushshift API, Twitter `snscrape`              | Daily/Hourly      | Clean text, deduplicate, FinBERT/Roberta for sentiment embeddings.               |

---

## 🔑 Integration Plan

| Phase | What’s Integrated                                | Goal                                                |
|------|--------------------------------------------------|----------------------------------------------------|
| **Phase 1 (Weeks 1–5)**  | Market Prices + News Headlines                | Build baseline deep learning models (LSTM/VAR).    |
| **Phase 2 (Weeks 6–9)**  | Add Macro Indicators (CPI, Rates, GDP, etc.) | Capture regime/contextual effects on lead-lag.     |
| **Phase 3 (Weeks 10–12)**| Add Social Media Sentiment                    | Detect retail-driven volatility + trend reversal.  |
| **Phase 4 (Weeks 13+)**  | Tune, explain, visualize                      | Model explainability, dashboards, Streamlit demo.  |

---

## 🧩 Deliverables
- **market_prices.csv** → Daily OHLCV data for selected assets.
- **news_sentiment.csv** → Financial news headlines with sentiment embeddings.
- **macro_indicators.csv** → CPI, interest rates, GDP, unemployment.
- **social_sentiment.csv** → Sentiment scores from Twitter/Reddit.

Each dataset will be aligned to a common **daily date index** for seamless modeling.

