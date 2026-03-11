# FinLagX Project - Work Log

---

## Date: October 21, 2025

---

## Key Achievement: Completion of Feature Engineering Phase

Today's session focused on implementing the final feature engineering pipeline using FinBERT for sentiment analysis and resolving critical environmental and infrastructural issues. This marks the completion of all data collection and feature enrichment tasks.

---

## 🤖 **Task 1: FinBERT Sentiment Analysis Pipeline**

- **Status:**   **Completed**
- **Details:**
  - Successfully implemented a new pipeline at `src/preprocessing/generate_sentiment.py`.
  - This script automates the process of fetching news articles from the MongoDB database.
  - It leverages the `ProsusAI/finbert` model from Hugging Face to analyze the sentiment of each article's title and summary.
  - The script now correctly updates the articles in MongoDB with a numerical sentiment score (1.0 for positive, 0.0 for neutral, -1.0 for negative), making this data ready for use in downstream models.

---

## ⚙️ **Task 2: Environment Setup & Troubleshooting**

- **Status:**   **Completed**
- **Details:**
  - **Conda Environment:** Created and configured a new, dedicated conda environment (`finlagx_env`) to resolve a critical `segmentation fault` caused by underlying library conflicts in the base environment.
  - **Docker Infrastructure:** Successfully installed and configured Docker Desktop on macOS. This resolved the `command not found` errors and allowed for the successful launch of the project's backend services (TimescaleDB, MongoDB, etc.).
  - **Dependency Management:** Updated the `requirements.txt` file to include the new `transformers` and `tqdm` libraries required for the sentiment analysis pipeline.
  - **Debugging:** Successfully diagnosed and fixed several key issues:
    - Resolved `Connection refused` errors by ensuring the Docker containers were running before executing scripts.
    - Corrected a MongoDB schema validation error by ensuring the sentiment score was saved as the correct data type (number instead of string).

---

## 📊 **Overall Project Status & Next Steps Planned**

- All data ingestion and feature engineering pipelines are now complete.
- The dataset is fully enriched with market data, macroeconomic indicators, and news sentiment scores.
- The `todo.md` file has been updated to reflect the current project status.
- **Next Task (Evening)**: Re-implement the `src/preprocessing/build_features.py` script to merge sentiment scores with market data, creating the final dataset for model integration.

---

## Date: March 3, 2026

---

## Key Achievement: Build Features Pipeline — Final Feature Dataset

Today's session focused on implementing the final Build Features pipeline (`src/preprocessing/build_features.py`) which creates the unified, feature-rich dataset required by the modeling phase.

---

## 🔧 **Task 1: Build Features Pipeline Implementation**

- **Status:**   **Completed**
- **Details:**
  - Created `src/preprocessing/build_features.py` — a comprehensive 6-step pipeline:
    1. **Fetch Market Data** from TimescaleDB via `MarketDataPreprocessor` (OHLCV, returns, volatility, SMAs).
    2. **Fetch News Sentiment** from MongoDB — extracts articles with FinBERT sentiment scores.
    3. **Aggregate Daily Sentiment** — computes per-category daily stats (mean, std, min, max, article count).
    4. **Align & Merge** — left-joins market data with daily sentiment on calendar date, filling missing days with zeros.
    5. **Derived Features** — adds price_vs_sma20, price_vs_sma50, sma_crossover, vol_regime, return_sign.
    6. **Save to Parquet** — outputs the final dataset to `data/processed/market/aligned_market_data.parquet` with gzip compression.
  - The pipeline gracefully handles scenarios where MongoDB is unavailable (proceeds with market-only features).
  - Updated `src/preprocessing/__init__.py` to export the new module.
  - CLI support with `--symbols`, `--start-date`, `--end-date` flags.

---

## 📊 **Overall Project Status & Next Steps Planned**

- **TODO Part 1 (Build Final Feature Dataset)** is now complete ✅
- Updated `todo.md` to reflect completion.
- **Next Task**: Re-run models with the new sentiment-enriched dataset (TODO Part 2).

---

## Date: March 11, 2026

---

## Key Achievement: Hitting Accuracy Goal & Pipeline Execution

Today's session focused on running the complete pipeline and upgrading the models with advanced technical indicators to improve directional accuracy.

---

## 🚀 **Task 1: Complete Pipeline Execution**

- **Status:**   **Completed**
- **Details:**
  - Ran `run_complete_pipeline.py` successfully to pull new data, preprocess it, and verify the feature store.
  - Re-ran Granger Causality tests identifying 212 significant lead-lag relationships.
  - Re-built the final feature dataset aligning market and macro data.

---

## ⚙️ **Task 2: Advanced Technical Indicators & LightGBM Accuracy Improvements**

- **Status:**   **Completed**
- **Details:**
  - Added Momentum (10-day), RSI (14-day), MACD, and Bollinger Bands to `src/preprocessing/market_preprocessing.py`.
  - Updated LSTM and TCN models to ingest the new features.
  - Formatted `lgbm_model.py` to target `SP500` specifically using the unified dataset. 
  - Achieved a strong **~55% Test Accuracy** and **>71% Test F1 Score** on the LightGBM baseline!
