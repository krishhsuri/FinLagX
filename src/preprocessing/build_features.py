"""
Build Features Pipeline - Final Feature Dataset Builder
=========================================================
Fetches market data from TimescaleDB and news sentiment from MongoDB,
aligns them on a daily basis, and produces the final feature-rich
dataset for the modeling phase.

Output: data/processed/market/aligned_market_data.parquet
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
import logging

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_storage.database_setup import get_engine
from src.preprocessing.market_preprocessing import MarketDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = os.path.join('data', 'processed', 'market')
OUTPUT_FILE = 'aligned_market_data.parquet'


# ============================================================================
# MongoDB Helpers  (graceful fallback if MongoDB is unavailable)
# ============================================================================

def _get_mongo_client():
    """Return a MongoClient or None if pymongo / MongoDB is unavailable."""
    try:
        from pymongo import MongoClient
        MONGO_CONFIG = {
            'host': os.getenv('MONGO_HOST', 'localhost'),
            'port': int(os.getenv('MONGO_PORT', '27017')),
            'username': os.getenv('MONGO_USER', 'admin'),
            'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
            'database': os.getenv('MONGO_DB', 'finlagx_news')
        }
        client = MongoClient(
            host=MONGO_CONFIG['host'],
            port=MONGO_CONFIG['port'],
            username=MONGO_CONFIG['username'],
            password=MONGO_CONFIG['password'],
            authSource='admin',
            serverSelectionTimeoutMS=5000
        )
        # Quick connectivity check
        client.admin.command('ismaster')
        return client, MONGO_CONFIG['database']
    except Exception as e:
        logger.warning(f"MongoDB unavailable ({e}). Proceeding without news data.")
        return None, None


# ============================================================================
# Step 1 – Fetch Market Data from TimescaleDB
# ============================================================================

def fetch_market_data(symbols=None, start_date=None, end_date=None):
    """
    Fetch processed market data from TimescaleDB.

    The MarketDataPreprocessor handles:
        - OHLCV fetch from market_data table
        - Cleaning (forward-fill, outlier removal)
        - Returns (1d, 5d, 10d log returns)
        - Essential features (volatility_20, sma_20, sma_50, volume_change)

    Returns
    -------
    pd.DataFrame
        Cleaned market data with all base features.
    """
    logger.info("=" * 70)
    logger.info("STEP 1: Fetching market data from TimescaleDB")
    logger.info("=" * 70)

    preprocessor = MarketDataPreprocessor()

    # Run full preprocessing (skip saving to DB – we only need the DataFrame)
    market_df = preprocessor.run_full_preprocessing(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        save=False
    )

    if market_df.empty:
        logger.error("No market data returned from TimescaleDB.")
        return pd.DataFrame()

    logger.info(f"  Market data fetched: {market_df.shape[0]} rows, "
                f"{market_df['symbol'].nunique()} symbols")
    logger.info(f"  Date range: {market_df['time'].min()} → {market_df['time'].max()}")
    logger.info(f"  Columns: {market_df.columns.tolist()}")

    return market_df


# ============================================================================
# Step 2 – Fetch News Sentiment from MongoDB
# ============================================================================

def fetch_news_sentiment():
    """
    Fetch news articles with sentiment scores from MongoDB.

    The function reads from the ``news_articles`` collection and
    extracts article_id, timestamp, category, and sentiment_score.
    Articles without a computed sentiment score are excluded.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, category, sentiment_score
    """
    logger.info("=" * 70)
    logger.info("STEP 2: Fetching news sentiment from MongoDB")
    logger.info("=" * 70)

    client, db_name = _get_mongo_client()

    if client is None:
        logger.warning("  Skipping news sentiment (MongoDB not available).")
        return pd.DataFrame()

    try:
        db = client[db_name]
        collection = db.news_articles

        # Only grab articles that have a computed sentiment score
        query = {"analysis.sentiment_score": {"$ne": None}}
        projection = {
            "_id": 0,
            "timestamp": 1,
            "source.category": 1,
            "analysis.sentiment_score": 1,
            "title": 1,
        }

        cursor = collection.find(query, projection).sort("timestamp", 1)
        articles = list(cursor)

        if not articles:
            logger.warning("  No articles with sentiment scores found in MongoDB.")
            return pd.DataFrame()

        # Flatten nested fields
        rows = []
        for art in articles:
            rows.append({
                'timestamp': art.get('timestamp'),
                'category': art.get('source', {}).get('category', 'unknown'),
                'sentiment_score': art.get('analysis', {}).get('sentiment_score'),
                'title': art.get('title', ''),
            })

        news_df = pd.DataFrame(rows)
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])

        # Drop rows where sentiment is still None / NaN
        news_df = news_df.dropna(subset=['sentiment_score'])

        logger.info(f"  Fetched {len(news_df)} articles with sentiment scores")
        logger.info(f"  Categories: {news_df['category'].unique().tolist()}")
        logger.info(f"  Sentiment range: [{news_df['sentiment_score'].min():.2f}, "
                    f"{news_df['sentiment_score'].max():.2f}]")

        return news_df

    except Exception as e:
        logger.error(f"  Error fetching news sentiment: {e}")
        return pd.DataFrame()

    finally:
        if client:
            client.close()


# ============================================================================
# Step 3 – Aggregate Daily Sentiment
# ============================================================================

def aggregate_daily_sentiment(news_df):
    """
    Aggregate per-article sentiment into daily statistics grouped by category.

    For each (date, category) pair we compute:
        - sentiment_mean   – average sentiment score
        - sentiment_std    – standard deviation of sentiment
        - sentiment_min    – most negative score
        - sentiment_max    – most positive score
        - news_count       – number of articles

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ``date`` with one set of columns
        per news category  (e.g. ``equities_sentiment_mean``, ``crypto_news_count``).
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Aggregating daily sentiment scores")
    logger.info("=" * 70)

    if news_df.empty:
        logger.warning("  No news data to aggregate.")
        return pd.DataFrame()

    news_df = news_df.copy()
    news_df['date'] = news_df['timestamp'].dt.date
    news_df['date'] = pd.to_datetime(news_df['date'])

    # Per-day, per-category aggregation
    daily = news_df.groupby(['date', 'category']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        sentiment_min=('sentiment_score', 'min'),
        sentiment_max=('sentiment_score', 'max'),
        news_count=('sentiment_score', 'count'),
    ).reset_index()

    # Fill NaN std (happens when a single article exists for that day)
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0.0)

    # Pivot wider: one set of columns per category
    pivot_dfs = []
    for metric in ['sentiment_mean', 'sentiment_std', 'sentiment_min',
                   'sentiment_max', 'news_count']:
        piv = daily.pivot_table(
            index='date', columns='category', values=metric, fill_value=0
        )
        piv.columns = [f"{cat}_{metric}" for cat in piv.columns]
        pivot_dfs.append(piv)

    daily_wide = pd.concat(pivot_dfs, axis=1).reset_index()

    # Also create aggregate (all-categories) sentiment
    all_daily = news_df.groupby('date').agg(
        overall_sentiment_mean=('sentiment_score', 'mean'),
        overall_sentiment_std=('sentiment_score', 'std'),
        overall_news_count=('sentiment_score', 'count'),
    ).reset_index()
    all_daily['overall_sentiment_std'] = all_daily['overall_sentiment_std'].fillna(0.0)

    daily_wide = daily_wide.merge(all_daily, on='date', how='outer')

    logger.info(f"  Daily sentiment aggregated: {daily_wide.shape[0]} days, "
                f"{daily_wide.shape[1]} columns")

    return daily_wide


# ============================================================================
# Step 4 – Align & Merge (Market + Sentiment)
# ============================================================================

def align_market_and_sentiment(market_df, sentiment_daily_df):
    """
    Merge market data with daily sentiment data on the calendar date.

    Market data is keyed by (symbol, time). We extract the date from ``time``
    and perform a left join so that every market row gets the corresponding
    day's sentiment features. Days without news default to 0.

    Returns
    -------
    pd.DataFrame
        Final merged dataset.
    """
    logger.info("=" * 70)
    logger.info("STEP 4: Aligning market data with sentiment features")
    logger.info("=" * 70)

    market_df = market_df.copy()
    market_df['date'] = pd.to_datetime(market_df['time']).dt.normalize()

    if sentiment_daily_df.empty:
        logger.warning("  No sentiment data – dataset will contain market features only.")
        # Add placeholder sentiment columns
        market_df['overall_sentiment_mean'] = 0.0
        market_df['overall_sentiment_std'] = 0.0
        market_df['overall_news_count'] = 0
        aligned = market_df
    else:
        sentiment_daily_df = sentiment_daily_df.copy()
        sentiment_daily_df['date'] = pd.to_datetime(sentiment_daily_df['date']).dt.normalize()

        aligned = market_df.merge(sentiment_daily_df, on='date', how='left')

        # Fill days with no news → 0 for counts, 0.0 for sentiment scores
        sentiment_cols = [c for c in aligned.columns
                         if 'sentiment' in c or 'news_count' in c]
        for col in sentiment_cols:
            if 'count' in col:
                aligned[col] = aligned[col].fillna(0).astype(int)
            else:
                aligned[col] = aligned[col].fillna(0.0)

    # Drop the helper 'date' column
    aligned = aligned.drop(columns=['date'], errors='ignore')

    # Sort final dataset
    aligned = aligned.sort_values(['symbol', 'time']).reset_index(drop=True)

    logger.info(f"  Aligned dataset: {aligned.shape[0]} rows × {aligned.shape[1]} columns")
    logger.info(f"  Symbols: {aligned['symbol'].nunique()}")
    logger.info(f"  Date range: {aligned['time'].min()} → {aligned['time'].max()}")

    return aligned


# ============================================================================
# Step 5 – Add Derived Features
# ============================================================================

def add_derived_features(df):
    """
    Add a handful of useful derived features on top of the aligned dataset.

    Derived features:
        - price_vs_sma20  : close_price / sma_20  (price position relative to SMA)
        - price_vs_sma50  : close_price / sma_50
        - sma_crossover   : sma_20 / sma_50  (golden/death cross signal)
        - vol_regime       : quintile rank of volatility_20 (1-5)
        - return_sign      : +1 / 0 / -1 based on return direction
    """
    logger.info("=" * 70)
    logger.info("STEP 5: Adding derived features")
    logger.info("=" * 70)

    df = df.copy()

    # Price relative to moving averages
    if 'sma_20' in df.columns:
        df['price_vs_sma20'] = np.where(
            df['sma_20'] != 0,
            df['close_price'] / df['sma_20'],
            1.0
        )

    if 'sma_50' in df.columns:
        df['price_vs_sma50'] = np.where(
            df['sma_50'] != 0,
            df['close_price'] / df['sma_50'],
            1.0
        )

    # SMA crossover signal (golden cross / death cross)
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        df['sma_crossover'] = np.where(
            df['sma_50'] != 0,
            df['sma_20'] / df['sma_50'],
            1.0
        )

    # Volatility regime (per symbol)
    if 'volatility_20' in df.columns:
        df['vol_regime'] = df.groupby('symbol')['volatility_20'].transform(
            lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        )

    # Return direction sign
    if 'returns' in df.columns:
        df['return_sign'] = np.sign(df['returns'])

    new_cols = [c for c in ['price_vs_sma20', 'price_vs_sma50',
                            'sma_crossover', 'vol_regime', 'return_sign']
                if c in df.columns]
    logger.info(f"  Added derived features: {new_cols}")

    return df


# ============================================================================
# Step 6 – Save Final Dataset
# ============================================================================

def save_final_dataset(df, output_dir=OUTPUT_DIR, output_file=OUTPUT_FILE):
    """
    Save the final aligned dataset to Parquet with gzip compression.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str
    output_file : str

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    logger.info("=" * 70)
    logger.info("STEP 6: Saving final feature dataset")
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_file)

    # Ensure no object-type columns that would break parquet
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.to_parquet(filepath, index=False, compression='gzip')

    file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
    logger.info(f"  Saved to: {filepath} ({file_size_mb:.2f} MB)")
    logger.info(f"  Shape: {df.shape}")

    return os.path.abspath(filepath)


# ============================================================================
# Main Pipeline
# ============================================================================

def build_features(symbols=None, start_date=None, end_date=None):
    """
    Run the complete Build Features pipeline.

    Pipeline steps:
        1. Fetch market data from TimescaleDB
        2. Fetch news sentiment from MongoDB
        3. Aggregate daily sentiment statistics
        4. Align and merge market + sentiment data
        5. Add derived features
        6. Save final dataset to Parquet

    Parameters
    ----------
    symbols : list[str], optional
        Restrict to specific ticker symbols.
    start_date : str, optional
        Start date for the dataset (YYYY-MM-DD).
    end_date : str, optional
        End date for the dataset (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        The final feature-rich aligned dataset.
    """
    logger.info("\n" + "█" * 70)
    logger.info("  FINLAGX — BUILD FEATURES PIPELINE")
    logger.info("█" * 70 + "\n")

    start_time = datetime.now()

    # Step 1 – Market data
    market_df = fetch_market_data(symbols, start_date, end_date)
    if market_df.empty:
        logger.error("Pipeline aborted: no market data available.")
        return pd.DataFrame()

    # Step 2 – News sentiment
    news_df = fetch_news_sentiment()

    # Step 3 – Daily aggregation
    sentiment_daily = aggregate_daily_sentiment(news_df)

    # Step 4 – Align & merge
    aligned_df = align_market_and_sentiment(market_df, sentiment_daily)

    # Step 5 – Derived features
    final_df = add_derived_features(aligned_df)

    # Step 6 – Save
    saved_path = save_final_dataset(final_df)

    # Summary
    duration = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  BUILD FEATURES PIPELINE — COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Output file   : {saved_path}")
    logger.info(f"  Total rows    : {len(final_df):,}")
    logger.info(f"  Total columns : {final_df.shape[1]}")
    logger.info(f"  Symbols       : {final_df['symbol'].nunique()}")
    logger.info(f"  Date range    : {final_df['time'].min()} → {final_df['time'].max()}")
    logger.info(f"  Duration      : {duration}")
    logger.info(f"  Memory        : {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Column breakdown
    logger.info(f"\n  Feature columns ({final_df.shape[1]}):")
    for i, col in enumerate(final_df.columns, 1):
        dtype = final_df[col].dtype
        non_null = final_df[col].notna().sum()
        logger.info(f"    {i:3d}. {col:<40s} {str(dtype):<15s} ({non_null:,} non-null)")

    logger.info("\n" + "=" * 70)
    logger.info("  Ready for modeling! Next steps:")
    logger.info("    1. python -m src.modeling.granger_causality")
    logger.info("    2. python -m src.modeling.lstm_leadlag")
    logger.info("    3. python -m src.modeling.tcn_leadlag")
    logger.info("=" * 70 + "\n")

    return final_df


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='FinLagX Build Features — Create final aligned dataset'
    )
    parser.add_argument(
        '--symbols', nargs='+', default=None,
        help='Restrict to specific symbols (e.g. ^GSPC BTC-USD)'
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='Start date YYYY-MM-DD'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date YYYY-MM-DD'
    )

    args = parser.parse_args()

    df = build_features(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if not df.empty:
        print("\n" + "=" * 70)
        print("  SAMPLE OUTPUT (first 10 rows)")
        print("=" * 70)
        print(df.head(10).to_string())
        print(f"\nShape: {df.shape}")
    else:
        print("\n  Pipeline produced no output. Check logs above for errors.")
        sys.exit(1)
