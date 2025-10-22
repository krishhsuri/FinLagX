# src/preprocessing/build_features.py

import pandas as pd
import os
import logging
from src.data_ingestion.market_data import get_latest_data
from src.data_ingestion.news_data import get_news_data
from src.data_storage.database_setup import test_connection

# --- Configuration ---
OUTPUT_PATH = "data/processed/market/aligned_market_data.parquet"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_feature_dataset():
    """
    Fetches market and news data, aligns them by date, and saves the combined feature set.
    """
    logger.info("🚀 Starting feature building pipeline...")

    # 0. Test database connections
    logger.info("🔗 Testing database connections...")
    try:
        test_connection() # This will test both TimescaleDB and MongoDB
    except Exception as e:
        logger.error(f"❌ Database connection failed. Ensure Docker is running and .env is correct. Error: {e}")
        return

    # 1. Fetch and pivot market data
    logger.info("Fetching market data from TimescaleDB...")
    # Fetch a large number of records to ensure we have enough data
    market_data = get_latest_data(limit=10000)
    if market_data.empty:
        logger.error("❌ No market data found. Exiting.")
        return

    market_pivot = market_data.pivot_table(
        index='time', columns='symbol', values='close_price'
    ).sort_index()
    # Ensure index is datetime and timezone-naive for easier joining
    market_pivot.index = pd.to_datetime(market_pivot.index).tz_localize(None)

    # 2. Fetch and aggregate news sentiment data
    logger.info("Fetching news data from MongoDB...")
    # Fetch a large number of articles to ensure good coverage
    news_data = get_news_data(limit=100000)
    if news_data.empty or 'analysis.sentiment_score' not in news_data.columns:
        logger.warning("⚠️ No news or sentiment data found. Proceeding without sentiment features.")
        daily_sentiment = pd.Series(dtype=float)
    else:
        # Ensure timestamp is datetime and timezone-naive
        news_data['time'] = pd.to_datetime(news_data['timestamp']).dt.tz_convert(None)
        # Extract just the date part for daily aggregation
        news_data['date'] = news_data['time'].dt.date
        # Calculate the average sentiment score for each day
        daily_sentiment = news_data.groupby('date')['analysis.sentiment_score'].mean()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

    # 3. Join datasets
    logger.info("Joining market and sentiment data...")
    final_df = market_pivot.copy()
    if not daily_sentiment.empty:
        # Join the sentiment series, renaming it to 'Sentiment'
        final_df = final_df.join(daily_sentiment.rename('Sentiment'), how='left')
        # Forward-fill sentiment scores for days with no news (e.g., weekends)
        final_df['Sentiment'] = final_df['Sentiment'].ffill()

    # Drop any remaining rows with NaN values (e.g., at the beginning before sentiment data starts)
    final_df = final_df.dropna()

    # 4. Save the final dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH)
    logger.info(f"✅ Successfully saved aligned feature dataset to '{OUTPUT_PATH}'. Shape: {final_df.shape}")
    print("\n--- Final Dataset Head ---")
    print(final_df.head())

if __name__ == "__main__":
    build_feature_dataset()
