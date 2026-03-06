"""
Weekly Market Data Resampler
============================
Converts daily aligned_market_data.parquet to weekly resolution.

Process:
1. Resamples OHLCV correctly:
   - Open: first of week
   - High: max of week
   - Low: min of week
   - Close: last of week
   - Volume: sum of week
2. Recalculates technical features (returns, volatility, moving averages, RSI)
   on the weekly timeframe.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'data/processed/market/aligned_market_data.parquet'
OUTPUT_FILE = 'data/processed/market/weekly_market_data.parquet'

# Required input columns before aggregation
OHLCV_COLS = ['open_price', 'high_price', 'low_price', 'close_price', 'adj_close', 'volume']
META_COLS = ['symbol', 'category']

# Aggregation rules for resampling
AGG_RULES = {
    'open_price': 'first',
    'high_price': 'max',
    'low_price': 'min',
    'close_price': 'last',
    'adj_close': 'last',
    'volume': 'sum'
}

def compute_rsi(prices, window=14):
    """Compute Relative Strength Index on weekly close prices"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Default to neutral if undefined

def process_symbol(df):
    """Group by symbol, resample to weekly, and calc features"""
    symbol = df['symbol'].iloc[0]
    category = df['category'].iloc[0]
    
    # Needs a DatetimeIndex to resample
    df = df.set_index('time')
    
    # 1. Resample to Weekly (W-FRI means week ends on Friday)
    weekly = df.resample('W-FRI').agg(AGG_RULES)
    
    # Drop weeks with no data (e.g., pure holidays)
    weekly = weekly.dropna(subset=['close_price']).copy()
    
    # 2. Recalculate Weekly Features
    
    # Returns
    weekly['returns'] = np.log(weekly['close_price'] / weekly['close_price'].shift(1))
    weekly['pct_returns'] = weekly['close_price'].pct_change()
    
    # Multi-week returns (1w = 1 row)
    weekly['return_2w'] = np.log(weekly['close_price'] / weekly['close_price'].shift(2))
    weekly['return_4w'] = np.log(weekly['close_price'] / weekly['close_price'].shift(4))
    
    # Range / Volatility
    weekly['intraday_range'] = (weekly['high_price'] - weekly['low_price']) / weekly['low_price']
    
    # Volatility (std dev of weekly returns)
    weekly['volatility_4'] = weekly['returns'].rolling(window=4).std()   # ~1 month
    weekly['volatility_12'] = weekly['returns'].rolling(window=12).std()  # ~3 months
    
    # Moving Averages (Weekly)
    weekly['sma_4'] = weekly['close_price'].rolling(window=4).mean()
    weekly['sma_12'] = weekly['close_price'].rolling(window=12).mean()
    weekly['sma_26'] = weekly['close_price'].rolling(window=26).mean()  # ~6 months
    
    weekly['ema_4'] = weekly['close_price'].ewm(span=4, adjust=False).mean()
    weekly['ema_12'] = weekly['close_price'].ewm(span=12, adjust=False).mean()
    
    # RSI
    weekly['rsi_14'] = compute_rsi(weekly['close_price'], window=14)
    
    # Volume features
    weekly['volume_change'] = weekly['volume'].pct_change()
    
    # Lagged features (last week's return, 2 weeks ago, etc.)
    weekly['returns_lag_1'] = weekly['returns'].shift(1)
    weekly['returns_lag_2'] = weekly['returns'].shift(2)
    weekly['returns_lag_4'] = weekly['returns'].shift(4)
    
    # Direction target (UP=1, DOWN=0)
    weekly['return_sign'] = np.where(weekly['returns'] > 0, 1, 0)
    
    # Add meta columns back
    weekly['symbol'] = symbol
    weekly['category'] = category
    
    # Cleanup Nans (from shifts/rolling)
    weekly = weekly.dropna()
    
    return weekly.reset_index()

def main():
    logger.info("=" * 70)
    logger.info("📈 Weekly Resampler Started")
    logger.info("=" * 70)
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
        
    logger.info(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    logger.info(f"Loaded {len(df)} daily rows.")
    symbols = df['symbol'].unique()
    
    weekly_dfs = []
    
    for i, symbol in enumerate(symbols):
        sym_df = df[df['symbol'] == symbol].copy()
        
        try:
            w_df = process_symbol(sym_df)
            weekly_dfs.append(w_df)
            logger.info(f"[{i+1}/{len(symbols)}] Processed {symbol}: {len(sym_df)} daily -> {len(w_df)} weekly rows")
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            
    if not weekly_dfs:
        logger.error("No weekly data generated!")
        return
        
    final_df = pd.concat(weekly_dfs, ignore_index=True)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE)
    
    logger.info("=" * 70)
    logger.info(f"✅ Saved weekly data: {len(final_df)} rows")
    logger.info(f"📁 Path: {OUTPUT_FILE}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
