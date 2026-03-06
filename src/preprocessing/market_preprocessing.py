"""
Market Data Preprocessing — Full Feature Pipeline
==================================================
Fetches OHLCV from TimescaleDB market_data table,
cleans it, and computes a rich feature set for modeling.

Output columns (~35):
  - OHLCV + adj_close
  - Returns (1d, 5d, 10d, pct)
  - Intraday range
  - SMA (5, 10, 20, 50), EMA (5, 10, 20, 50)
  - Volatility (5, 10, 20, 50)
  - Volume MA (5, 10, 20, 50)
  - RSI-14
  - Return lags (1, 2, 3, 5, 10), Volume lags (1, 2, 3, 5, 10)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from sqlalchemy import text
from src.data_storage.database_setup import get_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataPreprocessor:
    """Full-feature market data preprocessor"""
    
    def __init__(self):
        self.engine = get_engine()
    
    def get_market_data(self, symbols=None, start_date=None, end_date=None):
        """Fetch FULL OHLCV from TimescaleDB"""
        query = """
        SELECT 
            time,
            symbol,
            category,
            open_price,
            high_price,
            low_price,
            close_price,
            adj_close,
            volume
        FROM market_data
        WHERE 1=1
        """
        
        params = {}
        
        if symbols:
            query += " AND symbol = ANY(:symbols)"
            params['symbols'] = symbols
        
        if start_date:
            query += " AND time >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND time <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY symbol, time"
        
        logger.info("Fetching market data (full OHLCV)...")
        df = pd.read_sql(query, self.engine, params=params)
        logger.info(f"  Fetched {len(df)} rows for {df['symbol'].nunique()} symbols")
        
        return df
    
    def clean_data(self, df):
        """Clean data: sort, forward-fill, remove outliers"""
        logger.info("🧹 Cleaning market data...")
        
        original_rows = len(df)
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['symbol', 'time'])
        
        # Forward fill missing values within each symbol
        df = df.groupby('symbol', group_keys=False).apply(
            lambda x: x.ffill()
        ).reset_index(drop=True)
        
        # Drop remaining NaN in close_price
        df = df.dropna(subset=['close_price'])
        
        # Remove extreme outliers (>50% daily change — likely data errors)
        df['_pct_change'] = df.groupby('symbol')['close_price'].pct_change()
        outliers = (df['_pct_change'].abs() > 0.5) & (df['_pct_change'].notna())
        
        if outliers.sum() > 0:
            logger.warning(f"  Removing {outliers.sum()} outlier rows")
            df = df[~outliers]
        
        df = df.drop('_pct_change', axis=1)
        
        logger.info(f"  Cleaned: {original_rows} → {len(df)} rows")
        return df
    
    def calculate_returns(self, df):
        """Calculate returns at multiple horizons"""
        logger.info("📊 Calculating returns...")
        
        df = df.copy()
        grouped = df.groupby('symbol')['close_price']
        
        # Log returns
        df['returns'] = grouped.transform(lambda x: np.log(x / x.shift(1)))
        df['return_5d'] = grouped.transform(lambda x: np.log(x / x.shift(5)))
        df['return_10d'] = grouped.transform(lambda x: np.log(x / x.shift(10)))
        
        # Percentage returns (simpler interpretation)
        df['pct_returns'] = grouped.pct_change()
        
        # Intraday range: (high - low) / close — measures daily volatility
        if 'high_price' in df.columns and 'low_price' in df.columns:
            df['intraday_range'] = (df['high_price'] - df['low_price']) / df['close_price']
        
        logger.info("  Returns calculated")
        return df
    
    def calculate_features(self, df):
        """
        Compute full feature set per symbol:
          - Moving averages (SMA/EMA at 5, 10, 20, 50)
          - Volatility at multiple scales
          - Volume moving averages
          - RSI-14
          - Auto-regression lags
        """
        logger.info("📈 Calculating full feature set...")
        
        df = df.copy()
        
        windows = [5, 10, 20, 50]
        
        for w in windows:
            # Simple Moving Average
            df[f'sma_{w}'] = df.groupby('symbol')['close_price'].transform(
                lambda x: x.rolling(window=w, min_periods=w).mean()
            )
            
            # Exponential Moving Average
            df[f'ema_{w}'] = df.groupby('symbol')['close_price'].transform(
                lambda x: x.ewm(span=w, adjust=False).mean()
            )
            
            # Rolling Volatility (std of returns)
            df[f'volatility_{w}'] = df.groupby('symbol')['returns'].transform(
                lambda x: x.rolling(window=w, min_periods=w).std()
            )
            
            # Volume Moving Average
            df[f'volume_ma_{w}'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window=w, min_periods=w).mean()
            )
        
        # RSI-14
        df['rsi_14'] = df.groupby('symbol')['returns'].transform(self._compute_rsi)
        
        # Auto-regression: lagged returns and volume
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df.groupby('symbol')['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['volume'].shift(lag)
        
        # Volume change
        df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
        
        logger.info("  Full feature set calculated")
        return df
    
    @staticmethod
    def _compute_rsi(series, period=14):
        """Compute RSI for a single symbol's return series"""
        delta = series.copy()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_to_database(self, df, table_name='market_features'):
        """Save processed features to TimescaleDB"""
        logger.info(f"💾 Saving to {table_name}...")
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                conn.commit()
            
            df.to_sql(table_name, self.engine, if_exists='replace', index=False, method='multi')
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_time_symbol 
                    ON {table_name} (time, symbol)
                """))
                conn.commit()
            
            logger.info(f"  Saved {len(df)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"  Error saving to database: {e}")
            raise
    
    def run_full_preprocessing(self, symbols=None, start_date=None, end_date=None,
                               save=True, table_name='market_features'):
        """Run full preprocessing pipeline"""
        logger.info("  Starting FULL market data preprocessing...\n")
        
        # 1. Fetch full OHLCV
        df = self.get_market_data(symbols, start_date, end_date)
        
        if df.empty:
            logger.error("  No market data found!")
            return pd.DataFrame()
        
        # 2. Clean
        df = self.clean_data(df)
        
        # 3. Returns
        df = self.calculate_returns(df)
        
        # 4. Full feature set
        df = self.calculate_features(df)
        
        # 5. Drop rows where essential features are NaN (from rolling windows)
        df = df.dropna(subset=['returns', 'volatility_20'])
        
        # 6. Save to DB
        if save:
            self.save_to_database(df, table_name)
        
        logger.info("\n  Market data preprocessing completed!")
        logger.info(f"   Final shape: {df.shape}")
        logger.info(f"   Features: {df.columns.tolist()}")
        logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df


def get_processed_market_data(symbols=None, start_date=None, end_date=None):
    """Get processed market data from database"""
    engine = get_engine()
    
    query = """
    SELECT * FROM market_features
    WHERE 1=1
    """
    
    params = {}
    
    if symbols:
        query += " AND symbol = ANY(:symbols)"
        params['symbols'] = symbols
    
    if start_date:
        query += " AND time >= :start_date"
        params['start_date'] = start_date
    
    if end_date:
        query += " AND time <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY symbol, time"
    
    return pd.read_sql(query, engine, params=params)


if __name__ == "__main__":
    preprocessor = MarketDataPreprocessor()
    df = preprocessor.run_full_preprocessing()
    
    print("\n" + "="*80)
    print("FULL PROCESSED DATA")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nSample (first 5 rows):")
    print(df.head(5).to_string())
    print(f"\nMemory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")