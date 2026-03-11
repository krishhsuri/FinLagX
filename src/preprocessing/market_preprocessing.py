"""
Market Data Preprocessing - Lean Version
Only essential features for lead-lag analysis
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.data_storage.database_setup import get_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataPreprocessor:
    """Lean preprocessor - only essential features"""
    
    def __init__(self):
        self.engine = get_engine()
    
    def get_market_data(self, symbols=None, start_date=None, end_date=None):
        """Fetch market data from TimescaleDB"""
        query = """
        SELECT 
            time,
            symbol,
            category,
            close_price,
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
        
        logger.info(f"Fetching market data...")
        df = pd.read_sql(query, self.engine, params=params)
        logger.info(f"  Fetched {len(df)} rows for {df['symbol'].nunique()} symbols")
        
        return df
    
    def clean_data(self, df):
        """Basic cleaning only"""
        logger.info("🧹 Cleaning market data...")
        
        original_rows = len(df)
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['symbol', 'time'])
        
        # Forward fill missing values within each symbol
        df = df.groupby('symbol').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
        
        # Drop remaining NaN
        df = df.dropna(subset=['close_price'])
        
        # Remove extreme outliers (>50% daily change - likely errors)
        df['price_change'] = df.groupby('symbol')['close_price'].pct_change()
        outliers = (df['price_change'].abs() > 0.5) & (df['price_change'].notna())
        
        if outliers.sum() > 0:
            logger.warning(f"  Removing {outliers.sum()} outlier rows")
            df = df[~outliers]
        
        df = df.drop('price_change', axis=1)
        
        logger.info(f"  Cleaned: {original_rows} → {len(df)} rows")
        
        return df
    
    def calculate_returns(self, df):
        """Calculate log returns at multiple horizons"""
        logger.info("📊 Calculating returns...")
        
        df = df.copy()
        
        grouped = df.groupby('symbol')['close_price']
        df['returns'] = grouped.transform(lambda x: np.log(x / x.shift(1)))
        df['return_5d'] = grouped.transform(lambda x: np.log(x / x.shift(5)))
        df['return_10d'] = grouped.transform(lambda x: np.log(x / x.shift(10)))
        
        logger.info("  Returns calculated")
        
        return df
    
    def calculate_essential_features(self, df):
        """
        Calculates essential and advanced technical indicators:
        - 20-day volatility
        - 20-day & 50-day SMA
        - Bollinger Bands
        - MACD
        - RSI (14-day)
        - Momentum (10-day)
        - Volume change
        """
        logger.info("📈 Calculating essential and advanced technical features...")
        
        df = df.copy()
        
        # 20-day rolling volatility
        df['volatility_20'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(window=20, min_periods=20).std()
        )
        
        # 20-day simple moving average (trend indicator)
        df['sma_20'] = df.groupby('symbol')['close_price'].transform(
            lambda x: x.rolling(window=20, min_periods=20).mean()
        )
        
        # 50-day simple moving average
        df['sma_50'] = df.groupby('symbol')['close_price'].transform(
            lambda x: x.rolling(window=50, min_periods=50).mean()
        )
        
        # Bollinger Bands (20-day SMA +/- 2*STD)
        df['bb_upper'] = df['sma_20'] + 2 * df.groupby('symbol')['close_price'].transform(lambda x: x.rolling(20, min_periods=20).std())
        df['bb_lower'] = df['sma_20'] - 2 * df.groupby('symbol')['close_price'].transform(lambda x: x.rolling(20, min_periods=20).std())
        
        # MACD (12-day EMA - 26-day EMA)
        ema12 = df.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = df.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        
        # RSI (14-day)
        def compute_rsi(close_prices, window=14):
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
            
        df['rsi_14'] = df.groupby('symbol')['close_price'].transform(compute_rsi)
        
        # Momentum (10-day price change)
        df['momentum_10'] = df.groupby('symbol')['close_price'].transform(lambda x: x.pct_change(periods=10))
        
        # Volume change
        df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
        
        logger.info("  Essential and advanced technical features calculated")
        
        return df
    
    def save_to_database(self, df, table_name='market_data_processed'):
        """Save processed data to TimescaleDB"""
        logger.info(f"💾 Saving processed data to {table_name}...")
        
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
                               save=True, table_name='market_data_processed'):
        """Run lean preprocessing pipeline"""
        logger.info("  Starting LEAN market data preprocessing...\n")
        
        df = self.get_market_data(symbols, start_date, end_date)
        df = self.clean_data(df)
        df = self.calculate_returns(df)
        df = self.calculate_essential_features(df)
        
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['returns', 'volatility_20'])
        
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
    SELECT * FROM market_data_processed
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
    print("LEAN PROCESSED DATA")
    print("="*80)
    print(df.head(10))
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")