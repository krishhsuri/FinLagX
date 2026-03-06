"""
TCN (Temporal Convolutional Network) Lead-Lag Detection
Advanced time series modeling using dilated causal convolutions
with enriched features and early stopping
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
import logging

from src.feature_store.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

PARQUET_PATH = "data/processed/market/aligned_market_data.parquet"
PARQUET_PATH_FALLBACK = "data/processed/aligned_dataset.parquet"

# Features to use from the enriched parquet
ENRICHED_FEATURES = [
    # Price-derived
    'returns', 'pct_returns', 'return_5d', 'return_10d', 'intraday_range',
    # Trend indicators (multi-scale)
    'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_50', 'ema_50',
    # Volatility (multi-scale)
    'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
    # Volume activity
    'volume_change',
    'volume_ma_5', 'volume_ma_10', 'volume_ma_20', 'volume_ma_50',
    # Momentum
    'rsi_14',
    # Auto-regression lags
    'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5', 'returns_lag_10',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
    # News sentiment
    'news_commodities', 'news_crypto', 'news_emerging', 'news_equities', 'news_forex',
    # Macro indicators
    'macro_cpi', 'macro_gdp', 'macro_unemployment', 'macro_fedfunds', 'macro_us10y_yield',
    'macro_cpi_change', 'macro_rate_spread', 'macro_unemployment_chg',
    # Derived features
    'price_vs_sma20', 'price_vs_sma50', 'sma_crossover', 'return_sign',
    # Overall sentiment
    'overall_sentiment_mean', 'overall_sentiment_std',
]

BASE_FEATURES = ['returns', 'volatility_20', 'sma_20', 'sma_50']

# Training hyperparameters
LOOKBACK = 30
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
TCN_CHANNELS = [64, 64, 64]
TCN_KERNEL = 3
DROPOUT = 0.2


# ============================================================================
# Model Definition
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal Block: dilated causal convolutions + residual connections
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]
        
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated causal convolutions
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], kernel_size=3, dropout=0.3):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq_len) for Conv1d
        y = self.network(x)
        y = y[:, :, -1]  # Take last timestep
        return self.fc(y)


class MonotonicRankingLoss(nn.Module):
    """
    Monotonic Logistic Ranking Loss (DeltaLag paper).
    Optimizes for correct ordering of predictions.
    """
    def __init__(self, alpha=0.5, n_pairs=64):
        super(MonotonicRankingLoss, self).__init__()
        self.alpha = alpha
        self.n_pairs = n_pairs
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        batch_size = pred.shape[0]
        mse_loss = self.mse(pred, target)
        
        if batch_size < 2:
            return mse_loss
        
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, batch_size, (n_pairs,), device=pred.device)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        
        if len(idx_i) == 0:
            return mse_loss
        
        pred_diff = pred[idx_i] - pred[idx_j]
        target_diff = target[idx_i] - target[idx_j]
        agreement = torch.tanh(pred_diff) * torch.sign(target_diff)
        ranking_loss = torch.mean(torch.log1p(torch.exp(-agreement)))
        
        return self.alpha * mse_loss + (1 - self.alpha) * ranking_loss


# ============================================================================
# Data Loading
# ============================================================================

def load_enriched_data():
    """Load the enriched parquet dataset (tries new path first, then fallback)"""
    for path in [PARQUET_PATH, PARQUET_PATH_FALLBACK]:
        if os.path.exists(path):
            logger.info(f"📂 Loading enriched dataset from {path}...")
            df = pd.read_parquet(path)
            logger.info(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns, {df['symbol'].nunique()} symbols")
            return df
    
    logger.warning(f"⚠️  No enriched parquet found at {PARQUET_PATH} or {PARQUET_PATH_FALLBACK}")
    return None


def get_available_symbols(fs, enriched_df=None):
    """Get all available symbols"""
    if enriched_df is not None:
        return sorted(enriched_df['symbol'].unique().tolist())
    with fs.engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT symbol FROM market_features ORDER BY symbol"))
        return [row[0] for row in result]


def get_granger_relationships(fs, symbol):
    """Get Granger causality relationships for a symbol"""
    logger.info(f"   🔗 Fetching Granger relationships for {symbol}...")
    
    try:
        with fs.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT asset_x, asset_y, optimal_lag, granger_score, is_significant
                FROM granger_results 
                WHERE asset_y = :symbol 
                AND is_significant = TRUE
                ORDER BY granger_score DESC
            """), {'symbol': symbol})
            
            relationships = result.fetchall()
        
        if relationships:
            logger.info(f"     Found {len(relationships)} significant lead-lag relationships")
            for asset_x, asset_y, lag, score, sig in relationships:
                logger.info(f"      • {asset_x} leads {asset_y} by {lag} days (score: {score:.4f})")
        else:
            logger.info(f"     No significant Granger relationships found")
        
        return relationships
    except Exception as e:
        logger.warning(f"     Could not fetch Granger relationships: {e}")
        return []


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data_with_leadlag(fs, symbol, enriched_df=None, lookback=LOOKBACK):
    """Prepare TCN data using enriched parquet + lead-lag features"""
    logger.info(f"\n📊 Preparing data for {symbol}...")
    
    # --- Step 1: Get base data ---
    if enriched_df is not None:
        df = enriched_df[enriched_df['symbol'] == symbol].copy()
        df = df.sort_values('time').reset_index(drop=True)
        feature_cols = [col for col in ENRICHED_FEATURES if col in df.columns]
        logger.info(f"   📥 Using enriched parquet: {len(df)} rows, {len(feature_cols)} features")
    else:
        query = """
            SELECT * FROM market_features 
            WHERE symbol = %(symbol)s 
            ORDER BY time
        """
        df = pd.read_sql(query, fs.engine, params={'symbol': symbol})
        feature_cols = [col for col in BASE_FEATURES if col in df.columns]
        logger.info(f"   📥 Using DB fallback: {len(df)} rows, {len(feature_cols)} features")
    
    if df.empty:
        logger.error(f"  ❌ No data found for {symbol}")
        return None
    
    logger.info(f"   📅 Date range: {df['time'].min()} to {df['time'].max()}")
    
    # --- Step 2: Add Granger lead-lag features ---
    granger_rels = get_granger_relationships(fs, symbol)
    granger_info = []
    
    for asset_x, asset_y, lag, score, sig in granger_rels:
        col_name = f'{asset_x}_lag{lag}'
        
        if col_name in df.columns or col_name in feature_cols:
            continue
        
        if enriched_df is not None:
            lead_df = enriched_df[enriched_df['symbol'] == asset_x][['time', 'returns']].copy()
        else:
            lead_query = """
                SELECT time, returns FROM market_features 
                WHERE symbol = %(symbol)s ORDER BY time
            """
            lead_df = pd.read_sql(lead_query, fs.engine, params={'symbol': asset_x})
        
        if not lead_df.empty:
            lead_df['time'] = pd.to_datetime(lead_df['time'])
            df['time'] = pd.to_datetime(df['time'])
            
            if lead_df['time'].dt.tz is not None:
                lead_df['time'] = lead_df['time'].dt.tz_localize(None)
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            
            lead_df[col_name] = lead_df['returns'].shift(lag)
            df = df.merge(lead_df[['time', col_name]], on='time', how='left')
            feature_cols.append(col_name)
            granger_info.append({
                'Leading_Asset': asset_x,
                'Target_Asset': asset_y,
                'Lag_Days': lag,
                'Granger_Score': float(score),
                'Feature_Name': col_name
            })
            logger.info(f"     ✅ Added lead-lag: {col_name} (score: {score:.4f})")
    
    # --- Step 2b: Add VAR features (fitted value + residual) ---
    try:
        var_query = """
            SELECT time, var_fitted_value, var_residual
            FROM var_features 
            WHERE symbol = %(symbol)s 
            ORDER BY time
        """
        var_df = pd.read_sql(var_query, fs.engine, params={'symbol': symbol})
        
        if not var_df.empty:
            var_df['time'] = pd.to_datetime(var_df['time'])
            if var_df['time'].dt.tz is not None:
                var_df['time'] = var_df['time'].dt.tz_localize(None)
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            
            df = df.merge(var_df, on='time', how='left')
            
            if 'var_fitted_value' in df.columns:
                feature_cols.append('var_fitted_value')
                feature_cols.append('var_residual')
                logger.info(f"     ✅ Added VAR features (fitted_value + residual)")
        else:
            logger.info(f"     ℹ️  No VAR features found for {symbol}")
    except Exception as e:
        logger.warning(f"     Could not fetch VAR features: {e}")
    
    # --- Step 3: Clean and normalize ---
    df = df.dropna(subset=['returns']).reset_index(drop=True)
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    features = df[feature_cols].values.astype(np.float32)
    target = df['returns'].values.astype(np.float32)
    dates = df['time'].values
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # NOTE: No normalization here to avoid data leakage.
    # Normalization happens AFTER train/test split in main().
    
    # --- Step 4: Create sequences ---
    X, y, seq_dates = [], [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
        seq_dates.append(dates[i + lookback])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    lead_lag_indicator = np.mean([g['Granger_Score'] for g in granger_info]) if granger_info else 0.0
    
    logger.info(f"   ✅ Created {len(X)} sequences (shape: {X.shape})")
    logger.info(f"   📊 Total features: {len(feature_cols)} ({len(feature_cols) - len(granger_info)} enriched + {len(granger_info)} lead-lag)")
    logger.info(f"   📍 Lead-lag indicator: {lead_lag_indicator:.4f}")
    
    return {
        'X': X,
        'y': y,
        'dates': seq_dates,
        'feature_cols': feature_cols,
        'granger_info': granger_info,
        'lead_lag_indicator': lead_lag_indicator,
        'raw_data': df
    }


# ============================================================================
# Training with Early Stopping
# ============================================================================

def train_and_predict(X_train, y_train, X_test, epochs=EPOCHS, patience=PATIENCE):
    """Train TCN with early stopping, LR scheduling, and gradient clipping"""
    logger.info(f"\n🏋️ Training TCN (max {epochs} epochs, patience {patience})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"   Device: {device}")
    
    model = TCNModel(
        input_dim=X_train.shape[2],
        num_channels=TCN_CHANNELS,
        kernel_size=TCN_KERNEL,
        dropout=DROPOUT
    ).to(device)
    
    criterion = MonotonicRankingLoss(alpha=0.3, n_pairs=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Validation split (10%)
    val_size = max(1, int(len(X_train) * 0.1))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_actual = X_train[:-val_size]
    y_train_actual = y_train[:-val_size]
    
    X_train_t = torch.FloatTensor(X_train_actual).to(device)
    y_train_t = torch.FloatTensor(y_train_actual).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        perm = torch.randperm(len(X_train_t))
        X_train_shuffled = X_train_t[perm]
        y_train_shuffled = y_train_t[perm]
        
        for i in range(0, len(X_train_shuffled), BATCH_SIZE):
            batch_X = X_train_shuffled[i:i + BATCH_SIZE]
            batch_y = y_train_shuffled[i:i + BATCH_SIZE]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"   Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )
        
        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            logger.info(f"   ⏹️  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Predict
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).cpu().numpy()
    
    logger.info("   ✅ Training complete!")
    return predictions


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(y_true, predictions):
    """Calculate performance metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    y_dir = (y_true.flatten() > 0).astype(int)
    p_dir = (predictions.flatten() > 0).astype(int)
    directional_acc = (y_dir == p_dir).mean()
    
    correlation = np.corrcoef(y_true.flatten(), predictions.flatten())[0, 1]
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Directional_Accuracy_%': directional_acc * 100,
        'Correlation': correlation,
        'Correct_Predictions': int((y_dir == p_dir).sum()),
        'Total_Predictions': len(y_dir),
        'Model': 'TCN'
    }
    
    logger.info(f"\n📈 TCN Performance Metrics:")
    logger.info(f"   RMSE:                 {rmse:.6f}")
    logger.info(f"   MAE:                  {mae:.6f}")
    logger.info(f"   Directional Accuracy: {directional_acc:.2%}")
    logger.info(f"   Correlation:          {correlation:.4f}")
    
    return metrics


# ============================================================================
# CSV Export
# ============================================================================

def export_to_csv(symbol, data_dict, y_test, predictions, metrics):
    """Export all results to CSV files with tcn_leadlag prefix"""
    
    data_folder = 'd:/FinLagX/data'
    os.makedirs(data_folder, exist_ok=True)
    
    symbol_lower = symbol.lower()
    
    logger.info(f"\n💾 Exporting TCN results to CSV files...")
    
    # File 1: Predictions
    pred_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_predictions.csv')
    pred_df = pd.DataFrame({
        'Date': data_dict['dates'],
        'Actual_Return': y_test.flatten(),
        'Predicted_Return': predictions.flatten(),
        'Prediction_Error': y_test.flatten() - predictions.flatten(),
        'Actual_Direction': ['UP' if x > 0 else 'DOWN' for x in y_test.flatten()],
        'Predicted_Direction': ['UP' if x > 0 else 'DOWN' for x in predictions.flatten()],
        'Correct_Prediction': (y_test.flatten() > 0) == (predictions.flatten() > 0),
        'Lead_Lag_Indicator': data_dict['lead_lag_indicator'],
        'Model': 'TCN'
    })
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"     Saved predictions to {pred_file}")
    
    # File 2: Metrics
    metrics_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_metrics.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"     Saved metrics to {metrics_file}")
    
    # File 3: Lead-Lag Relationships
    if data_dict['granger_info']:
        granger_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_relationships.csv')
        granger_df = pd.DataFrame(data_dict['granger_info'])
        granger_df.to_csv(granger_file, index=False)
        logger.info(f"     Saved relationships to {granger_file}")
    
    # File 4: Summary
    summary_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_summary.csv')
    n_enriched = len([f for f in data_dict['feature_cols'] if f in set(ENRICHED_FEATURES)])
    n_leadlag = len(data_dict['granger_info'])
    n_news = len([f for f in data_dict['feature_cols'] if f.startswith('news_')])
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Symbol', 'Model_Type',
            'Total_Data_Points', 'Training_Samples', 'Test_Samples',
            'Lookback_Window', 'Total_Features', 'Enriched_Features',
            'Lead_Lag_Features', 'News_Features',
            'Average_Granger_Score', 'TCN_Accuracy_%',
            'TCN_Correlation', 'TCN_RMSE',
            'TCN_Layers', 'Dilation_Pattern',
            'Model_Config', 'Generated_At'
        ],
        'Value': [
            symbol, 'Temporal Convolutional Network',
            len(data_dict['raw_data']),
            len(data_dict['X']) - len(predictions), len(predictions),
            LOOKBACK, len(data_dict['feature_cols']),
            n_enriched, n_leadlag, n_news,
            f"{data_dict['lead_lag_indicator']:.4f}",
            f"{metrics['Directional_Accuracy_%']:.2f}",
            f"{metrics['Correlation']:.4f}",
            f"{metrics['RMSE']:.6f}",
            f"{len(TCN_CHANNELS)} ({'-'.join(map(str, TCN_CHANNELS))} channels)",
            f"Exponential ({', '.join(str(2**i) for i in range(len(TCN_CHANNELS)))})",
            f"kernel={TCN_KERNEL},dropout={DROPOUT},patience={PATIENCE}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    })
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"     Saved summary to {summary_file}")
    
    logger.info(f"\n   ✅ All TCN CSV files exported for {symbol}")
    return pred_file


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("📊 TCN Lead-Lag Detection (Enriched Features + Early Stopping)")
    logger.info("="*70)
    
    # Initialize
    fs = FeatureStore()
    
    # Try loading enriched parquet
    enriched_df = load_enriched_data()
    
    if enriched_df is not None:
        logger.info(f"✅ Using enriched dataset with {len(enriched_df.columns)} features")
    else:
        logger.info("⚠️  Falling back to market_features table (limited features)")
    
    # Get available symbols
    logger.info("\n🔍 Checking available symbols...")
    symbols = get_available_symbols(fs, enriched_df)
    
    if not symbols:
        logger.error("❌ No symbols found!")
        return
    
    logger.info(f"   Found {len(symbols)} symbols: {symbols}")
    
    # Track results
    all_results = []
    
    # Process each symbol
    for idx, symbol in enumerate(symbols):
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 [{idx+1}/{len(symbols)}] Processing {symbol} with TCN")
            logger.info(f"{'='*70}")
            
            # Prepare data
            data_dict = prepare_data_with_leadlag(fs, symbol, enriched_df, lookback=LOOKBACK)
            if data_dict is None:
                continue
            
            # Train/test split (chronological)
            split_idx = int(len(data_dict['X']) * 0.8)
            X_train = data_dict['X'][:split_idx]
            X_test = data_dict['X'][split_idx:]
            y_train = data_dict['y'][:split_idx]
            y_test = data_dict['y'][split_idx:]
            data_dict['dates'] = data_dict['dates'][split_idx:]
            
            # Normalize using ONLY training set statistics (prevent data leakage)
            train_mean = np.mean(X_train.reshape(-1, X_train.shape[-1]), axis=0)
            train_std  = np.std( X_train.reshape(-1, X_train.shape[-1]), axis=0) + 1e-8
            X_train = (X_train - train_mean) / train_std
            X_test  = (X_test  - train_mean) / train_std
            
            logger.info(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test | {X_train.shape[-1]} features")
            
            if len(X_train) < 100 or len(X_test) < 10:
                logger.warning(f"   ⚠️ Not enough data for {symbol}, skipping")
                continue
            
            # Normalize targets using train-only statistics to center around 0
            y_train_mean = np.mean(y_train)
            y_train_std  = np.std(y_train) + 1e-8
            y_train_norm = (y_train - y_train_mean) / y_train_std
            y_test_norm  = (y_test  - y_train_mean) / y_train_std
            
            # Train and predict (on normalized targets)
            predictions_norm = train_and_predict(X_train, y_train_norm, X_test)
            
            # De-normalize predictions back to original scale
            predictions = predictions_norm * y_train_std + y_train_mean
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, predictions)
            
            # Export to CSV
            export_to_csv(symbol, data_dict, y_test, predictions, metrics)
            
            all_results.append({
                'Symbol': symbol,
                'Accuracy': metrics['Directional_Accuracy_%'],
                'Correlation': metrics['Correlation'],
                'RMSE': metrics['RMSE'],
                'Features': len(data_dict['feature_cols'])
            })
            
            logger.info(f"\n✅ Completed TCN for {symbol}!\n")
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 TCN FINAL RESULTS SUMMARY")
    logger.info("="*70)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        logger.info(f"\n{results_df.to_string(index=False)}")
        logger.info(f"\n   Average Accuracy:   {results_df['Accuracy'].mean():.2f}%")
        logger.info(f"   Average Correlation: {results_df['Correlation'].mean():.4f}")
        logger.info(f"   Average RMSE:        {results_df['RMSE'].mean():.6f}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL DONE! TCN models trained for all assets")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
