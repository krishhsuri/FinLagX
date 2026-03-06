# LSTM Predictor for FinLagX

## What It Does

Advanced LSTM model that:
1. Loads enriched features from `data/processed/aligned_dataset.parquet` (44+ features)
2. **Integrates Granger causality lead-lag relationships** (e.g., if GOLD leads BITCOIN by 3 days)
3. **Uses news sentiment features** (per-category: equities, commodities, crypto, forex, emerging)
4. **Uses macro indicators** (CPI, GDP, Fed Funds, 10Y yield, unemployment)
5. Trains with **early stopping** to avoid overfitting
6. Saves predictions to CSV files in `data/`

## Key Features

### Rich Feature Set (44+ features)
- **Base features**: returns, pct_returns, intraday_range
- **Trend indicators**: SMA (5/10/20/50), EMA (5/10/20/50)
- **Volatility**: Multiple scales (5/10/20/50 day)
- **Volume**: Moving averages at multiple windows
- **Momentum**: RSI-14
- **Lag features**: Auto-regressive return and volume lags (1/2/3/5/10 days)
- **News sentiment**: Per-category scores (equities, crypto, commodities, forex, emerging)
- **Macro indicators**: CPI, GDP, unemployment, Fed Funds rate, 10Y yield, yield spread
- **Granger lead-lag**: Automatically adds lagged returns from assets that Granger-cause the target

### Smart Training
- **Early stopping** with patience=12 (stops when validation loss plateaus)
- **Learning rate scheduling** (ReduceLROnPlateau, halves LR after 5 epochs without improvement)
- **Dropout** (0.3) for regularization
- **Gradient clipping** to prevent exploding gradients
- **Validation split** (10%) for monitoring overfitting
- **Data shuffling** each epoch for better generalization

## Usage

```bash
cd d:\FinLagX
python src/modeling/lstm_leadlag.py
```

This will:
1. Load the enriched parquet (falls back to DB if not available)
2. Find all symbols (up to 20 assets)
3. Add Granger lead-lag features from TimescaleDB
4. Train LSTM for each symbol with early stopping
5. Save all results to `data/` folder

**Estimated Runtime**: 10-25 minutes for all assets on CPU (faster with early stopping)

## What Gets Saved

For each asset, 6 CSV files in `data/`:

1. **`{symbol}_predictions.csv`** — Date, Actual/Predicted returns, Direction, Correct flag
2. **`{symbol}_metrics.csv`** — RMSE, MAE, Directional Accuracy, Correlation
3. **`{symbol}_leadlag_relationships.csv`** — Leading assets, lag days, Granger scores
4. **`{symbol}_raw_data.csv`** — Full feature dataset used for training
5. **`{symbol}_features.csv`** — Feature names and types (Base/Lead-Lag/News)
6. **`{symbol}_summary.csv`** — Model config, dataset stats, accuracy

## Model Details

- **Architecture**: 1-layer LSTM (64 hidden units) → Dropout → Dense (32) → Dropout → Output (1)
- **Input**: 20 days of enriched features (44+ columns)
- **Output**: Next-day return prediction
- **Training**: Max 80 epochs, Adam optimizer with weight decay, MSE loss
- **Early Stopping**: Patience 12 on validation loss
- **Metrics**: RMSE, MAE, Directional Accuracy, Correlation

## Configuration

Edit the constants at the top of `lstm_leadlag.py`:

```python
LOOKBACK = 20       # Look-back window (days)
EPOCHS = 80         # Maximum epochs
PATIENCE = 12       # Early stopping patience
HIDDEN_DIM = 64     # LSTM hidden size
DROPOUT = 0.3       # Dropout rate
BATCH_SIZE = 32     # Training batch size
LEARNING_RATE = 0.001  # Initial learning rate
```

## Requirements

- Enriched parquet at `data/processed/aligned_dataset.parquet` (run `build_features.py` first)
- PyTorch installed: `pip install torch`
- TimescaleDB running (for Granger relationships)

## Troubleshooting

**"Enriched parquet not found"**  
→ Run: `python src/preprocessing/build_features.py`

**"Module not found: torch"**  
→ Run: `pip install torch`

**Database connection error**  
→ Check: `docker ps` (TimescaleDB should be running)

**Training too slow?**
→ Early stopping should help, but you can also reduce `EPOCHS` or `HIDDEN_DIM`

---

**Author**: FinLagX Research Team  
**Updated**: March 2026  
**Model Type**: LSTM with Early Stopping + Enriched Features
