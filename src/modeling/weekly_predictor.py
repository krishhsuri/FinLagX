"""
Weekly DeltaLag Predictor (Two-Stage)
=====================================
Uses the resampled weekly data to predict:
Stage 1: Direction (UP/DOWN) -> Binary Classification
Stage 2: Intensity (Magnitude) -> Regression

This approaches the low signal-to-noise ratio in financial data by separating
the "will it go up?" problem from the "by how much?" problem.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import time
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from src.modeling.delta_lag_detector import DeltaLagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Hyperparameters
# ============================================================================

LOOKBACK = 8           # 8 weeks (~2 months)
MAX_LAG = 4            # Up to 4 weeks lag
TOP_K = 3              # Top 3 leaders
HIDDEN_DIM = 64
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PARQUET_PATH = 'data/processed/market/weekly_market_data.parquet'

FEATURES = [
    'returns', 'pct_returns', 'return_2w', 'return_4w', 'intraday_range',
    'sma_4', 'sma_12', 'sma_26',
    'volatility_4', 'volatility_12',
    'volume_change', 'rsi_14',
    'returns_lag_1', 'returns_lag_2', 'returns_lag_4'
]

# ============================================================================
# Models
# ============================================================================

class BinaryFocalLoss(nn.Module):
    """Focal Loss for classification to handle class imbalance / hard examples"""
    def __init__(self, alpha=1, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

class Stage1DirectionModel(DeltaLagModel):
    """Classifies UP (1) vs DOWN (0)"""
    def __init__(self, feature_dim, hidden_dim=64, max_lag=4, top_k=3):
        super().__init__(feature_dim, hidden_dim, max_lag, top_k, num_layers=1)
        # Inherit exact DeltaLag architecture, just change the output intention
        # (It outputs 1 logit for BCEWithLogitsLoss)

class Stage2IntensityModel(DeltaLagModel):
    """Predicts absolute return magnitude > 0"""
    def __init__(self, feature_dim, hidden_dim=64, max_lag=4, top_k=3):
        super().__init__(feature_dim, hidden_dim, max_lag, top_k, num_layers=1)
        # Use softplus applied to MLP output to ensure magnitude is strictly positive
        self.softplus = nn.Softplus()
        
    def forward(self, target_seq, leader_raw_features):
        pred, info = super().forward(target_seq, leader_raw_features)
        return self.softplus(pred), info

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_weekly_data(df, feature_cols, lookback=LOOKBACK, max_lag=MAX_LAG):
    logger.info("🔄 Building weekly multi-asset tensors...")
    t0 = time.time()
    
    symbols = sorted(df['symbol'].unique())
    n_symbols = len(symbols)
    feature_cols_avail = [c for c in feature_cols if c in df.columns]
    n_features = len(feature_cols_avail)
    n_leaders = n_symbols - 1
    
    sym_features, sym_returns, sym_times = {}, {}, {}
    for sym in symbols:
        sdf = df[df['symbol'] == sym].sort_values('time').reset_index(drop=True)
        feats = sdf[feature_cols_avail].values.astype(np.float32)
        sym_features[sym] = np.nan_to_num(feats)
        sym_returns[sym] = sdf['returns'].values.astype(np.float32)
        sym_times[sym] = sdf['time'].values
        
    min_len = min(len(sym_features[s]) for s in symbols)
    for sym in symbols:
        off = len(sym_features[sym]) - min_len
        sym_features[sym] = sym_features[sym][off:]
        sym_returns[sym] = sym_returns[sym][off:]
        sym_times[sym] = sym_times[sym][off:]
        
    all_features = np.stack([sym_features[s] for s in symbols])
    all_returns = np.stack([sym_returns[s] for s in symbols])
    times = sym_times[symbols[0]]
    
    total = n_symbols * (min_len - lookback)
    
    target_seqs = np.zeros((total, lookback, n_features), dtype=np.float32)
    leader_raw  = np.zeros((total, n_leaders, max_lag, n_features), dtype=np.float32)
    targets     = np.zeros((total, 1), dtype=np.float32)
    sample_syms = []
    sample_dates = []
    
    idx = 0
    for ti, tsym in enumerate(symbols):
        leader_is = [i for i in range(n_symbols) if i != ti]
        
        for t in range(lookback, min_len):
            target_seqs[idx] = all_features[ti, t - lookback:t]
            targets[idx, 0] = all_returns[ti, t]
            
            for li, leader_i in enumerate(leader_is):
                for lag in range(max_lag):
                    ld = t - 1 - lag
                    if ld >= 0:
                        leader_raw[idx, li, lag] = all_features[leader_i, ld]
                        
            sample_syms.append(tsym)
            sample_dates.append(times[t])
            idx += 1
            
    logger.info(f"   ✅ Built {total} weekly samples in {time.time()-t0:.1f}s")
    
    return {
        'target_seqs': target_seqs,
        'leader_raw': leader_raw,
        'targets': targets,  # Raw returns
        'symbols': np.array(sample_syms),
        'dates': np.array(sample_dates)
    }

# ============================================================================
# Two-Stage Training
# ============================================================================

def train_stage1_direction(data, split_idx, device):
    """Train Binary Classifier for UP/DOWN"""
    logger.info("\n" + "="*50)
    logger.info("🏋️ STAGE 1: Training Direction Classifier (UP/DOWN)")
    logger.info("="*50)
    
    n_features = data['target_seqs'].shape[-1]
    
    # Target is binary: 1 if return > 0, else 0
    y_binary = (data['targets'] > 0).astype(np.float32)
    
    tr_tgt = data['target_seqs'][:split_idx]
    tr_raw = data['leader_raw'][:split_idx]
    tr_y   = y_binary[:split_idx]
    
    te_tgt = data['target_seqs'][split_idx:]
    te_raw = data['leader_raw'][split_idx:]
    te_y   = y_binary[split_idx:]
    
    train_size = int(len(tr_y) * 0.9)
    val_size = len(tr_y) - train_size
    
    model = Stage1DirectionModel(n_features, HIDDEN_DIM, MAX_LAG, TOP_K).to(device)
    criterion = BinaryFocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(train_size)
        epoch_loss = 0
        
        for i in range(0, train_size, BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            bt = torch.FloatTensor(tr_tgt[bi]).to(device)
            br = torch.FloatTensor(tr_raw[bi]).to(device)
            by = torch.FloatTensor(tr_y[bi]).to(device)
            
            optimizer.zero_grad()
            pred, _ = model(bt, br)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            vt = torch.FloatTensor(tr_tgt[-val_size:]).to(device)
            vr = torch.FloatTensor(tr_raw[-val_size:]).to(device)
            vy = torch.FloatTensor(tr_y[-val_size:]).to(device)
            vp, _ = model(vt, vr)
            val_loss = criterion(vp, vy).item()
            
            # Val accuracy
            val_preds = (torch.sigmoid(vp) > 0.5).float()
            val_acc = (val_preds == vy).float().mean().item()
            
        if (epoch+1) % 5 == 0:
            logger.info(f"   Epoch {epoch+1:2d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test Prediction
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(te_y), BATCH_SIZE):
            bt = torch.FloatTensor(te_tgt[i:i+BATCH_SIZE]).to(device)
            br = torch.FloatTensor(te_raw[i:i+BATCH_SIZE]).to(device)
            p, _ = model(bt, br)
            all_logits.append(p.cpu().numpy())
            
    test_logits = np.concatenate(all_logits)
    test_probs = 1 / (1 + np.exp(-test_logits))  # Sigmoid
    test_preds = (test_probs > 0.5).astype(int)
    
    acc = accuracy_score(te_y, test_preds)
    logger.info(f"   => Stage 1 Test Accuracy: {acc*100:.2f}%")
    
    return model, test_preds, test_probs, te_y

def train_stage2_intensity(data, split_idx, device):
    """Train Regressor for Magnitude (Absolute Return)"""
    logger.info("\n" + "="*50)
    logger.info("🏋️ STAGE 2: Training Intensity Regressor (|Return|)")
    logger.info("="*50)
    
    n_features = data['target_seqs'].shape[-1]
    
    # Target is strictly magnitude |return|
    y_mag = np.abs(data['targets'])
    
    tr_tgt = data['target_seqs'][:split_idx]
    tr_raw = data['leader_raw'][:split_idx]
    tr_y   = y_mag[:split_idx]
    
    te_tgt = data['target_seqs'][split_idx:]
    te_raw = data['leader_raw'][split_idx:]
    
    train_size = int(len(tr_y) * 0.9)
    val_size = len(tr_y) - train_size
    
    model = Stage2IntensityModel(n_features, HIDDEN_DIM, MAX_LAG, TOP_K).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(train_size)
        epoch_loss = 0
        for i in range(0, train_size, BATCH_SIZE):
            bi = perm[i:i+BATCH_SIZE]
            bt = torch.FloatTensor(tr_tgt[bi]).to(device)
            br = torch.FloatTensor(tr_raw[bi]).to(device)
            by = torch.FloatTensor(tr_y[bi]).to(device)
            
            optimizer.zero_grad()
            pred, _ = model(bt, br)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            vt = torch.FloatTensor(tr_tgt[-val_size:]).to(device)
            vr = torch.FloatTensor(tr_raw[-val_size:]).to(device)
            vy = torch.FloatTensor(tr_y[-val_size:]).to(device)
            vp, _ = model(vt, vr)
            val_loss = criterion(vp, vy).item()
            
        if (epoch+1) % 5 == 0:
            logger.info(f"   Epoch {epoch+1:2d} | Val MSE: {val_loss:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test Prediction
    model.eval()
    all_mags = []
    with torch.no_grad():
        for i in range(0, len(te_tgt), BATCH_SIZE):
            bt = torch.FloatTensor(te_tgt[i:i+BATCH_SIZE]).to(device)
            br = torch.FloatTensor(te_raw[i:i+BATCH_SIZE]).to(device)
            p, _ = model(bt, br)
            all_mags.append(p.cpu().numpy())
            
    test_mags = np.concatenate(all_mags)
    return model, test_mags

# ============================================================================
# Main
# ============================================================================
def main():
    if not os.path.exists(PARQUET_PATH):
        logger.error(f"Need to run weekly_resampler.py first! Could not find {PARQUET_PATH}")
        return
        
    df = pd.read_parquet(PARQUET_PATH)
    data = prepare_weekly_data(df, FEATURES)
    
    split_idx = int(len(data['targets']) * 0.8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize features BEFORE training stages
    train_flat = data['target_seqs'][:split_idx].reshape(-1, len(FEATURES))
    f_mean, f_std = train_flat.mean(0), train_flat.std(0) + 1e-8
    data['target_seqs'] = (data['target_seqs'] - f_mean) / f_std
    data['leader_raw'] = (data['leader_raw'] - f_mean) / f_std
    
    # 1. Train Direction (UP/DOWN)
    _, dir_preds, dir_probs, y_true_dir = train_stage1_direction(data, split_idx, device)
    
    # 2. Train Intensity (|Return|)
    _, mag_preds = train_stage2_intensity(data, split_idx, device)
    
    # 3. Combine: Final Prediction = Direction_Sign * Magnitude
    # If dir_pred is 1 -> +1, if 0 -> -1
    dir_signs = np.where(dir_preds == 1, 1, -1)
    final_preds = dir_signs * mag_preds
    
    y_test_actual = data['targets'][split_idx:]
    test_syms = data['symbols'][split_idx:]
    test_dates = data['dates'][split_idx:]
    
    logger.info("\n" + "="*70)
    logger.info("📊 TWO-STAGE WEEKLY RESULTS")
    logger.info("="*70)
    
    results = []
    for sym in np.unique(test_syms):
        mask = test_syms == sym
        ys = y_test_actual[mask].flatten()
        ps = final_preds[mask].flatten()
        
        if len(ys) < 5: continue
        
        acc = np.sum(np.sign(ys) == np.sign(ps)) / len(ys) * 100
        corr = np.corrcoef(ps, ys)[0, 1] if np.std(ps) > 1e-8 else 0.0
        rmse = np.sqrt(mean_squared_error(ys, ps))
        
        logger.info(f"   {sym:>15s} | Weekly Acc: {acc:5.2f}% | Corr: {corr:+.4f} | RMSE: {rmse:.4f}")
        results.append({
            'Symbol': sym, 'Accuracy_%': f"{acc:.2f}",
            'Correlation': f"{corr:.4f}", 'RMSE': f"{rmse:.4f}"
        })
        
    pd.DataFrame(results).to_csv('data/weekly_deltalag_summary.csv', index=False)
    
    # Save full predictions to compare
    df_out = pd.DataFrame({
        'symbol': test_syms,
        'date': test_dates,
        'actual_return': y_test_actual.flatten(),
        'predicted_return': final_preds.flatten(),
        'predicted_dir_prob': dir_probs.flatten(),
        'predicted_magnitude': mag_preds.flatten()
    })
    df_out.to_csv('data/weekly_deltalag_predictions.csv', index=False)
    logger.info("\n✅ Weekly Two-Stage prediction complete!")

if __name__ == "__main__":
    main()
