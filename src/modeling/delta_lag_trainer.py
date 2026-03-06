"""
DeltaLag Trainer — End-to-End Joint Multi-Asset Training (FAST)
================================================================
Optimized for CPU training:
  - NO leader_seqs (eliminated 2GB tensor + 16x LSTM calls)
  - Only target_seqs + leader_raw_features needed
  - Batch-level GPU/CPU transfer
  - Target normalization  
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

from src.feature_store.feature_store import FeatureStore
from src.modeling.delta_lag_detector import DeltaLagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameters
# ============================================================================

LOOKBACK = 30
MAX_LAG = 10
TOP_K = 3
HIDDEN_DIM = 64
EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 256       # Large batch since no leader LSTM anymore
LEARNING_RATE = 0.0005
PARQUET_PATH = 'data/processed/market/aligned_market_data.parquet'
PARQUET_PATH_FALLBACK = 'data/processed/market/aligned_dataset.parquet'

FEATURES = [
    'returns', 'pct_returns', 'return_5d', 'return_10d', 'intraday_range',
    'sma_5', 'sma_20', 'sma_50',
    'ema_5', 'ema_20',
    'volatility_5', 'volatility_20',
    'volume_change',
    'rsi_14',
    'returns_lag_1', 'returns_lag_2', 'returns_lag_5',
]


# ============================================================================
# Ranking Loss
# ============================================================================

class MonotonicRankingLoss(nn.Module):
    def __init__(self, alpha=0.5, n_pairs=64):
        super().__init__()
        self.alpha = alpha
        self.n_pairs = n_pairs
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        bs = pred.shape[0]
        mse = self.mse(pred, target)
        if bs < 2:
            return mse
        
        n = min(self.n_pairs, bs * (bs - 1) // 2)
        i = torch.randint(0, bs, (n,), device=pred.device)
        j = torch.randint(0, bs, (n,), device=pred.device)
        mask = i != j
        i, j = i[mask], j[mask]
        if len(i) == 0:
            return mse
        
        agreement = torch.tanh(pred[i] - pred[j]) * torch.sign(target[i] - target[j])
        rank = torch.mean(torch.log1p(torch.exp(-agreement)))
        return self.alpha * mse + (1 - self.alpha) * rank


# ============================================================================
# Data Preparation (NO leader_seqs — saves ~2GB RAM)
# ============================================================================

def load_data():
    for path in [PARQUET_PATH, PARQUET_PATH_FALLBACK]:
        if os.path.exists(path):
            logger.info(f"📂 Loading data from {path}...")
            df = pd.read_parquet(path)
            logger.info(f"   ✅ {len(df)} rows, {df['symbol'].nunique()} symbols")
            return df
    logger.error("❌ No parquet found!")
    return None


def prepare_multi_asset_data(df, feature_cols, lookback=LOOKBACK, max_lag=MAX_LAG):
    """Build tensors — only target_seqs and leader_raw_features (no leader_seqs!)."""
    logger.info("🔄 Building multi-asset training tensors...")
    t0 = time.time()
    
    symbols = sorted(df['symbol'].unique())
    n_symbols = len(symbols)
    feature_cols_available = [c for c in feature_cols if c in df.columns]
    n_features = len(feature_cols_available)
    n_leaders = n_symbols - 1
    
    logger.info(f"   Symbols ({n_symbols}): {symbols}")
    logger.info(f"   Features: {n_features}")
    
    # Build per-symbol feature matrices
    sym_features = {}
    sym_returns = {}
    sym_times = {}
    
    for sym in symbols:
        sdf = df[df['symbol'] == sym].sort_values('time').reset_index(drop=True)
        feats = sdf[feature_cols_available].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        sym_features[sym] = feats
        sym_returns[sym] = sdf['returns'].values.astype(np.float32)
        sym_times[sym] = sdf['time'].values
    
    # Trim to common length
    min_len = min(len(sym_features[s]) for s in symbols)
    for sym in symbols:
        off = len(sym_features[sym]) - min_len
        sym_features[sym] = sym_features[sym][off:]
        sym_returns[sym] = sym_returns[sym][off:]
        sym_times[sym] = sym_times[sym][off:]
    
    logger.info(f"   Common time steps: {min_len}")
    
    # Stack into [n_symbols, T, F]
    all_features = np.stack([sym_features[s] for s in symbols])
    all_returns = np.stack([sym_returns[s] for s in symbols])
    times = sym_times[symbols[0]]
    
    n_samples_per_sym = min_len - lookback
    total = n_symbols * n_samples_per_sym
    
    logger.info(f"   Building {total} samples...")
    
    # Pre-allocate
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
    
    elapsed = time.time() - t0
    logger.info(f"   ✅ Built {total} samples in {elapsed:.1f}s")
    logger.info(f"      target_seqs: {target_seqs.shape} ({target_seqs.nbytes / 1e6:.0f}MB)")
    logger.info(f"      leader_raw:  {leader_raw.shape} ({leader_raw.nbytes / 1e6:.0f}MB)")
    
    return {
        'target_seqs': target_seqs,
        'leader_raw':  leader_raw,
        'targets':     targets,
        'symbols':     np.array(sample_syms),
        'dates':       np.array(sample_dates),
    }


# ============================================================================
# Training
# ============================================================================

def train_deltalag(data, epochs=EPOCHS, patience=PATIENCE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🏋️ Training DeltaLag (max {epochs} epochs, patience {patience})...")
    logger.info(f"   Device: {device}")
    
    n_features = data['target_seqs'].shape[-1]
    n_leaders = data['leader_raw'].shape[1]
    n_samples = len(data['targets'])
    
    split_idx = int(n_samples * 0.8)
    
    # Feature normalization (train-only stats)
    train_flat = data['target_seqs'][:split_idx].reshape(-1, n_features)
    f_mean, f_std = train_flat.mean(0), train_flat.std(0) + 1e-8
    
    tgt_n = (data['target_seqs'] - f_mean) / f_std
    raw_n = (data['leader_raw'] - f_mean) / f_std
    
    # Target normalization
    y_mean = data['targets'][:split_idx].mean()
    y_std  = data['targets'][:split_idx].std() + 1e-8
    y_n = (data['targets'] - y_mean) / y_std
    
    # Split (stay on CPU)
    tr_tgt, te_tgt = tgt_n[:split_idx], tgt_n[split_idx:]
    tr_raw, te_raw = raw_n[:split_idx], raw_n[split_idx:]
    tr_y,   te_y   = y_n[:split_idx],   y_n[split_idx:]
    
    test_syms  = data['symbols'][split_idx:]
    test_dates = data['dates'][split_idx:]
    
    n_train = len(tr_y)
    val_size = max(1, int(n_train * 0.1))
    train_size = n_train - val_size
    
    logger.info(f"   Train: {train_size} | Val: {val_size} | Test: {len(te_y)} | Features: {n_features} | Leaders: {n_leaders}")
    
    model = DeltaLagModel(
        feature_dim=n_features,
        hidden_dim=HIDDEN_DIM,
        max_lag=MAX_LAG,
        top_k=min(TOP_K, n_leaders * MAX_LAG),
        num_layers=1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model parameters: {n_params:,}")
    
    criterion = MonotonicRankingLoss(alpha=0.5, n_pairs=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    n_batches_per_epoch = (train_size + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"   Batches per epoch: {n_batches_per_epoch}")
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0
        nb = 0
        
        perm = np.random.permutation(train_size)
        
        for i in range(0, train_size, BATCH_SIZE):
            bi = perm[i:i + BATCH_SIZE]
            
            bt = torch.FloatTensor(tr_tgt[bi]).to(device)
            br = torch.FloatTensor(tr_raw[bi]).to(device)
            by = torch.FloatTensor(tr_y[bi]).to(device)
            
            optimizer.zero_grad()
            pred, _ = model(bt, br)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            nb += 1
        
        avg = epoch_loss / max(nb, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            vt = torch.FloatTensor(tr_tgt[-val_size:]).to(device)
            vr = torch.FloatTensor(tr_raw[-val_size:]).to(device)
            vy = torch.FloatTensor(tr_y[-val_size:]).to(device)
            vp, _ = model(vt, vr)
            vl = criterion(vp, vy).item()
        
        scheduler.step(vl)
        dt = time.time() - t0
        
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"   Epoch {epoch+1:3d}/{epochs} | Train: {avg:.6f} | Val: {vl:.6f} | LR: {lr:.6f} | {dt:.1f}s")
        
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test predictions
    model.eval()
    preds_list = []
    info_list = []
    
    with torch.no_grad():
        for i in range(0, len(te_y), BATCH_SIZE):
            bt = torch.FloatTensor(te_tgt[i:i+BATCH_SIZE]).to(device)
            br = torch.FloatTensor(te_raw[i:i+BATCH_SIZE]).to(device)
            p, info = model(bt, br)
            preds_list.append(p.cpu().numpy())
            info_list.append({
                'top_k_indices': info['top_k_indices'].cpu().numpy(),
                'top_k_scores': info['top_k_scores'].cpu().numpy()
            })
    
    # De-normalize
    preds = np.concatenate(preds_list) * y_std + y_mean
    y_test = data['targets'][split_idx:]
    
    return model, preds, y_test, test_syms, test_dates, info_list


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(y_true, predictions, symbols):
    from sklearn.metrics import mean_squared_error
    
    results = {}
    for sym in np.unique(symbols):
        mask = symbols == sym
        ys = y_true[mask].flatten()
        ps = predictions[mask].flatten()
        if len(ys) < 5:
            continue
        
        acc = np.sum(np.sign(ps) == np.sign(ys)) / len(ys) * 100
        corr = np.corrcoef(ps, ys)[0, 1] if np.std(ps) > 1e-10 else 0.0
        rmse = np.sqrt(mean_squared_error(ys, ps))
        
        results[sym] = {'Accuracy_%': acc, 'Correlation': corr, 'RMSE': rmse, 'Samples': len(ys)}
        logger.info(f"   {sym:>15s} | Acc: {acc:5.2f}% | Corr: {corr:+.4f} | RMSE: {rmse:.6f}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("🚀 DeltaLag — Dynamic Cross-Attention Lead-Lag Detection")
    logger.info("="*70)
    
    df = load_data()
    if df is None:
        return
    
    data = prepare_multi_asset_data(df, FEATURES, lookback=LOOKBACK, max_lag=MAX_LAG)
    if data is None:
        return
    
    model, preds, y_test, syms, dates, info = train_deltalag(data)
    
    logger.info("\n" + "="*70)
    logger.info("📊 RESULTS")
    logger.info("="*70)
    results = calculate_metrics(y_test, preds, syms)
    
    os.makedirs('data', exist_ok=True)
    pd.DataFrame({
        'symbol': syms, 'date': dates,
        'actual': y_test.flatten(), 'predicted': preds.flatten()
    }).to_csv('data/deltalag_predictions.csv', index=False)
    
    rows = [{'Symbol': s, **{k: (f"{v:.2f}" if 'Acc' in k else f"{v:.4f}") for k, v in m.items()}} for s, m in results.items()]
    pd.DataFrame(rows).to_csv('data/deltalag_summary.csv', index=False)
    
    logger.info("\n✅ DeltaLag training complete!")


if __name__ == "__main__":
    main()
