"""
DeltaLag: Dynamic Cross-Attention Lead-Lag Detector (FAST)
==========================================================
Optimized version: leaders use lightweight linear projection, NOT LSTM encoding.
Paper ablation confirmed: raw features > temporal embeddings.

Architecture:
  1. AssetEncoder (LSTM) encodes ONLY the target asset → query vector
  2. Linear projection on leaders' RAW features at lag times → key vectors
  3. Cross-attention selects top-k (leader, lag) pairs
  4. Attention-weighted raw features → MLP → predicted return
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DeltaLagModel(nn.Module):
    """
    DeltaLag model (lightweight version):
      - Target: LSTM encoder → query (1 LSTM per batch, cheap)
      - Leaders: linear projection on raw features → keys (NO LSTM, fast)
      - Cross-attention → top-k (leader, lag)
      - Weighted raw features → MLP → prediction
    """
    def __init__(self, feature_dim, hidden_dim=64, max_lag=10, top_k=5, num_layers=1):
        super(DeltaLagModel, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.top_k = top_k
        
        # Target encoder (LSTM — only runs once per batch)
        self.target_lstm = nn.LSTM(
            feature_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Query/key projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, hidden_dim, bias=False)  # directly from raw features!
        self.scale = hidden_dim ** 0.5
        
        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, target_seq, leader_raw_features):
        """
        Args:
            target_seq:          [B, L, F]  — target asset's lookback window
            leader_raw_features: [B, n_leaders, max_lag, F]  — raw features at lag days
        
        Returns:
            pred:          [B, 1]
            lead_lag_info: dict with attention scores and indices
        """
        batch_size = target_seq.shape[0]
        n_leaders = leader_raw_features.shape[1]
        
        # 1. Encode target → query (only 1 LSTM call per batch, FAST)
        target_hidden, _ = self.target_lstm(target_seq)     # [B, L, N]
        query = self.W_Q(target_hidden[:, -1, :])           # [B, N]
        query = query.unsqueeze(1).unsqueeze(1)             # [B, 1, 1, N]
        
        # 2. Project leader raw features → keys (linear, FAST — no LSTM!)
        # leader_raw_features: [B, n_leaders, max_lag, F]
        keys = self.W_K(leader_raw_features)                # [B, n_leaders, max_lag, N]
        
        # 3. Cross-attention scores
        attn = (query * keys).sum(dim=-1) / self.scale      # [B, n_leaders, max_lag]
        
        # Find top-k across all (leader, lag) combinations
        flat_attn = attn.view(batch_size, -1)               # [B, n_leaders * max_lag]
        k = min(self.top_k, flat_attn.shape[1])
        top_values, top_flat_idx = torch.topk(flat_attn, k, dim=1)
        
        leader_indices = top_flat_idx // self.max_lag
        lag_indices = top_flat_idx % self.max_lag
        top_k_indices = torch.stack([leader_indices, lag_indices], dim=-1)  # [B, k, 2]
        top_k_scores = F.softmax(top_values, dim=1)                       # [B, k]
        
        # 4. Gather raw features at selected (leader, lag) positions (vectorized)
        raw_flat = leader_raw_features.view(batch_size, n_leaders * self.max_lag, self.feature_dim)
        flat_idx_exp = top_flat_idx.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        selected = torch.gather(raw_flat, 1, flat_idx_exp.long())         # [B, k, F]
        
        # Weighted sum
        weighted = (selected * top_k_scores.unsqueeze(-1)).sum(dim=1)     # [B, F]
        
        # 5. Predict
        pred = self.predictor(weighted)
        
        return pred, {
            'top_k_indices': top_k_indices,
            'top_k_scores': top_k_scores,
            'full_attention': attn
        }
