"""
Model Architectures Page
Breakdown of Deep Learning models used in FinLagX for Academic Viva
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import inject_glassmorphism_css

st.set_page_config(
    page_title="Architectures - FinLagX",
    page_icon="🧠",
    layout="wide"
)

inject_glassmorphism_css()

st.markdown("<div class='main-header'>🧠 Deep Learning Architectures</div>", unsafe_allow_html=True)
st.markdown("""
<div style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>
    Technical breakdown of the proprietary sequence modeling and cross-asset attention mechanisms 
    developed for the FinLagX research project.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. TCN (Temporal Convolutional Network)")
    st.markdown("""
    <div class='glass-card'>
    The <b>TCN</b> model matches or beats standard LSTMs for financial sequence modeling by utilizing 1D causal convolutions.
    <br><br>
    <b>Key Academic Contributions:</b>
    <ul>
        <li><b>Causal Dilated Convolutions:</b> Prevents data leakage from the future while allowing the receptive field to expand exponentially (dilation factors $d = 1, 2, 4$). The receptive field $R$ is defined as $R = 1 + 2 \\times (K-1) \\times \\sum d_i$.</li>
        <li><b>Residual Connections:</b> Allows deep layers without vanishing gradients via $F(x) + x$.</li>
        <li><b>Monotonic Ranking Loss:</b> Tuned specifically for financial returns rather than standard MSE, directly optimizing for directional sorting: $L(y, \\hat{y}) = \\max(0, -y \\cdot \\hat{y} + \\epsilon)$.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 2. DeltaLag (Cross-Asset Attention)")
    st.markdown("""
    <div class='glass-card'>
    <b>DeltaLag</b> is our novel implementation of the Transformer self-attention mechanism adapted for multivariate financial time series.
    <br><br>
    <b>Key Academic Contributions:</b>
    <ul>
        <li><b>Dynamic Leader Selection:</b> Unlike static VAR models, DeltaLag dynamically attends over all $(Leader, Lag)$ combinations to identify predictive signals per target asset using scaled dot-product attention: $Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$.</li>
        <li><b>No Fixed Lag Structure:</b> Removes the necessity of a priori lag setting (which traditional econometric models require).</li>
        <li><b>Attention Aggregation:</b> Extracts top-k strongest signals across 15+ assets and 5 macro indicators to fuel the final dense layer.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### ⚙️ Stage 1 & 2: LightGBM Hurdle Model")
st.info("The Hurdle Model acts as our rigorous Machine Learning baseline, proving that Deep Learning is necessary for complex temporal tasks.")

col3, col4 = st.columns(2)
with col3:
    st.markdown("""
    <div class='glass-card'>
    <h4>Stage 1: Binary Classifier (Focal Loss)</h4>
    <ul>
        <li>Predicts the <i>direction</i> (Up/Down) of the asset.</li>
        <li><b>Focal Loss Implementation:</b> Modifies standard Cross-Entropy to handle class imbalances in highly volatile market regimes. $FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$. This down-weights easily classified normal days and focuses the model on hard-to-predict volatile days.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
with col4:
    st.markdown("""
    <div class='glass-card'>
    <h4>Stage 2: Magnitude Regressor</h4>
    <ul>
        <li>Predicts the <i>absolute magnitude</i> of the move, but <b>only</b> if Stage 1 crosses a strict confidence threshold.</li>
        <li>Prevents the model from trading in low-signal, noisy regimes.</li>
        <li><b>Optuna Tuning:</b> Hyperparameters optimized using strict <code>TimeSeriesSplit</code> cross-validation to prevent forward-looking bias.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🔄 Walk-Forward Backtesting Validation")
st.markdown("""
<div style='background-color: #1e222d; padding: 1.5rem; border-radius: 4px; border: 1px solid #2a2e39;'>
    <b>Purged Chronological Cross-Validation</b><br>
    To strictly prevent data leakage, all models (TCN, DeltaLag, LightGBM) are evaluated using a purged walk-forward methodology:
    <ol>
        <li><b>Train Split:</b> $T_0$ to $T_1$ (e.g., 2010 - 2018).</li>
        <li><b>Purge Gap:</b> Drops exactly 10 days of data between Train and Test sets to prevent overlapping rolling windows (e.g., moving averages) from bleeding future data.</li>
        <li><b>Test Split:</b> $T_2$ to $T_3$ (e.g., 2018 - 2024).</li>
    </ol>
    This guarantees that the out-of-sample directional accuracy and Sharpe Ratios reported are entirely reflective of live deployment capability.
</div>
""", unsafe_allow_html=True)
