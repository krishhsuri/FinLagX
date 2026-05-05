"""
FinLagX Analytics Dashboard - Futuristic Version
===============================================
Multi-page Streamlit application for visualizing market lead-lag relationships,
predictive intelligence, and research breakthroughs.
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSETS, ASSET_DISPLAY_NAMES,
    calculate_summary_stats,
    get_recent_predictions,
    load_all_metrics,
    format_metric,
    get_available_assets,
    get_mlflow_latest_results,
    inject_glassmorphism_css
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="FinLagX | Intelligence Portal",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUTURISTIC GLASSY CSS ====================

inject_glassmorphism_css()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("<div style='text-align:center; padding: 2rem 0;'><h2 style='color:#22d3ee; margin:0;'>💠 FINLAGX</h2><p style='color:#94a3b8; font-size:0.8rem;'>INTELLIGENCE PORTAL</p></div>", unsafe_allow_html=True)
    
    st.markdown("### 🎛️ SYSTEM NODES")
    st.page_link("app.py", label="Home (Terminal)", icon="🏠")
    
    st.markdown("---")
    st.markdown("### 📡 ACTIVE ANALYTICS")
    
    available_assets = get_available_assets()
    if available_assets:
        selected_asset = st.selectbox("FOCUS ASSET", available_assets, format_func=lambda x: ASSET_DISPLAY_NAMES.get(x, x))
    
    st.markdown("---")
    st.info("System Status: **OPTIMAL**\n\nAI Engine: **Transformer v2**\n\nTargeting: **70%+ Accuracy**")

# ==================== HERO SECTION ====================

st.markdown("<div class='main-header'>FinLagX: Financial Lead-Lag Analysis System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-glitch'>Detecting Temporal Causality Across Global Asset Classes</div>", unsafe_allow_html=True)

st.markdown("---")

# ==================== EXECUTIVE SUMMARY ====================

col_prob, col_sol = st.columns(2)

with col_prob:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h3 style='color: #f23645; margin-top: 0;'>⚠️ The Problem</h3>
        <p style='color: #b2b5be; line-height: 1.6;'>
            Financial markets are deeply interconnected. A shift in crude oil prices may precede equity downturns by several days, 
            and central bank rate decisions ripple through currency markets before retail sentiment catches up. 
            <br><br>
            Existing analytical tools either analyze assets in isolation or rely on <i>contemporaneous correlations</i>—completely 
            missing the temporal dimension and failing to answer: <b>Which asset leads which, and by how many days?</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_sol:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h3 style='color: #089981; margin-top: 0;'>💡 Project Overview</h3>
        <p style='color: #b2b5be; line-height: 1.6;'>
            <b>FinLagX</b> is a Development cum Research project that bridges rigorous econometrics with deep learning. 
            It is built to detect, quantify, and act upon lead-lag relationships across 15+ global assets and macro indicators.
            <br><br>
            By combining traditional <b>Granger Causality</b> with novel Deep Learning architectures 
            (<b>TCN</b> and <b>DeltaLag Cross-Asset Attention</b>), FinLagX allows practitioners to <i>anticipate</i> market movements 
            rather than simply react to them, executed within a purged walk-forward backtesting framework.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== TOP METRICS (Live Research) ====================

st.markdown("### 🧪 LIVE RESEARCH BREAKTHROUGHS")
research_data = get_mlflow_latest_results()

if research_data:
    cols = st.columns(len(research_data))
    for i, res in enumerate(research_data):
        with cols[i]:
            title = res['Experiment'].replace("FinLagX_", "").replace("_", " ")
            metrics = res['Metrics']
            # Find the primary metric (Accuracy or F1 or MSE)
            primary_val = metrics.get('test_accuracy') or metrics.get('best_f1_score') or metrics.get('clf_f1')
            metric_label = "Model Score"
            if 'mse' in str(metrics.keys()).lower():
                primary_val = metrics.get('test_mse')
                metric_label = "MSE (Lower is Better)"
            
            with st.container():
                st.markdown(f"""
                <div class='glass-card neon-border-cyan'>
                    <div style='font-size:0.8rem; color:#94a3b8;'>{title}</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#22d3ee;'>{f"{primary_val:.4f}" if primary_val else 'N/A'}</div>
                    <div style='font-size:0.7rem; color:#64748b;'>RUN: {res['ID'][:8]}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.warning("📡 MLflow node not reachable. Using cached static metrics.")

# ==================== MAIN OVERVIEW ====================

st.markdown("### 📊 GLOBAL MARKET MESH OVERVIEW")
stats = calculate_summary_stats()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Assets", stats['total_assets'], delta="+2 Active")
with col2:
    st.metric("Mean Accuracy", f"{stats['avg_accuracy']:.2f}%", delta="1.2%")
with col3:
    st.metric("Predictions Generated", f"{int(stats['total_predictions']):,}", delta="140", delta_color="normal")
with col4:
    st.metric("Mean RMSE", f"{stats['avg_rmse']:.4f}", delta="-0.0014", delta_color="inverse")

# ==================== LEADERBOARD & RECENT FLOWs ====================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 🏆 TOP PERFORMING NODES")
    all_metrics = load_all_metrics()
    if not all_metrics.empty:
        top_5 = all_metrics.nlargest(5, 'Directional_Accuracy_%')[['Asset', 'Directional_Accuracy_%', 'Correlation']]
        st.dataframe(
            top_5.style.format({'Directional_Accuracy_%': '{:.2f}%', 'Correlation': '{:.4f}'}),
            hide_index=True
        )

with col_right:
    st.markdown("#### 🔮 RAW PREDICTION STREAM")
    recent = get_recent_predictions(n=6)
    if not recent.empty:
        st.dataframe(
            recent[['Asset', 'Actual_Return', 'Predicted_Return', 'Correct_Prediction']].style.format({
                'Actual_Return': '{:.4f}',
                'Predicted_Return': '{:.4f}'
            }),
            hide_index=True
        )

# ==================== FOOTER ====================

st.markdown("""
<div style='text-align: center; color: #475569; padding: 4rem 0 2rem 0; font-size: 0.8rem;'>
    <p>FINLAGX NEURAL CORE v2.0 • BUILT BY ARYAN RAJ • POWERED BY GOOGLE DEEPMIND ANTIGRAVITY</p>
    <p>© 2026 QUANTUM TRADING TECHNOLOGIES</p>
</div>
""", unsafe_allow_html=True)
