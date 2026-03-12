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
    get_mlflow_latest_results
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="FinLagX | Intelligence Portal",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUTURISTIC GLASSY CSS ====================

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }

    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #22d3ee 0%, #818cf8 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(34, 211, 238, 0.3);
        margin-bottom: 0.5rem;
    }
    
    .sub-glitch {
        color: #94a3b8;
        font-size: 1.2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1rem;
    }

    .neon-border-cyan { border-left: 4px solid #22d3ee; }
    .neon-border-magenta { border-left: 4px solid #d946ef; }
    .neon-border-green { border-left: 4px solid #4ade80; }

    /* Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #22d3ee !important;
        font-size: 2.2rem;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

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

st.markdown("<div class='main-header'>FinLagX Intelligence Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-glitch'>Real-Time Predictive Market Mesh</div>", unsafe_allow_html=True)

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
                    <div style='font-size:1.8rem; font-weight:bold; color:#22d3ee;'>{primary_val:.4f if primary_val else 'N/A'}</div>
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
    st.metric("Total Assets", stats['total_assets'])
with col2:
    st.metric("Mean Accuracy", f"{stats['avg_accuracy']:.2f}%")
with col3:
    st.metric("Predictions Generated", f"{int(stats['total_predictions']):,}")
with col4:
    st.metric("Mean RMSE", f"{stats['avg_rmse']:.4f}")

# ==================== LEADERBOARD & RECENT FLOWs ====================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🏆 TOP PERFORMING NODES")
    all_metrics = load_all_metrics()
    if not all_metrics.empty:
        top_5 = all_metrics.nlargest(5, 'Directional_Accuracy_%')[['Asset', 'Directional_Accuracy_%', 'Correlation']]
        st.dataframe(
            top_5.style.format({'Directional_Accuracy_%': '{:.2f}%', 'Correlation': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔮 RAW PREDICTION STREAM")
    recent = get_recent_predictions(n=6)
    if not recent.empty:
        st.dataframe(
            recent[['Asset', 'Actual_Return', 'Predicted_Return', 'Correct_Prediction']].style.format({
                'Actual_Return': '{:.4f}',
                'Predicted_Return': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER ====================

st.markdown("""
<div style='text-align: center; color: #475569; padding: 4rem 0 2rem 0; font-size: 0.8rem;'>
    <p>FINLAGX NEURAL CORE v2.0 • BUILT BY ARYAN RAJ • POWERED BY GOOGLE DEEPMIND ANTIGRAVITY</p>
    <p>© 2026 QUANTUM TRADING TECHNOLOGIES</p>
</div>
""", unsafe_allow_html=True)
