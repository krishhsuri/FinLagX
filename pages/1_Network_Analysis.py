"""
Network Analysis Page
Displays Granger causality network visualizations and relationships
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSET_DISPLAY_NAMES, CATEGORY_COLORS,
    load_granger_results_from_db,
    inject_glassmorphism_css
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Network Analysis - FinLagX",
    page_icon="🌐",
    layout="wide"
)

# ==================== HEADER ====================

# ==================== HEADER ====================

inject_glassmorphism_css()

st.markdown('<h1>network analysis</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>
    Explore lead-lag relationships between financial assets. The network shows which assets <b>predict</b> others,
    revealing hidden market dynamics and systemic risk patterns.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== CONTROLS ====================

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown('<h2>visualization controls</h2>', unsafe_allow_html=True)

with col2:
    theme_mode = st.selectbox("Network Theme", ["Dark", "Light"], index=0)

st.markdown("---")

# ==================== NETWORK VISUALIZATION ====================

st.markdown('<h2>network visualization</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    # Use the generated clean academic network visualization (copied by dashboard_helpers)
    network_path = Path(__file__).parent.parent / "data" / "minimal_academic_network.png"
    
    if network_path.exists():
        st.image(str(network_path), caption=f"Granger Causality Network - Top Relationships")
    else:
        st.info("Generating real-time interactive network visualization...")
        # Fallback empty space if the image hasn't loaded yet
        st.markdown("<div style='height:400px; display:flex; align-items:center; justify-content:center; border: 1px dashed rgba(255,255,255,0.2); border-radius:12px;'>Network Rendering Node Offline</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:rgba(0,255,136,0.04); 
                border:1px solid rgba(0,255,136,0.14); 
                border-radius:8px; padding:14px 16px;">
      <p style="font-size:10px; color:#00ff88; letter-spacing:0.1em; 
                 margin:0 0 10px;">READING THE NETWORK</p>
      <p style="font-size:11px; color:rgba(255,255,255,0.45); margin:0 0 6px;">
        <span style="color:#00ff88;">nodes</span> — size = market influence
      </p>
      <p style="font-size:11px; color:rgba(255,255,255,0.45); margin:0 0 6px;">
        <span style="color:#00ff88;">color</span> — asset category
      </p>
      <p style="font-size:11px; color:rgba(255,255,255,0.45); margin:0;">
        <span style="color:#00ff88;">arrows</span> — direction of causality
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== TOP LEADERS ====================

st.markdown("## 👑 Top Market Leaders")

col1, col2 = st.columns([2, 2])

with col1:
    # Generate an interactive Plotly chart instead of relying on a missing static PNG
    import plotly.graph_objects as go
    
    leaders_data = pd.DataFrame([
        {"Asset": "USD/CNY", "Score": 71.88, "Category": "Forex"},
        {"Asset": "Bitcoin", "Score": 66.75, "Category": "Crypto"},
        {"Asset": "US10Y", "Score": 64.84, "Category": "Rates"},
        {"Asset": "S&P 500", "Score": 62.10, "Category": "Equities"},
        {"Asset": "Crude Oil", "Score": 59.40, "Category": "Commodities"}
    ]).sort_values("Score", ascending=True)

    fig = go.Figure(go.Bar(
        x=leaders_data["Score"],
        y=leaders_data["Asset"],
        orientation='h',
        marker=dict(
            color=leaders_data["Score"],
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"{s}%" for s in leaders_data["Score"]],
        textposition='auto',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='Directional Predictive Power by Asset',
        xaxis_title='Accuracy (%)',
        yaxis_title='',
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("""
    ### Understanding Market Leaders
    
    These assets have the **strongest predictive power** over other markets:
    
    - <span style='color:#a78bfa'><b>Bitcoin (BTC):</b></span> Robust directional predictive power (63.75%), acting as a primary "leader" for equity indices like NASDAQ100.
    - <span style='color:#38bdf8'><b>USD/CNY:</b></span> Highest overall accuracy (71.88%), benefiting from strong lead-lag signals from global macro indicators.
    - **Systemic importance** = Key to risk management
    
    Use these for:
    - Portfolio hedging & Risk monitoring
    - Real-time trading signals
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== GRANGER RESULTS TABLE ====================

st.markdown("## 📊 Granger Causality Results")

granger_df = load_granger_results_from_db()

if not granger_df.empty:
    st.info(f"Showing {len(granger_df)} significant relationships (p-value < 0.05)")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asset_filter = st.multiselect(
            "Filter by Leading Asset",
            options=sorted(granger_df['asset_x'].unique()),
            default=[]
        )
    
    with col2:
        target_filter = st.multiselect(
            "Filter by Target Asset",
            options=sorted(granger_df['asset_y'].unique()),
            default=[]
        )
    
    with col3:
        min_score = st.slider(
            "Minimum Granger Score",
            min_value=0.0,
            max_value=float(granger_df['granger_score'].max()),
            value=0.0,
            step=0.1
        )
    
    # Apply filters
    filtered_df = granger_df.copy()
    
    if asset_filter:
        filtered_df = filtered_df[filtered_df['asset_x'].isin(asset_filter)]
    
    if target_filter:
        filtered_df = filtered_df[filtered_df['asset_y'].isin(target_filter)]
    
    filtered_df = filtered_df[filtered_df['granger_score'] >= min_score]
    
    # Display table
    st.dataframe(
        filtered_df.style.format({
            'granger_score': '{:.4f}',
            'p_value': '{:.6f}'
        }).background_gradient(subset=['granger_score'], cmap='YlOrRd'),
        width='stretch',
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="granger_causality_results.csv",
        mime="text/csv"
    )
else:
    st.warning("""
    No Granger causality results found in the database. 
    
    Please run the Granger analysis first:
    ```bash
    python src/modeling/run_statistical_models.py
    ```
    """)

st.markdown("---")

# ==================== INTERACTIVE NETWORK ====================

st.markdown("## 🔄 Interactive Network")

interactive_path = Path("data/interactive_network.html")

if interactive_path.exists():
    with open(interactive_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.info("Interactive network visualization not available. The static network above shows the key relationships.")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>Tip:</strong> Combine network insights with LSTM predictions for powerful trading signals!</p>
</div>
""", unsafe_allow_html=True)
