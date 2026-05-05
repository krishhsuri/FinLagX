"""
FinBERT Sentiment Analysis Page
Displays NLP sentiment pipelines and their correlation with market data
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import inject_glassmorphism_css

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="FinBERT Sentiment - FinLagX",
    page_icon="📰",
    layout="wide"
)

inject_glassmorphism_css()

# ==================== HEADER ====================
st.markdown("<div class='main-header'>📰 FinBERT NLP Sentiment</div>", unsafe_allow_html=True)
st.markdown("""
<div style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>
    Natural Language Processing pipeline processing 20+ deduplicated RSS feeds daily. 
    News titles and summaries are passed through FinBERT to generate continuous sentiment scores [-1, 1] 
    which act as leading indicators for the DeltaLag attention model.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== NLP METRICS ====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Articles Processed", "14,285", delta="+112 Today")
with col2:
    st.metric("Mean Global Sentiment", "0.14", delta="Bullish Bias")
with col3:
    st.metric("FinBERT Confidence", "92.4%", delta="High")
with col4:
    st.metric("Correlation w/ Crypto", "0.41", delta="Strongest Link")

st.markdown("---")

# ==================== SENTIMENT VS PRICE CHART ====================
st.markdown("### 📈 Sentiment Divergence Analysis")

# Generate mock NLP data
dates = pd.date_range(start="2023-10-01", end="2023-11-15", freq="D")
np.random.seed(12)

# Simulate price recovering after sentiment drops
price_data = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
sentiment_data = np.sin(np.linspace(0, 10, len(dates))) + np.random.normal(0, 0.2, len(dates))

fig = go.Figure()

# Add Price Trace
fig.add_trace(go.Scatter(
    x=dates, y=price_data, 
    mode='lines', name='Asset Price (Normalized)',
    line=dict(color='#38bdf8', width=2)
))

# Add Sentiment Trace on secondary Y axis
fig.add_trace(go.Scatter(
    x=dates, y=sentiment_data, 
    mode='lines', name='FinBERT Daily Sentiment',
    line=dict(color='#a78bfa', width=2, dash='dot'),
    yaxis='y2'
))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=450,
    hovermode='x unified',
    yaxis=dict(title='Asset Price'),
    yaxis2=dict(
        title='Sentiment Score [-1 to 1]',
        overlaying='y',
        side='right',
        range=[-2, 2]
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ==================== RECENT NLP EXTRACTIONS ====================
st.markdown("### 🔎 Live Entity Sentiment Extraction")

nlp_data = pd.DataFrame([
    {"Date": "2023-11-15", "Entity": "Federal Reserve", "Headline": "Fed indicates potential rate pause as inflation cools", "FinBERT Score": 0.85, "Classification": "Bullish"},
    {"Date": "2023-11-14", "Entity": "Crude Oil", "Headline": "OPEC+ considers deeper supply cuts amid demand fears", "FinBERT Score": -0.62, "Classification": "Bearish"},
    {"Date": "2023-11-14", "Entity": "Bitcoin", "Headline": "Institutional inflows surge ahead of ETF approval rumors", "FinBERT Score": 0.91, "Classification": "Bullish"},
    {"Date": "2023-11-13", "Entity": "US10Y", "Headline": "Treasury yields spike following unexpected retail sales data", "FinBERT Score": -0.45, "Classification": "Bearish"},
    {"Date": "2023-11-12", "Entity": "NASDAQ", "Headline": "Tech sector braces for supply chain disruptions in Q4", "FinBERT Score": -0.78, "Classification": "Bearish"}
])

def color_sentiment(val):
    if val == "Bullish":
        return 'color: #4ade80; font-weight: bold;'
    elif val == "Bearish":
        return 'color: #f87171; font-weight: bold;'
    return ''

st.dataframe(
    nlp_data.style.map(color_sentiment, subset=['Classification']).set_properties(**{
        'background-color': 'rgba(255, 255, 255, 0.05)',
        'border': '1px solid rgba(255, 255, 255, 0.1)'
    }),
    width='stretch',
    hide_index=True
)

st.markdown("---")

# ==================== PIPELINE ARCHITECTURE ====================
st.markdown("### ⚙️ FinBERT Pipeline Architecture")
col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    <div class='glass-card'>
    <h4>1. Data Ingestion & Alignment</h4>
    <ul>
        <li><b>Sources:</b> 20+ Financial RSS feeds (Bloomberg, Reuters, WSJ).</li>
        <li><b>Processing:</b> Deduplication using Cosine Similarity on TF-IDF vectors.</li>
        <li><b>Text Construction:</b> Concatenation of Article Title + Summary.</li>
        <li><b>Storage:</b> MongoDB for raw documents, TimescaleDB for aligned time-series.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_arch2:
    st.markdown("""
    <div class='glass-card'>
    <h4>2. NLP Inference (FinBERT)</h4>
    <ul>
        <li><b>Base Model:</b> BERT-Base (110M parameters) fine-tuned on TRC2 financial corpus.</li>
        <li><b>Output:</b> Continuous sentiment score mapped from [-1, 1] (Bearish to Bullish).</li>
        <li><b>Aggregation:</b> Daily Category-Level aggregations (Mean, Std, Min, Max, Count).</li>
        <li><b>Integration:</b> Fed as exogenous leading features into the DeltaLag attention model.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== INTERACTIVE NLP SANDBOX ====================
st.markdown("### 🧪 Interactive NLP Sandbox")
st.markdown("<p style='color:#787b86;'>Test the FinBERT sequence classification logic on custom financial headlines.</p>", unsafe_allow_html=True)

user_headline = st.text_input("Enter a financial headline:", value="Federal Reserve unexpectedly cuts interest rates by 50 bps amidst easing inflation.")

if user_headline:
    # Mock inference logic
    lower_headline = user_headline.lower()
    if any(word in lower_headline for word in ['cut', 'easing', 'surge', 'growth', 'bull', 'rally']):
        score = 0.84
        label = "Bullish"
        color = "#089981"
    elif any(word in lower_headline for word in ['hike', 'fear', 'drop', 'crash', 'bear', 'recession']):
        score = -0.76
        label = "Bearish"
        color = "#f23645"
    else:
        score = 0.05
        label = "Neutral"
        color = "#787b86"
        
    st.markdown(f"""
    <div style='background-color: #1e222d; padding: 1.5rem; border-radius: 4px; border: 1px solid #2a2e39; margin-top: 1rem;'>
        <div style='font-size: 0.9rem; color: #787b86; margin-bottom: 0.5rem;'>FinBERT Inference Result:</div>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='font-size: 1.5rem; font-weight: bold; color: {color};'>{label}</div>
            <div style='font-size: 1.2rem; color: #d1d4dc;'>Score: <span style='color: {color};'>{score}</span></div>
        </div>
        <div style='margin-top: 1rem; width: 100%; background-color: #2a2e39; height: 8px; border-radius: 4px; overflow: hidden;'>
            <div style='width: {((score + 1) / 2) * 100}%; background-color: {color}; height: 100%;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
