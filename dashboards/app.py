import streamlit as st
from datetime import date
import plotly.express as px
import pandas as pd

# --- THEME & SIDEBAR ---
st.set_page_config(
    page_title="FinLagX Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.title("FinLagX Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "📈 Market Data", "📉 Macro Data", "📰 News", "🤖 Models"],
    format_func=lambda x: x
)

# --- HEADER ---
st.title("FinLagX Financial Analytics Platform")
st.markdown("---")

# --- OVERVIEW PAGE ---
if page == "📊 Overview":
    st.header("📊 Dashboard Overview")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Top Performing Asset", "Gold +2.3%", "+0.5%")
    kpi2.metric("Most Volatile", "BTC 8.2%", "+1.1%")
    kpi3.metric("News Sentiment", "Neutral", "")

    st.markdown("---")
    st.subheader("Market Snapshot Heatmap")
    st.info("[Heatmap: Color-coded grid showing daily % change of assets (green = up, red = down).]")
    st.caption("This gives a one-glance view of which markets are bullish/bearish today.")

    st.subheader("Top Movers & Sentiment Ticker")
    st.info("[Scrolling list of most volatile assets + news sentiment score placeholder]")

    st.subheader("Global Clock + Economic Calendar")
    st.info("[Upcoming events: Fed meeting, CPI release, etc.]")

    st.markdown("---")
    st.subheader("Today's Market Narrative")
    st.success("Auto-generated news sentiment summary will appear here.")

# --- MARKET DATA PAGE ---
elif page == "📈 Market Data":
    st.header("📈 Asset Analytics")
    asset_class = st.sidebar.selectbox("Asset Class", ["Equities", "Forex", "Commodities", "Crypto"])
    asset = st.sidebar.selectbox("Asset", ["SP500", "Gold", "BTC", "EURUSD", "DAX40"])  # Example assets
    date_range = st.sidebar.date_input("Date Range", [date(2025, 1, 1), date.today()])

    st.markdown("---")
    tabs = st.tabs(["Charts", "Correlations", "Signals", "Lead-Lag Explorer", "Volatility Dashboard"])
    with tabs[0]:
        st.subheader(f"{asset} Interactive Candlestick Chart")
        st.info("[Candlestick/line chart with overlays: moving averages, Bollinger Bands]")
    with tabs[1]:
        st.subheader("Correlation Heatmap")
        st.info("[Correlation heatmap placeholder]")
    with tabs[2]:
        st.subheader("Trading Signals Dashboard")
        st.info("[Traffic-light style indicators: Buy = green, Hold = yellow, Sell = red]")
    with tabs[3]:
        st.subheader("Lead-Lag Explorer")
        st.info("[Correlation heatmap + time-lag slider to show which asset leads another]")
        st.caption("We can see how asset A tends to move before asset B, supporting our lead-lag hypothesis.")
    with tabs[4]:
        st.subheader("Volatility Dashboard")
        st.info("[Rolling volatility & correlation vs. VIX placeholder]")

# --- MACRO DATA PAGE ---
elif page == "📉 Macro Data":
    st.header("📉 Economic Insights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("GDP Growth", "3.1%", "+0.2%")
    k2.metric("CPI", "2.7%", "+0.1%")
    k3.metric("Unemployment", "4.2%", "-0.1%")
    k4.metric("Fed Rate", "5.25%", "")
    st.markdown("---")
    st.subheader("Global Macro Trends with Recession Bands")
    st.info("[Time-series chart with shaded regions when GDP < 0]")
    st.subheader("Global Macro Map")
    st.info("[Choropleth world map showing inflation or rates by country]")
    st.caption("This helps us connect financial markets to real-world macroeconomics visually.")

# --- NEWS PAGE ---
elif page == "📰 News":
    st.header("📰 Sentiment & Narratives")
    sentiment = st.selectbox("Sentiment Filter", ["All", "Positive", "Negative", "Neutral"])
    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Live News Feed by Asset")
        st.info("[Select asset to see recent news headlines filtered + sentiment color tag (green/red)]")
    with col2:
        st.subheader("Sentiment Over Time")
        st.info("[Line chart of daily sentiment scores vs. asset price]")
    st.subheader("Word Cloud: Trending Financial Terms")
    st.info("[Word cloud: inflation, earnings, AI, etc.]")
    st.markdown("---")
    st.subheader("Event Tracker")
    st.info("[Event tagging: Fed, Oil, Crypto regulation, etc.]")

# --- MODELS PAGE ---
elif page == "🤖 Models":
    st.header("🤖 Predictions & Backtests")
    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Predicted vs. Actual Prices")
        st.info("[Line chart showing model forecast vs. real data]")
    with col2:
        st.subheader("Backtest Results")
        st.info("[Sharpe ratio, accuracy, confusion matrix for signals]")
    st.markdown("---")
    st.subheader("Signal Dashboard")
    st.info("[Traffic-light style indicators: Buy = green, Hold = yellow, Sell = red]")
    st.markdown("---")
    st.subheader("Comparison Tabs")
    st.info("[Tabs: Classical Models vs. Deep Learning]")
    st.subheader("Replay Feature")
    st.info("[Pick a past crisis (2008, 2020 COVID crash) and replay market + sentiment dynamics]")
    st.subheader("Explainability Widget")
    st.info("[SHAP or feature importance bars to show what drove predictions]")
    st.subheader("Download Buttons")
    st.info("[Export CSV/Excel of processed data or results]")