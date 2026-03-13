import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News & Financial Sentiment")
st.markdown("""
View the latest ingested news articles and their sentiment scores generated via **FinBERT**.
""")

# MongoDB Configuration
MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'username': os.getenv('MONGO_USER', 'admin'),
    'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
    'database': os.getenv('MONGO_DB', 'finlagx_news')
}

@st.cache_resource
def get_mongo_client():
    return MongoClient(
        host=MONGO_CONFIG['host'],
        port=MONGO_CONFIG['port'],
        username=MONGO_CONFIG['username'],
        password=MONGO_CONFIG['password'],
        authSource='admin'
    )

def fetch_sentiment_data(limit=1000):
    client = get_mongo_client()
    db = client[MONGO_CONFIG['database']]
    collection = db.news_articles
    
    # Retrieve only articles with sentiment
    cursor = collection.find({
        "analysis.sentiment_score": {"$ne": None}
    }).sort("timestamp", -1).limit(limit)
    
    data = []
    for doc in cursor:
        analysis = doc.get("analysis", {})
        source_info = doc.get("source", {})
        
        data.append({
            "Timestamp": doc.get("timestamp"),
            "Title": doc.get("title"),
            "Source": source_info.get("name", "Unknown"),
            "Category": source_info.get("category", "Unknown"),
            "Sentiment": analysis.get("sentiment_label", "neutral").capitalize(),
            "Score": analysis.get("sentiment_score", 0.0),
            "Confidence": analysis.get("sentiment_confidence", 0.0)
        })
        
    return pd.DataFrame(data)

# Fetch Data
st.info("🔌 Connecting to MongoDB...")
df = fetch_sentiment_data()

if df.empty:
    st.warning("No sentiment data found in the database. Please run the ingestion and FinBERT pipelines first.")
else:
    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles Processed", len(df))
    col2.metric("Positive Articles", len(df[df['Sentiment'] == 'Positive']))
    col3.metric("Negative Articles", len(df[df['Sentiment'] == 'Negative']))
    col4.metric("Neutral Articles", len(df[df['Sentiment'] == 'Neutral']))
    
    st.markdown("---")
    
    # --- Charts ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sentiment Distribution (Current View)")
        fig_pie = px.pie(
            df, 
            names='Sentiment', 
            color='Sentiment',
            color_discrete_map={
                "Positive": "#28a745", 
                "Negative": "#dc3545", 
                "Neutral": "#6c757d"
            },
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Sentiment Breakdown by Source")
        # Top 10 sources
        top_sources = df['Source'].value_counts().nlargest(10).index
        df_top = df[df['Source'].isin(top_sources)]
        
        fig_bar = px.histogram(
            df_top, 
            y="Source", 
            color="Sentiment",
            barmode="group",
            orientation="h",
            color_discrete_map={
                "Positive": "#28a745", 
                "Negative": "#dc3545", 
                "Neutral": "#6c757d"
            }
        ).update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # --- Data Table ---
    st.subheader("📰 Latest Sentiment Articles")
    
    # Filtering options
    with st.expander("Filter Data", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            sources_list = ["All"] + list(df['Source'].unique())
            selected_source = st.selectbox("Filter by Source", sources_list)
        with f2:
            sentiment_list = ["All", "Positive", "Negative", "Neutral"]
            selected_sentiment = st.selectbox("Filter by Sentiment", sentiment_list)
            
    # Apply filters
    filtered_df = df.copy()
    if selected_source != "All":
        filtered_df = filtered_df[filtered_df['Source'] == selected_source]
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
    
    # Display table with formatting
    def color_sentiment(val):
        color = '#28a745' if val == 'Positive' else '#dc3545' if val == 'Negative' else '#6c757d'
        return f'color: {color}; font-weight: bold;'
        
    def highlight_score(val):
        color = f'rgba(40, 167, 69, {abs(val)})' if val > 0 else f'rgba(220, 53, 69, {abs(val)})' if val < 0 else 'transparent'
        return f'background-color: {color}'
        
    styled_df = filtered_df[['Timestamp', 'Source', 'Title', 'Sentiment', 'Score', 'Confidence']].style.map(
        color_sentiment, subset=['Sentiment']
    ).map(
        highlight_score, subset=['Score']
    ).format(
        {"Score": "{:.4f}", "Confidence": "{:.4f}", "Timestamp": lambda t: t.strftime("%Y-%m-%d %H:%M")}
    )
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
