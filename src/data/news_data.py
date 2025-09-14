# src/data/news_data.py

import os
import yaml
import feedparser
import pandas as pd
from datetime import datetime

BASE_PATH = "data/raw/news"

def load_config(path="src/data/config_news.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fetch_rss_feed(url, limit=50):
    """Fetch RSS feed and return as list of dicts"""
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:limit]:
        articles.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
            "source": url
        })
    return articles

def save_articles(category, source_name, articles):
    if not articles:
        print(f"⚠️ No articles for {source_name}")
        return
    
    date_str = datetime.today().strftime("%Y-%m-%d")
    folder = os.path.join(BASE_PATH, category, source_name)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{source_name}_{date_str}.csv")
    pd.DataFrame(articles).to_csv(file_path, index=False)
    print(f"✅ Saved {len(articles)} articles -> {file_path}")

def download_all_news():
    config = load_config()
    print("📰 Starting News Collection Pipeline...\n")

    for category, feeds in config.items():
        print(f"📂 Category: {category}")
        for feed in feeds:
            try:
                articles = fetch_rss_feed(feed["url"])
                save_articles(category, feed["name"], articles)
            except Exception as e:
                print(f"❌ Error fetching {feed['name']}: {e}")

    print("\n✅ News pipeline finished.")
