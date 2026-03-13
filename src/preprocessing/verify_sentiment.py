import os
from pymongo import MongoClient
import pandas as pd

MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'username': os.getenv('MONGO_USER', 'admin'),
    'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
    'database': os.getenv('MONGO_DB', 'finlagx_news')
}

def check_processed_news():
    print("="*80)
    print("Checking FinBERT Processed News")
    print("="*80)
    
    try:
        client = MongoClient(
            host=MONGO_CONFIG['host'],
            port=MONGO_CONFIG['port'],
            username=MONGO_CONFIG['username'],
            password=MONGO_CONFIG['password'],
            authSource='admin'
        )
        db = client[MONGO_CONFIG['database']]
        collection = db.news_articles
        
        # Get count of total vs processed
        total = collection.count_documents({})
        processed = collection.count_documents({
            "analysis.sentiment_score": {"$ne": None}
        })
        
        print(f"Total News Records in DB: {total}")
        print(f"Records Processed by FinBERT: {processed}")
        print("-" * 80)
        
        # Fetch some processed examples
        if processed > 0:
            print("Latest 5 Processed Examples:\n")
            cursor = collection.find({
                "analysis.sentiment_score": {"$ne": None}
            }).sort("timestamp", -1).limit(5)
            
            for i, doc in enumerate(cursor):
                print(f"[{i+1}] Title: {doc.get('title')}")
                print(f"    Source: {doc.get('source', {}).get('name')}")
                analysis = doc.get('analysis', {})
                print(f"    Sentiment Label: {analysis.get('sentiment_label')} (Score: {analysis.get('sentiment_score'):.4f})")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

if __name__ == "__main__":
    check_processed_news()
