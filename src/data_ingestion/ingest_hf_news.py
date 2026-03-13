import os
import time
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
import logging
import hashlib
import json
import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'username': os.getenv('MONGO_USER', 'admin'),
    'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
    'database': os.getenv('MONGO_DB', 'finlagx_news')
}

# HuggingFace Configuration
# Ensure you set the HF_TOKEN environment variable in your terminal before running
# Example: set HF_TOKEN=hf_your_token_here
HF_TOKEN = os.getenv('HF_TOKEN')


def get_mongo_client():
    """Get MongoDB client connection"""
    try:
        client = MongoClient(
            host=MONGO_CONFIG['host'],
            port=MONGO_CONFIG['port'],
            username=MONGO_CONFIG['username'],
            password=MONGO_CONFIG['password'],
            authSource='admin'
        )
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def get_news_collection():
    """Get news collection from MongoDB"""
    client = get_mongo_client()
    db = client[MONGO_CONFIG['database']]
    return db.news_articles


def generate_article_id(title, url):
    """Generate unique article ID from title and URL"""
    content = f"{title}{url}".encode('utf-8')
    return hashlib.md5(content).hexdigest()

def clean_text(text, max_length=None):
    """Clean and truncate text"""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text

def process_batch(rows, collection):
    articles = []

    for item in rows:
        try:
            # The API returns rows with a "row" key containing the actual columns
            row_data = item.get("row", {})
            
            # Extract basic fields
            date_str = row_data.get('date')
            text = row_data.get('text', '')
            extra_fields_str = row_data.get('extra_fields', '{}')
            
            # Parse extra fields (JSON)
            try:
                extra_fields = json.loads(extra_fields_str) if isinstance(extra_fields_str, str) else extra_fields_str
            except:
                extra_fields = {}
                
            if isinstance(extra_fields, float): # handle nan float values from dataframe conversions
                extra_fields = {}
            
            # Extract details from extra_fields or use defaults
            url = extra_fields.get('url', '')
            source_name = extra_fields.get('publication', extra_fields.get('source', 'Unknown'))
            author = extra_fields.get('author', '')
            dataset = extra_fields.get('dataset', 'Brianferrell787/financial-news-multisource')
            
            title = extra_fields.get('title', '')
            if not title:
                title = text[:100] + "..." if len(text) > 100 else text
            
            summary = text[:2000] if text else ""
            
            article_time = None
            if date_str:
                try:
                    article_time = pd.to_datetime(date_str)
                except:
                    pass
            if article_time is None:
               article_time = datetime.now()

            article_id = generate_article_id(title, url)

            article_doc = {
                "_id": article_id,
                "article_id": article_id,
                "timestamp": article_time,
                "title": clean_text(title, 500),
                "summary": clean_text(summary, 2000),
                "content": clean_text(text),
                "url": url,
                "source": {
                    "name": source_name,
                    "url": "",
                    "category": dataset
                },
                "metadata": {
                    "author": author,
                    "tags": [],
                    "published_parsed": None,
                    "language": "en",
                    "word_count": len(str(text).split()) if text else 0,
                    "hf_extra_fields": extra_fields 
                },
                "analysis": {
                    "sentiment_score": None,
                    "entities": [],
                    "keywords": [],
                    "relevance_scores": {}
                },
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            articles.append(article_doc)
            
        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            continue

    if not articles:
        return 0
        
    try:
        from pymongo import InsertOne
        from pymongo.errors import BulkWriteError
        
        operations = [InsertOne(doc) for doc in articles]
        result = collection.bulk_write(operations, ordered=False)
        return result.inserted_count
    except BulkWriteError as bwe:
        return bwe.details['nInserted']
    except Exception as e:
        logger.error(f"Bulk write error: {e}")
        return 0


def ingest_hf_api(dataset="Brianferrell787/financial-news-multisource", config="data", split="train", limit=1000):
    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable is not set! Please set it before running this script.")
        logger.info("Example (Windows CMD): set HF_TOKEN=your_token_here")
        logger.info("Example (PowerShell): $env:HF_TOKEN=\"your_token_here\"")
        return
        
    logger.info(f"Starting ingestion from HuggingFace API: {dataset}")
    collection = get_news_collection()
    
    # Create indexes for better performance
    collection.create_index("timestamp")
    collection.create_index("source.category")
    collection.create_index([("title", "text"), ("summary", "text")])
    
    base_url = "https://datasets-server.huggingface.co/rows"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    offset = 0
    # The API limits max length per request (usually 100)
    batch_size = 100
    total_inserted = 0
    
    # We'll stop once we've requested 'limit' items (or if the API returns < batch_size items indicating EOF)
    remaining = limit
    
    pbar = tqdm(total=limit, desc="Fetching & Inserting")
    
    while remaining > 0:
        current_length = min(batch_size, remaining)
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": current_length
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                break
                
            data = response.json()
            rows = data.get("rows", [])
            
            if not rows:
                logger.info("No more rows returned by API. Reached the end of the dataset split.")
                break
                
            inserted = process_batch(rows, collection)
            total_inserted += inserted
            
            offset += len(rows)
            remaining -= len(rows)
            
            pbar.update(len(rows))
            pbar.set_postfix({"Inserted": total_inserted})
            
            # small sleep to avoid hammering the API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error during API request: {e}")
            break
            
    pbar.close()
    logger.info(f"Ingestion complete! Total Inserted matches: {total_inserted}")


if __name__ == "__main__":
    # Test queries setup
    logger.info("Testing MongoDB Data Retrieval...")
    collection = get_news_collection()
    count_before = collection.count_documents({})
    logger.info(f"Total documents before run: {count_before}")
    
    # Ingest using the API
    # Adjust the limit based on how many records you want to quickly test
    ingest_hf_api(limit=500)
    
    count_after = collection.count_documents({})
    logger.info(f"Total documents after run: {count_after}")
    
    # Get a sample
    sample = collection.find_one({}, sort=[("timestamp", -1)])
    if sample:
        logger.info(f"Latest Sample Title: {sample['title']}")
        logger.info(f"Latest Sample Source: {sample['source']}")
