"""
Generate Sentiment Scores for News Articles using FinBERT
"""
import os
import argparse
import logging
from tqdm import tqdm
from pymongo import MongoClient
import pandas as pd
from transformers import pipeline

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

class SentimentGenerator:
    """Class to manage the generation of sentiment scores using FinBERT"""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.client = self._get_mongo_client()
        self.db = self.client[MONGO_CONFIG['database']]
        self.collection = self.db.news_articles
        
        logger.info("Loading FinBERT model...")
        # ProsusAI/finbert is specialized for financial text
        self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        logger.info("FinBERT model loaded.")

    def _get_mongo_client(self):
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

    def get_pending_articles(self, limit=None):
        """Get articles that haven't been analyzed yet."""
        query = {
            "$or": [
                {"analysis.sentiment_score": None},
                {"analysis.sentiment_score": {"$exists": False}}
            ]
        }
        
        cursor = self.collection.find(query)
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)

    def process_batch(self, batch):
        """Process a batch of articles and update MongoDB."""
        from pymongo import UpdateOne
        
        if not batch:
            return 0
            
        operations = []
        
        for article in batch:
            text = article.get('summary') or article.get('title') or article.get('content')
            if not text:
                continue
                
            # Truncate text to 512 tokens roughly (using 1000 chars as heuristic to avoid token limit errors)
            text_truncated = text[:1000]
            
            try:
                result = self.nlp(text_truncated)[0]
                label = result['label']
                score = result['score']
                
                # Convert to numerical score: Positive = score, Negative = -score, Neutral = 0
                numerical_score = 0.0
                if label == 'positive':
                    numerical_score = float(score)
                elif label == 'negative':
                    numerical_score = -float(score)
                    
                operations.append(
                    UpdateOne(
                        {"_id": article['_id']},
                        {
                            "$set": {
                                "analysis.sentiment_score": numerical_score,
                                "analysis.sentiment_label": label,
                                "analysis.sentiment_confidence": float(score)
                            }
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error analyzing article {article.get('_id')}: {e}")
                
        if operations:
            try:
                result = self.collection.bulk_write(operations, ordered=False)
                return result.modified_count
            except Exception as e:
                logger.error(f"Bulk write error: {e}")
                return 0
                
        return 0

    def run(self, limit=None):
        logger.info("Fetching pending articles...")
        articles = self.get_pending_articles(limit)
        
        total_articles = len(articles)
        logger.info(f"Found {total_articles} articles to process.")
        
        if total_articles == 0:
            return
            
        processed_count = 0
        
        # Process in batches
        with tqdm(total=total_articles, desc="Generating Sentiment") as pbar:
            for i in range(0, total_articles, self.batch_size):
                batch = articles[i:i + self.batch_size]
                modified = self.process_batch(batch)
                processed_count += modified
                pbar.update(len(batch))
                
        logger.info(f"Sentiment generation complete. Updated {processed_count} articles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentiment scores using FinBERT.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of articles to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    args = parser.parse_args()
    
    generator = SentimentGenerator(batch_size=args.batch_size)
    generator.run(limit=args.limit)

