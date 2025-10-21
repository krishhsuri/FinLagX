# src/preprocessing/generate_sentiment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging
from src.data_ingestion.news_data import get_articles_for_sentiment_analysis, update_sentiment_analysis

# --- Configuration ---
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16  # Process 16 articles at a time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions ---

def load_sentiment_model():
    """Loads the FinBERT tokenizer and model from Hugging Face."""
    logger.info(f"Loading FinBERT model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        logger.info("✅ Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def analyze_sentiment_batch(texts: list, tokenizer, model):
    """Analyzes a batch of texts and returns sentiment labels."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_labels = [model.config.id2label[pred.argmax().item()] for pred in predictions]
    return sentiment_labels

def run_sentiment_pipeline():
    """
    Main pipeline to fetch articles, generate sentiment, and update the database.
    """
    logger.info("🚀 Starting Sentiment Analysis Pipeline...")
    
    # 1. Load Model
    tokenizer, model = load_sentiment_model()
    
    # 2. Fetch Articles
    articles_to_process = get_articles_for_sentiment_analysis(limit=0)  # limit=0 means get all
    
    if not articles_to_process:
        logger.info("✅ No new articles to analyze. Pipeline complete.")
        return
        
    logger.info(f"📰 Found {len(articles_to_process)} articles needing sentiment analysis.")
    
    # 3. Process in Batches
    num_updated = 0
    for i in tqdm(range(0, len(articles_to_process), BATCH_SIZE), desc="Analyzing Sentiment"):
        batch = articles_to_process[i:i + BATCH_SIZE]
        
        # Prepare text for analysis (title + summary)
        texts_to_analyze = [f"{article.get('title', '')}. {article.get('summary', '')}" for article in batch]
        
        try:
            sentiments = analyze_sentiment_batch(texts_to_analyze, tokenizer, model)
            
            # Update database with new scores
            for article, sentiment in zip(batch, sentiments):
                article_id = article["_id"]
                score_mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                numerical_score = score_mapping.get(sentiment, 0.0)

                # CORRECTED: Passes the numerical_score instead of the text label
                if update_sentiment_analysis(article_id, sentiment_score=numerical_score):
                    num_updated += 1

        except Exception as e:
            logger.error(f"❌ Failed to process batch starting at index {i}: {e}")

    logger.info(f"\n🎉 Sentiment Analysis Pipeline completed. Updated {num_updated} articles.")

# --- Main Execution ---

if __name__ == "__main__":
    run_sentiment_pipeline()