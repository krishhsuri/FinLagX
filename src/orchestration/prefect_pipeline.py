from prefect import flow, task
from src.data_ingestion.market_data import download_all_assets
from src.data_ingestion.macro_data import download_all_macro
from src.preprocessing.market_preprocessing import MarketDataPreprocessor
from src.preprocessing.build_features import main as run_feature_build
import logging

# Set up logging for Prefect tasks
logger = logging.getLogger(__name__)

@task(name="Ingest Market and Macro Data")
def ingest_data():
    logger.info("Starting Data Ingestion...")
    download_all_assets()
    download_all_macro()
    return "Ingestion Complete"

@task(name="Preprocess Market Data")
def preprocess_data():
    logger.info("Starting Preprocessing...")
    preprocessor = MarketDataPreprocessor()
    preprocessor.run_full_preprocessing(save=True)
    return "Preprocessing Complete"

@task(name="Build Unified Features")
def build_features():
    logger.info("Building feature store...")
    # Using the existing build_features main function logic
    run_feature_build()
    return "Features Built"

@flow(name="FinLagX End-to-End Pipeline")
def finlagx_pipeline():
    logger.info("🚀 Starting FinLagX Orchestrated Pipeline...")
    
    ingest_status = ingest_data()
    preprocess_status = preprocess_data() # Wait for ingestion
    feature_status = build_features() # Wait for preprocessing
    
    logger.info(f"✅ Pipeline Completed: {ingest_status}, {preprocess_status}, {feature_status}")

if __name__ == "__main__":
    finlagx_pipeline()
