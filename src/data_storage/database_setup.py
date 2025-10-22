import os
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

# --- Load Environment Variables ---
# This line MUST be at the top to load your .env file
load_dotenv()

# --- TimescaleDB Configuration ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'finlagx'),
    'user': os.getenv('DB_USER', 'finlagx'), # CORRECTED: Default is now 'finlagx'
    'password': os.getenv('DB_PASSWORD', 'finlagx_password')
}

# --- MongoDB Configuration ---
MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'username': os.getenv('MONGO_USER', 'admin'),
    'password': os.getenv('MONGO_PASSWORD', 'finlagx_mongo'),
    'database': os.getenv('MONGO_DB', 'finlagx_news')
}


def get_db_url():
    """Generate PostgreSQL connection URL"""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

def setup_timescaledb():
    """Install TimescaleDB extension and create hypertables"""
    engine = create_engine(get_db_url())
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL, symbol VARCHAR(20) NOT NULL, category VARCHAR(20) NOT NULL,
                    open_price DECIMAL(15,4), high_price DECIMAL(15,4), low_price DECIMAL(15,4),
                    close_price DECIMAL(15,4), adj_close DECIMAL(15,4), volume BIGINT,
                    PRIMARY KEY (time, symbol)
                );
            """))
            conn.execute(text("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    time TIMESTAMPTZ NOT NULL, indicator VARCHAR(50) NOT NULL, value DECIMAL(15,6),
                    PRIMARY KEY (time, indicator)
                );
            """))
            conn.execute(text("SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);"))
            conn.commit()
            print("✅ TimescaleDB schema created successfully.")
    except Exception as e:
        if "already exists" not in str(e):
             print(f"❌ Error setting up TimescaleDB: {e}")
        else:
            print("✅ TimescaleDB schema already exists.")


def get_engine():
    """Get SQLAlchemy engine for database operations"""
    return create_engine(get_db_url())

def test_connection():
    """Test all database connections."""
    print("\n--- Testing Database Connections ---")
    # Test TimescaleDB/PostgreSQL
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version.split(',')[0]}")
    except Exception as e:
        print(f"❌ PostgreSQL Connection Failed: {e}")
        raise

    # Test MongoDB
    try:
        client = MongoClient(
            host=MONGO_CONFIG['host'], port=MONGO_CONFIG['port'],
            username=MONGO_CONFIG['username'], password=MONGO_CONFIG['password'],
            authSource='admin', serverSelectionTimeoutMS=5000
        )
        client.admin.command('ismaster')
        print(f"✅ Connected to MongoDB: {client.server_info()['version']}")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        raise
    print("------------------------------------")


def clean_database_tables():
    """Truncate all data from market and macro tables."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE market_data, macro_data RESTART IDENTITY CASCADE;"))
            conn.commit()
            print("🧹 Cleaned all market and macro data from TimescaleDB.")
    except Exception as e:
        print(f"❌ Error cleaning TimescaleDB tables: {e}")

if __name__ == "__main__":
    print("🚀 Setting up FinLagX Databases...")
    # Note: Database creation is handled by docker-compose.
    # This script now only sets up tables and extensions.
    setup_timescaledb()
    test_connection()
    print("\n✅ Database setup completed!")
