-- Initialize TimescaleDB extension and create schema
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- RAW DATA TABLES (Source data from APIs)
-- ============================================================================

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    category VARCHAR(20) NOT NULL,
    open_price DECIMAL(15,4),
    high_price DECIMAL(15,4),
    low_price DECIMAL(15,4),
    close_price DECIMAL(15,4),
    adj_close DECIMAL(15,4),
    volume BIGINT,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_category ON market_data (category, time DESC);

-- Macro data table
CREATE TABLE IF NOT EXISTS macro_data (
    time TIMESTAMPTZ NOT NULL,
    indicator VARCHAR(50) NOT NULL,
    value DECIMAL(15,6),
    PRIMARY KEY (time, indicator)
);

SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_data (indicator, time DESC);

-- ============================================================================
-- PROCESSED FEATURES TABLES (For ML models)
-- ============================================================================

-- Processed market features (returns, volatility, technical indicators)
CREATE TABLE IF NOT EXISTS market_features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    returns NUMERIC,
    return_5d NUMERIC,
    return_10d NUMERIC,
    volatility_20 NUMERIC,
    sma_20 NUMERIC,
    sma_50 NUMERIC,
    volume_change NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_market_features_symbol ON market_features(symbol, time DESC);

-- Granger causality results
CREATE TABLE IF NOT EXISTS granger_results (
    id SERIAL PRIMARY KEY,
    computed_date DATE NOT NULL,
    asset_x VARCHAR(20) NOT NULL,
    asset_y VARCHAR(20) NOT NULL,
    optimal_lag INT NOT NULL,
    p_value NUMERIC,
    f_statistic NUMERIC,
    granger_score NUMERIC,
    is_significant BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_granger_date ON granger_results(computed_date DESC);
CREATE INDEX IF NOT EXISTS idx_granger_assets ON granger_results(asset_x, asset_y);
CREATE INDEX IF NOT EXISTS idx_granger_significant ON granger_results(is_significant, computed_date DESC);

-- VAR model features
CREATE TABLE IF NOT EXISTS var_features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    var_fitted_value NUMERIC,
    var_residual NUMERIC,
    impulse_response NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('var_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_var_features_symbol ON var_features(symbol, time DESC);

-- LSTM/Deep Learning predictions
CREATE TABLE IF NOT EXISTS lstm_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    predicted_return NUMERIC,
    confidence NUMERIC,
    lead_lag_indicator NUMERIC,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol, model_version)
);

SELECT create_hypertable('lstm_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_lstm_symbol ON lstm_predictions(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_lstm_version ON lstm_predictions(model_version, time DESC);

-- ============================================================================
-- VIEWS FOR EASY ACCESS
-- ============================================================================

-- Latest market prices
CREATE OR REPLACE VIEW latest_market_prices AS
SELECT DISTINCT ON (symbol)
    symbol,
    category,
    time,
    close_price,
    volume
FROM market_data
ORDER BY symbol, time DESC;

-- Latest features with all data sources
CREATE OR REPLACE VIEW latest_features AS
SELECT 
    mf.time,
    mf.symbol,
    mf.returns,
    mf.volatility_20,
    mf.sma_20,
    mf.volume_change,
    vf.var_fitted_value,
    vf.var_residual,
    vf.impulse_response
FROM market_features mf
LEFT JOIN var_features vf ON mf.time = vf.time AND mf.symbol = vf.symbol
WHERE mf.time >= NOW() - INTERVAL '30 days';

-- Granger network summary (latest significant relationships)
CREATE OR REPLACE VIEW granger_network AS
SELECT 
    asset_x,
    asset_y,
    granger_score,
    p_value,
    optimal_lag
FROM granger_results
WHERE computed_date = (SELECT MAX(computed_date) FROM granger_results)
AND is_significant = TRUE
ORDER BY granger_score DESC;

-- ============================================================================
-- MLFLOW DATABASE (Separate from main data)
-- ============================================================================

CREATE DATABASE mlflow;

-- Connect to mlflow database and create necessary extensions
\c mlflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

\c finlagx;

INSERT INTO market_data (time, symbol, category, close_price, volume) 
VALUES (NOW(), 'SP500', 'equities', 4500.00, 1000000)
ON CONFLICT DO NOTHING;

INSERT INTO macro_data (time, indicator, value)
VALUES (NOW(), 'CPI', 3.2)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON DATABASE finlagx IS 'FinLagX - Raw market and macro data + processed features for ML';
COMMENT ON DATABASE mlflow IS 'MLflow experiment tracking database';

COMMENT ON TABLE market_data IS 'Raw OHLCV market data from yfinance';
COMMENT ON TABLE macro_data IS 'Raw economic indicators from FRED';
COMMENT ON TABLE market_features IS 'Processed market features for ML models';
COMMENT ON TABLE granger_results IS 'Statistical causality analysis results';
COMMENT ON TABLE var_features IS 'Vector Autoregression model outputs';
COMMENT ON TABLE lstm_predictions IS 'Deep learning model predictions';

-- Show created hypertables
SELECT hypertable_name, hypertable_schema 
FROM timescaledb_information.hypertables 
WHERE hypertable_schema = 'public';