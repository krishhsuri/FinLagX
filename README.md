<div align="center">
  <h1>🌐 FinLagX</h1>
  <h3>Advanced Financial Risk Intelligence & Predictive Causality Engine</h3>
  <p>Leveraging Deep Learning (LSTM), NLP (FinBERT), and Statistical Causality (Granger Networks) to decode the financial markets.</p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00.svg" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Transformers-FinBERT-yellow.svg" alt="HuggingFace">
    <img src="https://img.shields.io/badge/Database-PostgreSQL%20%7C%20MongoDB-green.svg" alt="Database">
  </p>
</div>

---

## 📖 Executive Summary
Financial markets do not move in isolation. The movement of one asset (e.g., Crude Oil) often creates a ripple effect, leading to delayed directional movements in other assets (e.g., specific Forex pairs or Equities). **FinLagX** is an end-to-end Machine Learning pipeline and interactive intelligence dashboard designed to uncover these hidden systemic risks.

Instead of relying on basic univariate forecasting, FinLagX combines **Lead-Lag Network Graphs**, **News Sentiment Analysis**, and **Sequence Models** to generate highly accurate predictive signals for hedge fund managers, retail traders, and quantitative analysts.

---

## ✨ Core System Features

1. **High-End FinTech UI (Glassmorphism)**
   - Ultra-premium, interactive dashboard built on Streamlit with dynamic Plotly charts.
   - Dark mode mesh gradients, frosted glass metric cards, and responsive hover animations to emulate Tier-1 VC startups.

2. **Lead-Lag Network Analysis (Granger Causality)**
   - Mathematically identifies which global assets "lead" others and by how many days (optimal lag).
   - Generates interactive Network Graphs highlighting "Systemic Leaders" (e.g., Bitcoin, USD/CNY).

3. **FinBERT NLP Sentiment Engine**
   - Ingests real-time financial news headlines and computes market sentiment (Positive, Negative, Neutral).
   - Features an interactive "Sandbox" where users can test hypothetical headlines against the AI.

4. **LSTM Predictive Modeling Engine**
   - Long Short-Term Memory (LSTM) neural networks trained on historical sequences.
   - Outputs directional price movement predictions alongside Root Mean Square Error (RMSE) performance metrics.

5. **Purged Walk-Forward Backtesting**
   - Ensures no data leakage between training and testing sets, providing realistic, production-ready accuracy reports.

---

## 📂 Complete Project Structure

```text
FinLagX/
├── .streamlit/
│   └── config.toml                # Global UI Theme (High-End Dark Mode)
├── configs/
│   └── default_config.yaml        # Hyperparameters and system configs
├── data/
│   └── results/                   # Generated CSVs (Predictions, Lead-Lag scores)
├── docs/                          # Project documentation
├── models/                        # Saved LSTM/TCN model checkpoints (.h5 / .keras)
├── pages/                         # Streamlit Dashboard Pages
│   ├── 1_Network_Analysis.py
│   ├── 2_FinBERT_Sentiment.py
│   ├── 3_Model_Architectures.py
│   ├── 3_📰_News_Sentiment.py
│   ├── 4_LSTM_Predictions.py
│   ├── 5_Backtesting_Engine.py
│   ├── 6_Comparison.py
│   └── 7_Future_Scope.py
├── src/                           # Core Backend ML Logic
│   ├── data_ingestion/            # API pipelines (Yahoo Finance, News APIs)
│   ├── data_storage/              # PostgreSQL & MongoDB connection handlers
│   └── modeling/                  # ML training scripts (LSTM, Granger, FinBERT)
├── utils/
│   └── dashboard_helpers.py       # Global CSS injection, data mock fallbacks, UI logic
├── app.py                         # Main Streamlit Entry Point (Executive Overview)
├── docker-compose.yml             # Containerized DB setups (PostgreSQL, MongoDB)
├── requirements.txt               # Python dependencies
└── run_complete_pipeline.py       # Master script to run end-to-end data/ML pipeline
```

---

## 🧠 Deep Dive: The Machine Learning Architecture

### 1. Statistical Causality (Granger)
We utilize **Vector Autoregression (VAR)** and **Granger Causality tests** across a grid of 15 global assets to establish statistical significance (`p-value < 0.05`). This maps out the foundational edge-weights for our network graphs.

### 2. Time-Series Forecasting (LSTM & TCN)
The predictive engine utilizes a dual-model approach:
- **LSTM:** Excels at capturing long-term dependencies in non-stationary financial data.
- **TCN (Temporal Convolutional Networks):** Uses dilated causal convolutions to process sequential data in parallel, avoiding the vanishing gradient problem.

### 3. Natural Language Processing (FinBERT)
Using a pre-trained `ProsusAI/finbert` model from HuggingFace, the pipeline tokenizes live news articles and maps the output logits into softmax probabilities, generating a weighted confidence score for market sentiment.

---

## 🚀 Setup & Installation Guide

### Prerequisites
- Python 3.11+
- Git
- Docker (Optional, for running local databases)

### 1. Clone the Repository
```bash
git clone https://github.com/krishhsuri/FinLagX.git
cd FinLagX
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup (Optional)
If you wish to run the live data ingestion pipeline, spin up the Docker databases:
```bash
docker-compose up -d
```
*(Note: If databases are offline, the Streamlit dashboard automatically gracefully degrades to highly realistic mock data via `utils/dashboard_helpers.py`)*

### 5. Launch the Intelligence Dashboard
```bash
streamlit run app.py
```
The application will launch on `http://localhost:8501`.

---

## 🛠️ Pipeline Execution
To run the full end-to-end ML pipeline (Data Download -> Feature Engineering -> Granger Testing -> LSTM Training):
```bash
python run_complete_pipeline.py
```

---

## 👥 Contributors / Team
- **Aryan Raj** - ML Architect & Frontend UI Engineer
- **Krish Suri** - Data Engineering & Backend Pipeline
*(Add any other team members here)*

---
<div align="center">
  <small>Built for Academic Research & Real-World Financial Application.</small>
</div>
