# 🌐 FinLagX: Financial Risk Intelligence Engine

FinLagX is an advanced, AI-driven financial prediction platform. It leverages **Deep Learning (LSTM)**, **Natural Language Processing (FinBERT)**, and **Statistical Causality (Granger Networks)** to uncover hidden market dynamics, analyze news sentiment, and predict future asset price movements.

## 🔥 Key Features

- **High-End FinTech UI:** An ultra-premium, interactive *Glassmorphism* dashboard built on Streamlit with dynamic Plotly charts.
- **Lead-Lag Network Analysis:** Uses Granger Causality to mathematically identify which global assets "lead" others (e.g., how Bitcoin influences Nasdaq 100).
- **FinBERT Sentiment Engine:** Real-time NLP analysis of financial news headlines to gauge market fear/greed (Positive, Negative, Neutral).
- **LSTM Predictive Modeling:** Time-series forecasting memory networks trained on historical sequences to predict directional price movements.
- **Interactive Sandbox:** Test live news headlines instantly with our integrated NLP inference UI.

## ⚙️ Tech Stack

- **Machine Learning:** `Python`, `TensorFlow/Keras`, `Scikit-Learn`, `Transformers (HuggingFace FinBERT)`
- **Data Engineering:** `PostgreSQL` (Structured Time-Series), `MongoDB` (Unstructured News Data)
- **Frontend / Dashboard:** `Streamlit`, `Plotly`, `HTML/CSS (Glassmorphism)`

## 🚀 How to Run Locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the intelligence dashboard:
   ```bash
   streamlit run app.py
   ```

*(Disclaimer: This project was built for academic research and presentation purposes. It should not be used for live financial trading without independent verification.)*
