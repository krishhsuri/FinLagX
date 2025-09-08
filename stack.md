# ⚙️ FinLagX Tech Stack & Architecture

---

## 1. Core Setup
- **Language** → Python 3.10+  
- **Environment** → Conda (team reproducibility)  
- **Version Control** → Git + GitHub  

---

## 2. Data Layer
- **Market Data** → `yfinance` (free OHLCV, 10+ years)  
- **News Data** → `newspaper3k`, RSS feeds, BeautifulSoup  
- **Database** → PostgreSQL with TimescaleDB (optimized for time-series)  
- **Storage Format** → Parquet (fast, compressed, scalable)  

---

## 3. Data Engineering
- **Transformations** → pandas  
- **Pipeline Orchestration** → Prefect (keeps ETL jobs reliable + trackable)  

---

## 4. Modeling
- **Traditional Models** → statsmodels (Granger causality, VAR)  
- **Deep Learning** → PyTorch (LSTM, GRU, Transformer)  
- **Sentiment Analysis (News)** → Hugging Face Transformers (FinBERT)  

---

## 5. Collaboration & Tracking
- **Experiment Tracking** → MLflow (log metrics, models, comparisons)  
- **Data/Model Versioning** → DVC (sync datasets/models across team)  
- **CI/CD** → GitHub Actions (auto linting + tests on push)  

---

## 6. Visualization & Demo
- **EDA & Charts** → matplotlib, seaborn, plotly  
- **Graph Visualization** → networkx (lead-lag dependency graph)  
- **Final Demo** → Streamlit dashboard (interactive, free hosting)  

---

## 7. Documentation & Reporting
- **README.md** → Clear project guide  
- **Architecture Diagram** → draw.io or Mermaid (in repo/docs)  
- **Jupyter Notebooks** → For experiments & analysis  
- **Final Report** → LaTeX write-up  

---

# 📊 Project Architecture

```mermaid
flowchart TD

    subgraph Sources[Data Sources]
        A1[Equity Indices - yfinance]
        A2[Commodities - yfinance]
        A3[FX & Bonds - yfinance]
        A4[Crypto - yfinance]
        A5[News - newspaper3k + RSS]
    end

    subgraph Storage[Data Storage Layer]
        B1[(TimescaleDB - PostgreSQL)]
        B2[(Parquet Files)]
    end

    subgraph Processing[Data Engineering]
        C1[Prefect ETL Pipelines]
        C2[pandas Transformations]
    end

    subgraph Modeling[Modeling Layer]
        D1[Statsmodels - VAR/Granger]
        D2[PyTorch - LSTM/GRU/Transformers]
        D3[FinBERT - Sentiment Analysis]
    end

    subgraph Tracking[Collaboration & Tracking]
        E1[MLflow - Experiments]
        E2[DVC - Data/Model Versioning]
        E3[GitHub Actions - CI/CD]
    end

    subgraph Output[Visualization & Demo]
        F1[Matplotlib/Seaborn - Charts]
        F2[NetworkX - Lead-Lag Graph]
        F3[Streamlit - Dashboard]
    end

    Sources --> Storage
    Storage --> Processing
    Processing --> Modeling
    Modeling --> Tracking
    Modeling --> Output
