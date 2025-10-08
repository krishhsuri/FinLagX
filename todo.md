# FinLagX Project Plan

**Last Updated:** October 8, 2025

---

## 🎯 Current Phase: Feature Engineering & Dashboard

The project has successfully completed the initial data ingestion and modeling phases. The immediate focus is on enriching the data with sentiment analysis and presenting the findings in a user-friendly dashboard.

---

## ✅ Completed Milestones

-   [x] **Setup & Infrastructure**: Docker environment with TimescaleDB and MongoDB is fully operational.
-   [x] **Data Ingestion Pipeline**: Automated scripts for market, macro, and news data collection are in place.
-   [x] **Data Preprocessing**: Initial data cleaning and alignment pipelines are functional.
-   [x] **Statistical Modeling**: Implemented `statsmodels` for Granger Causality and VAR analysis.
-   [x] **Deep Learning Modeling**: Implemented `PyTorch` LSTM for time-series forecasting.
-   [x] **Experiment Tracking**: Integrated `MLflow` for tracking and visualizing model performance.
-   [x] **Version Control**: Modeling work has been pushed to the `modelingdev` branch on GitHub.

---

## ⏳ Next Steps & To-Do

### 1. **Implement FinBERT for Sentiment Analysis**
    - **Priority**: High
    - **Task**: Create a new script (`src/preprocessing/generate_sentiment.py`) to:
        - Fetch news articles from MongoDB that have not been analyzed.
        - Use a pre-trained `FinBERT` model from Hugging Face to generate sentiment scores.
        - Update the MongoDB articles with their sentiment scores.
    - **Goal**: Add sentiment as a core feature for the modeling pipelines.

### 2. **Integrate Sentiment into Models**
    - **Priority**: Medium
    - **Task**: Update the preprocessing and modeling scripts to include the new sentiment scores as an input feature.
    - **Goal**: Re-run the `statsmodels` and `PyTorch` models to see if sentiment improves predictive accuracy.

### 3. **Develop the Streamlit Dashboard**
    - **Priority**: High
    - **Task**: Begin development of the `Streamlit` dashboard to visualize project outputs.
    - **Initial Features**:
        - Display the lead-lag dependency graph from the `statsmodels` analysis.
        - Show the `PyTorch` model's predictions vs. actuals chart.
        - Create interactive charts for exploring the raw market and macro data.

---

##  backlog & Future Enhancements

-   **Solve News Data Limitation**: The RSS feeds provide limited historical data (~2 years).
    - **Option A**: Integrate a large, pre-existing news dataset from **Kaggle**.
    - **Option B**: Develop a more robust web scraper using `BeautifulSoup` to manually collect historical news archives.
-   **Refine Preprocessing**:
    - Implement more advanced feature engineering (e.g., volatility metrics, technical indicators).
    - Explore clustering techniques to group similar assets.
-   **Enhance Modeling**:
    - Experiment with more advanced `PyTorch` models like **Transformers**.
    - Tune hyperparameters for the LSTM model using MLflow's tracking capabilities.
-   **Orchestration**: Re-evaluate integrating **Prefect** for more robust ETL pipeline orchestration.