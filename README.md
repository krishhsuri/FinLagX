<div align="center">
  <h1>🌐 FinLagX: The Financial Intelligence Engine</h1>
  <p><i>Predicting the markets by understanding how everything is connected.</i></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-Glassmorphism-FF4B4B.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/Deep%20Learning-LSTM-FF6F00.svg" alt="TensorFlow">
    <img src="https://img.shields.io/badge/NLP-FinBERT-yellow.svg" alt="HuggingFace">
  </p>
</div>

---

## 📖 For the Non-Tech Readers: What is FinLagX?

Imagine you are trying to predict if it will rain tomorrow. Instead of just looking at the sky above your house, you look at the wind blowing from the neighboring city. 

Financial markets work the exact same way. **Everything is connected.** 
If Crude Oil prices crash today, airline stocks might go up tomorrow. If the USD/CNY currency pair moves, the global stock market might react two days later. 

Most traditional software only looks at one asset's history to predict its future. **FinLagX** is different. It is an AI-powered dashboard that acts like a financial detective. It figures out the hidden cause-and-effect relationships between different global assets, reads thousands of news headlines to understand human emotion, and uses advanced AI to predict what will happen next.

---

## 🧠 For the Tech Readers: The "Who, What, and Why"

FinLagX is an end-to-end Machine Learning pipeline. But instead of leaving it as a messy Python script, we wrapped it in a highly premium, interactive UI. Here is exactly **what** is running under the hood, and **why** we chose it:

### 1. 🕸️ The Network: Granger Causality
* **Why we need it:** We need to prove mathematically that Asset A actually causes a delayed movement in Asset B.
* **What it does:** Runs millions of Vector Autoregression (VAR) statistical tests to find the "Top Leaders" of the market (e.g., proving Bitcoin leads the Nasdaq).

### 2. 🤖 The Predictor: LSTM (Long Short-Term Memory)
* **Why we need it:** Standard ML models forget the distant past. Financial markets have long-term memories.
* **What it does:** Uses recurrent neural networks with "memory gates" to learn from historical price sequences and predict future directional movements.

### 3. 📰 The Reader: FinBERT (NLP)
* **Why we need it:** Markets run on fear and greed. Numbers aren't enough; we need to understand the news.
* **What it does:** We integrated HuggingFace's `FinBERT`, an NLP model specifically trained on financial texts, to read live headlines and instantly score them as Positive, Negative, or Neutral.

### 4. 💻 The Face: Streamlit Glassmorphism UI
* **Why we need it:** A powerful AI is useless if a hedge-fund manager or examiner can't understand it.
* **What it does:** We built an ultra-premium, dark-mode "Glassmorphism" dashboard using Streamlit and Plotly. It takes complex backend matrices and turns them into beautiful, interactive, real-time charts.

---

## 🏗️ System Architecture (How Data Flows)

1. **Data Collection:** Live prices are pulled via APIs (Yahoo Finance) and News is scraped.
2. **Storage:** Structured numbers go into **PostgreSQL**. Unstructured news text goes into **MongoDB**.
3. **The AI Brain:** The data is fed into our Python ML pipelines (Granger, LSTM, FinBERT) where it is cleaned and processed.
4. **The Dashboard:** Streamlit connects to our backend, visualizes the AI's predictions, and displays the interactive network graphs to the user.

---

## 📂 Project Directory (Where is everything?)

```text
FinLagX/
├── .streamlit/                # 🎨 UI Configs: Makes the dashboard look like a VC Startup
├── data/                      # 🗄️ Raw Data & Csvs: Where the AI's final answers are saved
├── models/                    # 🧠 AI Brains: Saved checkpoints of our trained Neural Networks
├── pages/                     # 🖥️ The Screens: The 7 beautiful pages you see on the dashboard
│   ├── 1_Network_Analysis.py  # Shows the causality spider-web
│   ├── 2_FinBERT_Sentiment.py # The interactive NLP news reader
│   └── ... (other pages)
├── src/                       # ⚙️ The Engine Room: Hardcore Python scripts that train the AI
│   ├── data_storage/          # Connects to PostgreSQL and MongoDB
│   └── modeling/              # The actual Math and Deep Learning code
├── utils/                     # 🛠️ Helpers: Global CSS styling and dashboard formatting
├── app.py                     # 🚀 The Entry Point: The main homepage of the dashboard
└── run_complete_pipeline.py   # 🏭 The Factory Button: Runs everything from scratch
```

---

## 🚀 How to Install & Run Locally

If you want to run this intelligence engine on your own machine:

**Step 1: Clone the Project**
```bash
git clone https://github.com/krishhsuri/FinLagX.git
cd FinLagX
```

**Step 2: Install the Brain (Dependencies)**
```bash
pip install -r requirements.txt
```

**Step 3: Launch the Dashboard**
```bash
streamlit run app.py
```
*The app will automatically open in your browser. (If your local databases are off, the app will smartly use mock data so the presentation never crashes).*

---

## 👥 The Architects (Team)

We divided the work to ensure both the backend logic and frontend user experience were flawless:

* **Aryan Raj (ML Architect & UI Engineer):** Designed and trained the Deep Learning models (LSTM), implemented the NLP Sentiment sandbox, and engineered the entire premium Glassmorphism Streamlit interface. (Built the *Brain* and the *Face*).
* **Krish Suri (Data Engineering):** Handled the data pipelines, API integrations, and database architectures (PostgreSQL & MongoDB). (Built the *Pipes* and the *Engine Room*).

---
<div align="center">
  <small>Disclaimer: Built purely for Academic Research and Viva Presentations. Not intended for live algorithmic trading.</small>
</div>
