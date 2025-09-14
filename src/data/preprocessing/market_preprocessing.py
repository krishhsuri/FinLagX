import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# Base directories
RAW_DATA_DIR = "data/raw/market"
PROCESSED_DATA_DIR = "data/processed/market"
GRAPHS_DIR = "data/processed/graphs"

def process_and_analyze_market_data():
    """
    Processes raw market data, performs EDA, and saves the cleaned data.
    """
    print("\n✨ Starting market data preprocessing & EDA...")
    if not os.path.exists(RAW_DATA_DIR):
        print(f"⚠️ Raw data directory not found at {RAW_DATA_DIR}.")
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                raw_file_path = os.path.join(root, file)
                asset_name = file.replace(".csv", "")

                try:
                    df = pd.read_csv(raw_file_path, parse_dates=True)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df.set_index('Date', inplace=True)
                    
                    print(f"\n📊 Analyzing {file}...")
                    print("--- DataFrame Info ---")
                    df.info()

                    all_possible_numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    numeric_cols = [col for col in all_possible_numeric_cols if col in df.columns]

                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    if numeric_cols:
                        print("\n📈 Statistical Summary:")
                        print(df[numeric_cols].describe())
                    else:
                        print("⚠️ No numeric columns found for statistical summary.")

                    df.ffill(inplace=True)
                    print("\n✅ Missing values handled.")
                    
                    # --- Advanced Feature Engineering ---
                    if 'Close' in df.columns:
                        # Daily Returns
                        df['Daily_Return'] = df['Close'].pct_change()
                        # Volatility (Rolling Standard Deviation of Returns)
                        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
                        print("📈 Daily Returns and Volatility calculated.")

                    # --- Outlier Detection and Handling ---
                    if 'Close' in df.columns:
                        Q1 = df['Close'].quantile(0.25)
                        Q3 = df['Close'].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]
                        print(f"\n🕵️ Found {len(outliers)} outliers based on Close Price.")
                        # You can add logic here to remove or cap the outliers if needed

                    # --- Data Visualization (Graphical Representation) ---
                    sns.set_style("whitegrid")

                    # Plot 1: Close Price & SMA
                    plt.figure(figsize=(12, 7))
                    sns.lineplot(data=df, x=df.index, y='Close', label='Close Price')
                    if 'Close' in df.columns:
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        sns.lineplot(data=df, x=df.index, y='SMA_50', label='SMA 50')
                        print("📈 SMA_50 calculated.")
                    
                    plt.title(f'{asset_name} Price and 50-Day SMA')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.savefig(os.path.join(GRAPHS_DIR, f"{asset_name}_price_sma.png"))
                    plt.show()

                    # Plot 2: Candlestick Chart
                    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                        mpf.plot(df, type='candle', style='charles', 
                                 title=f'{asset_name} Candlestick Chart',
                                 savefig=os.path.join(GRAPHS_DIR, f"{asset_name}_candlestick.png"))
                    else:
                        print(f"⚠️ Missing columns for candlestick chart.")

                    # Plot 3: Daily Returns Distribution
                    if 'Daily_Return' in df.columns:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(df['Daily_Return'].dropna(), kde=True, bins=50)
                        plt.title(f'{asset_name} Daily Returns Distribution')
                        plt.xlabel('Daily Return')
                        plt.ylabel('Frequency')
                        plt.savefig(os.path.join(GRAPHS_DIR, f"{asset_name}_returns_dist.png"))
                        plt.show()
                    
                    # Plot 4: Correlation Heatmap
                    if len(numeric_cols) > 1:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
                        plt.title(f'{asset_name} Correlation Heatmap')
                        plt.savefig(os.path.join(GRAPHS_DIR, f"{asset_name}_correlation_heatmap.png"))
                        plt.show()
                    else:
                        print("⚠️ Not enough numeric columns for a correlation heatmap.")

                    # --- Preprocessing: Normalization ---
                    if numeric_cols:
                        scaler = MinMaxScaler()
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        print("\n✅ Numeric data normalized.")

                    # --- Save the processed data ---
                    df_reset = df.reset_index()
                    relative_path = os.path.relpath(raw_file_path, RAW_DATA_DIR)
                    processed_file_path = os.path.join(PROCESSED_DATA_DIR, relative_path)
                    processed_category_dir = os.path.dirname(processed_file_path)
                    os.makedirs(processed_category_dir, exist_ok=True)
                    df_reset.to_csv(processed_file_path, index=False)
                    print(f"\n✅ Processed data saved to {processed_file_path}")

                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

    print("\n✅ Market data preprocessing & EDA finished.")

if __name__ == "__main__":
    process_and_analyze_market_data()