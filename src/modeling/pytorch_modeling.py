# src/modeling/pytorch_modeling.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
MLFLOW_EXPERIMENT_NAME = "FinLagX_LSTM_Forecasting"
TARGET_VARIABLE = "S&P 500"  # The asset we want to predict
SEQ_LENGTH = 10  # Look-back window (in days)
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# --- Data Preparation ---

def prepare_data_for_lstm(file_path: str, target_symbol: str, seq_length: int):
    """Loads, scales, and creates sequences for the LSTM model using sentiment features."""
    print("🔄 Preparing data for LSTM...")
    df = pd.read_parquet(file_path)
    
    # Filter by symbol
    df = df[df['symbol'] == target_symbol].copy()
    
    # Select features (returns + sentiment)
    sentiment_cols = [c for c in df.columns if 'sentiment' in c or 'news_count' in c]
    feature_cols = ['returns'] + sentiment_cols
    
    # Drop rows with NaN
    df = df.dropna(subset=feature_cols)
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    data = df[feature_cols].values.astype(float)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    # Target is the first column (returns) at the next time step
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length][0]) # [0] is the return
        
    train_size = int(len(X) * 0.8)
    X_train, X_test = torch.FloatTensor(np.array(X[:train_size])), torch.FloatTensor(np.array(X[train_size:]))
    y_train, y_test = torch.FloatTensor(np.array(y[:train_size]).reshape(-1, 1)), torch.FloatTensor(np.array(y[train_size:]).reshape(-1, 1))
    
    print(f"✅ Data preparation complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, y_train, X_test, y_test, scaler, len(feature_cols)

# --- PyTorch Model Definition ---

class LSTMModel(nn.Module):
    """A simple LSTM model for time-series forecasting."""
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# --- Training and Evaluation ---

def main():
    """Main function to run the PyTorch modeling pipeline."""
    print("🚀 Starting FinLagX PyTorch Modeling Pipeline...")
    
    # 1. Prepare Data
    target_symbol = "SP500"  # We'll use SP500 instead of "S&P 500" since it aligns with the dataset
    X_train, y_train, X_test, y_test, scaler, num_features = prepare_data_for_lstm(
        PROCESSED_DATA_PATH, target_symbol, SEQ_LENGTH
    )
    
    # 2. Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        print(f"🔬 MLflow Run Started (ID: {run.info.run_id})")
        
        # Log hyperparameters
        params = {
            "target_variable": target_symbol, "seq_length": SEQ_LENGTH, "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_features": num_features
        }
        mlflow.log_params(params)
        
        # 3. Initialize Model, Loss, and Optimizer
        model = LSTMModel(input_size=num_features, hidden_layer_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 4. Training Loop
        print("⏳ Training LSTM model...")
        for i in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            single_loss = loss_function(y_pred, y_train)
            single_loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch {i+1}/{EPOCHS}, Loss: {single_loss.item():.6f}')
            mlflow.log_metric("train_loss", single_loss.item(), step=i)
            
        # 5. Evaluation
        print("📈 Evaluating model on test data...")
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
        
        # To accurately inverse transform, we need to pad the predictions with zeros for the other features
        # since the scaler expects `num_features` columns
        
        def inverse_transform_target(scaled_preds, scaler_obj, n_features):
            dummy = np.zeros((len(scaled_preds), n_features))
            dummy[:, 0] = scaled_preds.flatten()
            return scaler_obj.inverse_transform(dummy)[:, 0]

        actual_preds = inverse_transform_target(test_preds.numpy(), scaler, num_features)
        actual_y_test = inverse_transform_target(y_test.numpy(), scaler, num_features)
        
        rmse = np.sqrt(mean_squared_error(actual_y_test, actual_preds))
        print(f"Test RMSE: {rmse:.4f}")
        mlflow.log_metric("test_rmse", rmse)
        
        # 6. Log Model
        mlflow.pytorch.log_model(model, "lstm_model")
        print("✅ Model logged to MLflow.")
        
        # 7. Log Prediction Chart (Optional)
        fig = plt.figure(figsize=(12, 6))
        plt.title(f"{TARGET_VARIABLE} Prediction vs Actual")
        plt.plot(actual_y_test, label="Actual Values")
        plt.plot(actual_preds, label="Predicted Values")
        plt.legend()
        mlflow.log_figure(fig, "prediction_vs_actual.png")
        print("✅ Prediction chart logged to MLflow.")
        
    print("\n🎉 FinLagX PyTorch Pipeline completed successfully!")

if __name__ == "__main__":
    main()