import pandas as pd
import os
import glob

def summarize_results(model_name):
    print("=" * 80)
    print(f"{model_name.upper()} MODEL RESULTS (with Sentiment Features)")
    print("=" * 80)
    
    metrics_files = glob.glob(f'data/results/{model_name.lower()}/*_metrics.csv')
    if not metrics_files:
        print(f"No results found for {model_name}")
        return
        
    all_metrics = []
    for f in sorted(metrics_files):
        df = pd.read_csv(f)
        symbol = os.path.basename(f).split('_')[1].upper() if model_name == 'tcn' else os.path.basename(f).split('_')[0].upper()
        df['Symbol'] = symbol
        df['Model'] = model_name.upper()
        all_metrics.append(df)

    combined = pd.concat(all_metrics)
    
    # Check if 'Model' column exists in dataframe, if not use the one we added
    if 'Model' not in combined.columns:
        combined['Model'] = model_name.upper()
        
    print(combined[['Symbol', 'Model', 'RMSE', 'MAE', 'Directional_Accuracy_%', 'Correlation']].to_string(index=False))
    print()
    
    avg_acc = combined["Directional_Accuracy_%"].mean()
    avg_rmse = combined["RMSE"].mean()
    avg_corr = combined["Correlation"].mean()
    
    print(f"  Average Directional Accuracy: {avg_acc:.2f}%")
    print(f"  Average RMSE: {avg_rmse:.6f}")
    print(f"  Average Correlation: {avg_corr:.4f}")
    print(f"  Total symbols processed: {len(combined)}")
    print()
    return combined

lstm_res = summarize_results('lstm')
tcn_res = summarize_results('tcn')

if lstm_res is not None and tcn_res is not None:
    print("=" * 80)
    print("COMPARISON: LSTM vs TCN")
    print("=" * 80)
    print(f"LSTM Avg Accuracy: {lstm_res['Directional_Accuracy_%'].mean():.2f}%")
    print(f"TCN  Avg Accuracy: {tcn_res['Directional_Accuracy_%'].mean():.2f}%")
    
    diff = tcn_res['Directional_Accuracy_%'].mean() - lstm_res['Directional_Accuracy_%'].mean()
    winner = "TCN" if diff > 0 else "LSTM"
    print(f"\n{winner} performed {abs(diff):.2f}% better overall.")
