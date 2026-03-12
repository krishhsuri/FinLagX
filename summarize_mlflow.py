import mlflow
from mlflow.tracking import MlflowClient

def get_latest_metrics():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    experiments = client.search_experiments()
    
    print("\n🚀 --- FinLagX Latest Model Performance ---")
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs:
            run = runs[0]
            print(f"\nExperiment: {exp.name}")
            print(f"Run ID: {run.info.run_id}")
            print("Metrics:")
            for k, v in run.data.metrics.items():
                print(f"  - {k}: {v:.4f}")
    print("\n-------------------------------------------")

if __name__ == "__main__":
    get_latest_metrics()
