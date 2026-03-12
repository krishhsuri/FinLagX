import subprocess
import os
from summarize_mlflow import get_latest_metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cmd(cmd):
    logger.info(f"🚀 Running: {cmd}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f".;{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, env=env, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"❌ Error running {cmd}: {result.stderr}")
    else:
        logger.info(f"✅ Finished: {cmd}")

def main():
    logger.info("🎬 Starting Final Modeling Benchmark...")
    
    # 1. Run LightGBM Baseline
    run_cmd("python -m src.modeling.lgbm_model")
    
    # 2. Run Hurdle Model (Advanced)
    run_cmd("python -m src.modeling.hurdle_model")
    
    # 3. Run Tuner (Best Params)
    # run_cmd("python -m src.modeling.tuner") # Skipping tuner as it takes too long for a single report run
    
    # 4. Generate Report
    get_latest_metrics()

if __name__ == "__main__":
    main()
