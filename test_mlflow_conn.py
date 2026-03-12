import mlflow
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    try:
        logger.info(f"Testing connection to {mlflow.get_tracking_uri()}")
        mlflow.set_experiment("Test_Experiment")
        with mlflow.start_run():
            mlflow.log_param("test", 1)
            # Create a dummy file and log it as artifact
            with open("test.txt", "w") as f:
                f.write("test")
            mlflow.log_artifact("test.txt")
        logger.info("✅ Connection successful!")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_mlflow()
