import os
import pandas as pd
from datetime import datetime
import yaml
import pandas_datareader.data as web

CONFIG_PATH = "src/data/config_macro.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

START_DATE = config["start_date"]
MACRO = config["macro"]

DATA_DIR = "data/raw/macro"
os.makedirs(DATA_DIR, exist_ok=True)

def download_macro_indicator(code: str, name: str, start: str, end: str):
    try:
        df = web.DataReader(code, "fred", start, end)
        if df.empty:
            print(f"⚠️ No data for {name} ({code})")
            return None
        file_path = os.path.join(DATA_DIR, f"{name}.parquet")
        df.to_parquet(file_path)
        print(f"✅ Saved {name} -> {file_path}")
        return df
    except Exception as e:
        print(f"❌ Failed {name} ({code}): {e}")
        return None

def download_all_macro():
    end_date = datetime.today().strftime("%Y-%m-%d")
    for name, code in MACRO.items():
        print(f"\n📊 Downloading {name} ({code})...")
        download_macro_indicator(code, name, START_DATE, end_date)
