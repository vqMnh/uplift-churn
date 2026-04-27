"""
Data ingestion for the Telco Customer Churn dataset.

Source: IBM Sample Data Sets (public domain)
URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
"""

import pathlib
import urllib.request

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"


def download_telco_churn(dest_dir: pathlib.Path | None = None, force: bool = False) -> pd.DataFrame:
    """Download the Telco Churn CSV and return as a DataFrame.

    The file is cached locally at dest_dir/telco_churn.csv so subsequent calls
    avoid re-downloading unless force=True.
    """
    dest_dir = pathlib.Path(dest_dir) if dest_dir else RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "telco_churn.csv"

    if not dest_path.exists() or force:
        print(f"Downloading dataset from {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, dest_path)
        print(f"Saved to {dest_path}")
    else:
        print(f"Using cached file: {dest_path}")

    df = pd.read_csv(dest_path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def load_raw(dest_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """Convenience wrapper used by other modules."""
    return download_telco_churn(dest_dir=dest_dir)
