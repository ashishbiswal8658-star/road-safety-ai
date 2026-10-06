import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file: {path}")


def raw_files():
    return sorted(p for p in config.RAW.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".xls"))


def clean_name(c: str) -> str:
    return "_".join(str(c).strip().lower().replace("/", " ").replace("-", " ").split())
