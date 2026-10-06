from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEANED = BASE / "data" / "cleaned"
MASTER = BASE / "data" / "master" / "master_dataset.csv"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

# If you know your target column, write it here (after running step 1).
# Example: TARGET = "accident_severity". Leave None to auto-detect.
TARGET = None
TARGET_KEYWORDS = ["severity", "risk", "accident", "fatal", "casualt", "injur"]

# Columns that are IDs / leak the answer and should not be used as features.
DROP_FEATURES = []

# Columns that may be shared between datasets and used to join them.
JOIN_KEY_CANDIDATES = ["state", "city", "district", "year", "month", "date", "road_type"]

MISSING_COLUMN_THRESHOLD = 0.60   # drop a column if more than 60% is empty
