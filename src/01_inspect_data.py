"""STEP 1 - Look at every dataset: shape, columns, types, missing values, duplicates."""
from utils import read_any, raw_files, config

lines = []
files = raw_files()
if not files:
    raise SystemExit(f"No datasets found. Put your CSV/XLSX files in: {config.RAW}")

for f in files:
    df = read_any(f)
    miss = df.isnull().sum()
    lines += [
        "=" * 70, f"FILE: {f.name}", f"shape: {df.shape}", "",
        "columns and types:", df.dtypes.to_string(), "",
        "missing values:", (miss[miss > 0].to_string() if miss.any() else "none"), "",
        f"duplicate rows: {df.duplicated().sum()}", "",
        "first 3 rows:", df.head(3).to_string(), "",
    ]

text = "\n".join(lines)
print(text)
config.REPORTS.mkdir(exist_ok=True)
(config.REPORTS / "inspection_report.txt").write_text(text, encoding="utf-8")
print("\nSaved: reports/inspection_report.txt")
