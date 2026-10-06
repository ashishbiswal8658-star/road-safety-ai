"""STEP 2 - Fix missing/null values, duplicates, types, dates. Saves to data/cleaned/."""
import pandas as pd
from utils import read_any, raw_files, clean_name, config

config.CLEANED.mkdir(parents=True, exist_ok=True)
config.REPORTS.mkdir(exist_ok=True)
report = []

for f in raw_files():
    df = read_any(f)
    before_rows, before_missing = len(df), int(df.isnull().sum().sum())

    df.columns = [clean_name(c) for c in df.columns]

    # strip text, turn blank/placeholder strings into real nulls
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].where(df[c].isnull(), df[c].astype(str).str.strip())
        df[c] = df[c].replace({"": None, "nan": None, "NaN": None, "None": None,
                               "null": None, "NULL": None, "N/A": None, "na": None, "-": None})

    df = df.drop_duplicates()

    # drop columns that are mostly empty
    mostly_empty = [c for c in df.columns if df[c].isnull().mean() > config.MISSING_COLUMN_THRESHOLD]
    df = df.drop(columns=mostly_empty)

    # dates -> year / month / day / dayofweek (+ hour if time present)
    for c in [c for c in df.columns if "date" in c or c == "timestamp"]:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notnull().mean() > 0.5:
            df[f"{c}_year"] = parsed.dt.year
            df[f"{c}_month"] = parsed.dt.month
            df[f"{c}_day"] = parsed.dt.day
            df[f"{c}_dayofweek"] = parsed.dt.dayofweek
            if (parsed.dt.hour.fillna(0) != 0).any():
                df[f"{c}_hour"] = parsed.dt.hour
            df = df.drop(columns=[c])

    # text columns that are really numbers
    for c in df.select_dtypes(include="object").columns:
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notnull().mean() > 0.9:
            df[c] = as_num

    # fill remaining nulls: numeric -> median, text -> most common value
    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                mode = df[c].mode()
                df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "unknown")

    out = config.CLEANED / f"{f.stem}.csv"
    df.to_csv(out, index=False)
    report.append(f"{f.name}: rows {before_rows}->{len(df)}, missing cells {before_missing}->"
                  f"{int(df.isnull().sum().sum())}, dropped columns {mostly_empty or 'none'}")
    print(report[-1])

(config.REPORTS / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nCleaned files saved in data/cleaned/")
