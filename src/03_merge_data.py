"""STEP 3 - Combine cleaned datasets into one master dataset.
Base = the accident dataset. Other datasets are joined only on columns they really share."""
import pandas as pd
from utils import config

files = sorted(config.CLEANED.glob("*.csv"))
if not files:
    raise SystemExit("Run 02_clean_data.py first.")

frames = {f.stem: pd.read_csv(f) for f in files}
base_name = next((n for n in frames if "accident" in n.lower()), max(frames, key=lambda n: len(frames[n])))
master = frames[base_name]
print(f"Base dataset: {base_name} {master.shape}")

for name, df in frames.items():
    if name == base_name:
        continue
    keys = [k for k in df.columns if k in master.columns and
            any(k.startswith(c) for c in config.JOIN_KEY_CANDIDATES)]
    if not keys:
        print(f"SKIP {name}: no shared key columns with the base dataset "
              f"(columns: {list(df.columns)[:8]}...)")
        continue
    num = df.select_dtypes(include="number").columns.difference(keys)
    agg = df.groupby(keys, as_index=False)[list(num)].mean() if len(num) else df[keys].drop_duplicates()
    master = master.merge(agg, on=keys, how="left", suffixes=("", f"_{name[:8]}"))
    print(f"MERGED {name} on {keys} -> {master.shape}")

# rows that got no match from a joined dataset: fill with median
for c in master.select_dtypes(include="number").columns:
    master[c] = master[c].fillna(master[c].median())

config.MASTER.parent.mkdir(parents=True, exist_ok=True)
master.to_csv(config.MASTER, index=False)
print(f"\nSaved master dataset: {config.MASTER}  shape={master.shape}")
