"""STEP 4 - Train and test the accident risk model."""
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from utils import config

path = config.MASTER
if not path.exists():
    raise SystemExit("Run 03_merge_data.py first.")
df = pd.read_csv(path)

# ---- choose target column ----
target = config.TARGET
if target is None:
    for kw in config.TARGET_KEYWORDS:
        hits = [c for c in df.columns if kw in c]
        if hits:
            target = hits[0]
            break
if target is None or target not in df.columns:
    raise SystemExit(f"Could not find a target column. Set TARGET in config.py. Columns: {list(df.columns)}")
print("Target column:", target)

y = df[target]
X = df.drop(columns=[target] + [c for c in config.DROP_FEATURES if c in df.columns])

# drop ID-like / very high-cardinality text columns
for c in list(X.columns):
    if X[c].dtype == "object" and X[c].nunique() > 50:
        X = X.drop(columns=c)
    elif X[c].nunique() == len(X):
        X = X.drop(columns=c)

is_class = y.dtype == "object" or y.nunique() <= 10
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])
est = (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
       if is_class else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
model = Pipeline([("pre", pre), ("model", est)])

strat = y if is_class and y.value_counts().min() >= 2 else None
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

if is_class:
    metrics = {"accuracy": float(accuracy_score(y_te, pred)),
               "f1_weighted": float(f1_score(y_te, pred, average="weighted"))}
    print(classification_report(y_te, pred, zero_division=0))
else:
    metrics = {"r2": float(r2_score(y_te, pred)), "mae": float(mean_absolute_error(y_te, pred))}
print("Test metrics:", metrics)
print("NOTE: if accuracy is ~100%, a feature probably leaks the answer. Add it to DROP_FEATURES in config.py.")

config.MODELS.mkdir(exist_ok=True)
joblib.dump(model, config.MODELS / "accident_model.joblib")
meta = {"target": target, "task": "classification" if is_class else "regression",
        "numeric_features": num_cols, "categorical_features": cat_cols, "metrics": metrics}
(config.MODELS / "road_model_metadata.json").write_text(json.dumps(meta, indent=2))
print("Saved model to models/accident_model.joblib")
