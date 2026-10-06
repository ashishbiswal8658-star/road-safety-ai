# AI-Based Intelligent Multi-Vehicle Predictive Road Safety & Accident Prevention System

## Setup
```
pip install -r requirements.txt
```
Copy your datasets (CSV/XLSX) into `data/raw/`.

## Run in order
| Step | Command | What it does |
|---|---|---|
| 1 | `python src/01_inspect_data.py` | shows columns, types, missing values |
| 2 | `python src/02_clean_data.py` | fixes missing/null values, duplicates, dates |
| 3 | `python src/03_merge_data.py` | builds the master dataset |
| 4 | `python src/04_train_model.py` | trains + tests the model |
| 5 | `python app.py` | starts the Flask API |

Open `http://127.0.0.1:5000/dashboard` for the dashboard.

## API
- `GET /` health check
- `POST /risk` rule-based risk (speed, visibility, rain, hazard within 300 m)
- `POST /predict` ML prediction from the trained model

## Later modules
- `COCO-conversion-script.py` / `YOLO-conversion-script.py`: prepare pothole/crack/manhole images for YOLO training
- GPS/map module and SQL database (SQLite) come after the model works
