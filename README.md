# Uber Fleet Operations: Cancellation Risk Model

Pre-dispatch cancellation classifier and booking-value analysis for the NCR (Delhi) ride-bookings dataset. Raw bookings land in SQLite, a Python pipeline cleans and encodes them, and a Random Forest scores each ride's probability of not completing. Power BI reads the same SQLite store for the operational view.

## Findings

From the Power BI layer over the raw `rides` table (~150K bookings, 7 vehicle classes):

| Metric | Value |
|:--|:--|
| Booking value, status `Completed` | ₹47M (91.16%) |
| Booking value, status `Incomplete` | ₹5M (8.84%) |
| Highest booking volume and highest absolute cancellation count | `Auto` |
| Model features retained | 8, ranked by ANOVA F-test |

![Power BI dashboard](dashboards/Uber%20Statistics.png)

The dashboard has two visuals: booking value split by status (pie) and booking count by vehicle type, stacked by status (bar). Status buckets are `Completed`, `Cancelled by Customer`, `Cancelled by Driver`, `Incomplete` and `No Driver Found`.

## Data flow

```
ncr_ride_bookings.csv
 │
 ├── importer.py ──► uber_operations.db  (SQLite, table `rides`, raw rows)
 │                     ├──► Power BI
 │                     └──► targeted_check.py   SELECT rowid, * FROM rides WHERE rowid = ?
 │                                │
 └── pipeline.py                  ▼
       clean → encode → select → SMOTE → fit     predict_cancellation.check_ride_risk()
                                      │                    ▲
                                      └──► models/active_model.pkl
```

`importer.py` loads the CSV unchanged with `to_sql(if_exists="replace")`, so re-runs are idempotent. Cleaning happens only in the training path. SQLite and Power BI hold the raw records.

## Training pipeline

`OperationsPipeline.auto_retrain(data_path)` in `pipeline.py`:

| Step | Implementation |
|:--|:--|
| Null repair | The source encodes missing values as the string `"null"`. Replaced with `NaN`, then filled with `0`. |
| Target | `Booking Status` → `0` for `completed`/`success`, `1` for every other status. |
| Time feature | `Time` (`%H:%M:%S`) → integer `Hour` (0–23). Parse failures → `0`. |
| Categorical encoding | `Vehicle Type` one-hot via `pd.get_dummies(drop_first=True)`. |
| Type coercion | `Avg VTAT`, `Avg CTAT`, `Cancelled Rides by Customer` → `float` via `pd.to_numeric(errors="coerce")`. |
| Candidate features | All numeric and boolean columns, minus the target and `id`, `rowid`, `time`, `unnamed: 0`. |
| Feature selection | `SelectKBest(f_classif, k=min(n_features, 8))`. |
| Class balancing | `SMOTE(random_state=42)`. Completed rides are the majority class. |
| Model | `RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)`. |
| Artifact | `joblib.dump({"model", "features", "target_name"}, "models/active_model.pkl")`. |

The artifact stores the model together with its ordered feature list, so inference never depends on column order in the caller's input.

## Inference

```python
from predict_cancellation import check_ride_risk

check_ride_risk({
    "Hour": 21,
    "Avg VTAT": 15.0,
    "Avg CTAT": 8.3,
    "Cancelled Rides by Customer": 3,
    "Vehicle Type_Auto": 1,
})
```

`check_ride_risk` loads the artifact, pads any feature missing from the input with `0.0`, selects columns in manifest order, prints `P(cancel)` and returns one of:

```
flag: high risk of cancellation. dispatch alternative.
clear: ride likely to complete.
```

`targeted_check.analyze_ride(rowid)` runs the same call on a stored record, fetched with a parameterized query.

## Setup

Python 3.8+.

```bash
git clone https://github.com/Jai2379/Uber-Fleet-Operations.git
cd Uber-Fleet-Operations
pip install -r requirements.txt
```

Download the dataset from [Kaggle: Uber Ride Analytics](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard). The two entry points read it from different paths:

| Script | Expects |
|:--|:--|
| `importer.py` | `./ncr_ride_bookings.csv` |
| `pipeline.py` | `./data/ncr_ride_bookings.csv` |

```bash
python importer.py        # CSV → uber_operations.db
python pipeline.py        # train, write models/active_model.pkl
python targeted_check.py  # score rowid 1 end to end
```

## Layout

```
.
├── dashboards/Uber Statistics.png   Power BI export
├── models/active_model.pkl          trained model + feature manifest
├── importer.py                      CSV → SQLite loader
├── pipeline.py                      OperationsPipeline (train + serialize)
├── predict_cancellation.py          check_ride_risk()
├── targeted_check.py                rowid lookup → inference
└── requirements.txt
```

## Known gaps

- No held-out split. SMOTE and the fit run on the full dataset, so the repo reports no precision or recall.
- Feature selection is fit on the same data as the model.
- `targeted_check.py` scores raw SQLite rows, which lack the training-time `Hour` and one-hot columns. Those features are zero-padded at inference.
- `pipeline.py` calls `auto_retrain` at module level, so importing it starts a training run.
