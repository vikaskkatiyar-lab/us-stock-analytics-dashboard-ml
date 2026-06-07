import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_features import FEATURE_COLUMNS


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "latest_model.pkl"
METADATA_FILE = MODEL_DIR / "latest_model_metadata.json"


def time_based_split(features: pd.DataFrame, test_fraction: float = 0.2):
    dates = sorted(pd.to_datetime(features["date"], errors="coerce").dropna().unique())
    if len(dates) < 20:
        raise ValueError("Not enough dated rows to train and test the ML models.")
    split_index = max(1, int(len(dates) * (1 - test_fraction)))
    split_date = dates[split_index]
    train = features[pd.to_datetime(features["date"], errors="coerce") < split_date].copy()
    test = features[pd.to_datetime(features["date"], errors="coerce") >= split_date].copy()
    return train, test, pd.Timestamp(split_date).date().isoformat()


def _candidate_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
    }


def _evaluate_model(model, test: pd.DataFrame) -> dict:
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target_next_close_up"].astype(int)
    predicted = model.predict(X_test)
    probability_up = model.predict_proba(X_test)[:, 1]
    confidence = abs(probability_up - 0.5) * 2
    return {
        "direction_accuracy": float(accuracy_score(y_test, predicted)),
        "precision_predicted_up": float(precision_score(y_test, predicted, zero_division=0)),
        "average_prediction_confidence": float(confidence.mean()),
    }


def train_and_select_model(features: pd.DataFrame) -> dict:
    train, test, split_date = time_based_split(features)
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_next_close_up"].astype(int)

    results = []
    models = {}
    for name, model in _candidate_models().items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
            model.fit(X_train, y_train)
            metrics = _evaluate_model(model, test)
        metrics["model_name"] = name
        results.append(metrics)
        models[name] = model

    comparison = pd.DataFrame(results).sort_values(
        ["direction_accuracy", "precision_predicted_up", "average_prediction_confidence", "model_name"],
        ascending=[False, False, False, True],
    )
    selected_name = comparison.iloc[0]["model_name"]
    selected_model = models[selected_name]

    metadata = {
        "selected_model": selected_name,
        "split_date": split_date,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": FEATURE_COLUMNS,
        "model_comparison": comparison.to_dict(orient="records"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_FILE)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": selected_model,
        "model_name": selected_name,
        "metadata": metadata,
        "train": train,
        "test": test,
        "comparison": comparison,
        "model_file": MODEL_FILE,
        "metadata_file": METADATA_FILE,
    }
