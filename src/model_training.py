import pandas as pd
import joblib
import os
import json
import sys

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Include src directory in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import preprocess_data


def train_and_tune_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            }
        }
    }

    best_model = None
    best_score = 0
    best_model_name = ""
    all_results = {}

    for name, cfg in models.items():
        print(f"🔍 Tuning {name}...")
        clf = GridSearchCV(cfg["model"], cfg["params"], cv=3, scoring="f1", n_jobs=-1)
        clf.fit(X_train, y_train)

        score = clf.best_score_
        print(f"✅ {name} best F1 score: {score:.4f}")

        y_pred = clf.best_estimator_.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        all_results[name] = {
            "best_f1_cv": score,
            "test_report": report,
            "model": clf.best_estimator_
        }

        if score > best_score:
            best_score = score
            best_model = clf.best_estimator_
            best_model_name = name

    return all_results, best_model_name


def main():
    raw_data_path = "data/raw/noisy_manufacturing_quality_dataset.csv"
    preprocessor_path = "models/preprocessor_pipeline.pkl"
    model_output_path = "models/best_model.pkl"
    metrics_output_path = "reports/model_metrics.json"

    print("📦 Preprocessing data...")
    X, y = preprocess_data(input_path=raw_data_path, preprocessor_save_path=preprocessor_path)

    print("🏗️ Training and tuning models...")
    all_results, best_model_name = train_and_tune_models(X, y)

    # Create output folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save best model
    joblib.dump(all_results[best_model_name]["model"], model_output_path)

    # Save all metrics
    with open(metrics_output_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "all_results": {
                name: {
                    "best_f1_cv": res["best_f1_cv"],
                    "test_report": res["test_report"]
                } for name, res in all_results.items()
            }
        }, f, indent=2)

    print(f"✅ Best model ({best_model_name}) saved to {model_output_path}")
    print(f"📊 Evaluation report saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
