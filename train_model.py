"""
Customer Churn Defuser - Model Training Pipeline
Performs EDA, trains Logistic Regression / Random Forest / Gradient Boosting,
compares them, picks the best model by ROC-AUC, and persists the model + scaler.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import joblib

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("dataset", "customer_churn_dataset.csv")
CHARTS_DIR = os.path.join("static", "images")
MODELS_DIR = "models"

FEATURE_COLUMNS = [
    "age",
    "monthly_fee",
    "subscription_months",
    "login_frequency",
    "session_duration",
    "support_tickets",
    "payment_failures",
    "last_active_days",
    "customer_satisfaction",
    "feature_usage_score",
    "referrals",
    "engagement_score",
]


def ensure_dirs():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def run_eda(df):
    print("\n===== EXPLORATORY DATA ANALYSIS =====")

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nSummary statistics:")
    print(df[FEATURE_COLUMNS + ["churn"]].describe())

    # Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="churn", data=df, palette=["#2E86DE", "#EE5253"])
    plt.title("Churn Distribution (0 = Active, 1 = Churned)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "churn_distribution.png"))
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(12, 9))
    corr = df[FEATURE_COLUMNS + ["churn"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"))
    plt.close()

    # Feature distributions
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E86DE")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "feature_distributions.png"))
    plt.close()

    # Pairplot on a sample subset of features (full pairplot is expensive)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    pair_cols = ["login_frequency", "engagement_score", "support_tickets", "last_active_days", "churn"]
    pp = sns.pairplot(sample_df[pair_cols], hue="churn", palette=["#2E86DE", "#EE5253"])
    pp.savefig(os.path.join(CHARTS_DIR, "pairplot.png"))
    plt.close()

    print(f"\nEDA charts saved to: {CHARTS_DIR}")


def build_models():
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [6, 10, None],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        },
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    results = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, spec in build_models().items():
        print(f"\nTraining {name} with GridSearchCV...")
        grid = GridSearchCV(
            spec["model"], spec["params"], cv=cv, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "best_params": grid.best_params_,
        }

        results[name] = metrics
        fitted_models[name] = best_model

        print(f"{name} results: {metrics}")

    return results, fitted_models


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[idx],
        y=np.array(feature_names)[idx],
        color="#2E86DE",
    )
    plt.title("Feature Importance - Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"))
    plt.close()


def main():
    ensure_dirs()

    df = pd.read_csv(DATA_PATH)

    run_eda(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, fitted_models = train_and_compare(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]

    print(f"\n===== BEST MODEL: {best_name} =====")
    print(results[best_name])

    plot_feature_importance(best_model, FEATURE_COLUMNS)

    # Persist model + scaler
    joblib.dump(best_model, "churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "all_results": results,
                "feature_columns": FEATURE_COLUMNS,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nModel and scaler saved as churn_model.pkl and scaler.pkl")
    print(f"Metrics saved to {MODELS_DIR}/model_metrics.json")


if __name__ == "__main__":
    main()
