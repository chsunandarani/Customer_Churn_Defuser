"""
Customer Churn Defuser - Flask Web Application
AI-Powered Customer Retention and Churn Prediction System
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import plotly
import plotly.graph_objs as go
import plotly.express as px

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
    session,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "churn-defuser-secret-key-change-in-production"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

DATASET_PATH = os.path.join("dataset", "customer_churn_dataset.csv")
LAST_RESULTS_PATH = os.path.join(UPLOAD_FOLDER, "predicted_results.csv")

# ---------------------------------------------------------------------------
# Load model + scaler at startup
# ---------------------------------------------------------------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    MODEL_LOADED = True
except Exception as e:  # noqa: BLE001
    print(f"WARNING: could not load model/scaler: {e}")
    model = None
    scaler = None
    MODEL_LOADED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def risk_level_from_score(score):
    """score is 0-100"""
    if score <= 40:
        return "Safe"
    elif score <= 70:
        return "Medium Risk"
    else:
        return "High Risk"


def recommendation_from_level(level):
    if level == "High Risk":
        return ["Offer 20% Discount", "Customer Success Call"]
    elif level == "Medium Risk":
        return ["Personalized Email", "Product Tutorial"]
    else:
        return ["Loyalty Rewards"]


def predict_single(features_dict):
    """features_dict: dict with FEATURE_COLUMNS keys (raw values)"""
    row = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
    scaled = scaler.transform(row)
    proba = model.predict_proba(scaled)[0][1]
    risk_score = round(float(proba) * 100, 2)
    level = risk_level_from_score(risk_score)
    recs = recommendation_from_level(level)
    return risk_score, level, recs


def predict_batch(df):
    """df must contain FEATURE_COLUMNS. Returns df with risk columns appended."""
    X = df[FEATURE_COLUMNS]
    scaled = scaler.transform(X)
    probs = model.predict_proba(scaled)[:, 1]
    risk_scores = np.round(probs * 100, 2)
    levels = [risk_level_from_score(s) for s in risk_scores]
    recs = [", ".join(recommendation_from_level(l)) for l in levels]

    out = df.copy()
    out["risk_score"] = risk_scores
    out["risk_level"] = levels
    out["recommendation"] = recs
    return out


def load_dashboard_stats():
    """Compute dashboard KPIs from the base dataset + model predictions."""
    if not os.path.exists(DATASET_PATH):
        return {
            "total_customers": 0,
            "churn_rate": 0,
            "active_customers": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "safe": 0,
        }

    df = pd.read_csv(DATASET_PATH)
    total_customers = len(df)
    churn_rate = round(df["churn"].mean() * 100, 2)
    active_customers = int((df["churn"] == 0).sum())

    high_risk = medium_risk = safe = 0
    if MODEL_LOADED:
        sample = df.sample(min(3000, len(df)), random_state=1)
        preds = predict_batch(sample)
        high_risk = int((preds["risk_level"] == "High Risk").sum())
        medium_risk = int((preds["risk_level"] == "Medium Risk").sum())
        safe = int((preds["risk_level"] == "Safe").sum())
        # scale up to full dataset proportion
        scale_factor = total_customers / len(sample)
        high_risk = int(high_risk * scale_factor)
        medium_risk = int(medium_risk * scale_factor)
        safe = int(safe * scale_factor)

    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "active_customers": active_customers,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "safe": safe,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Demo login - any credentials accepted
        session["logged_in"] = True
        session["username"] = request.form.get("username", "Admin")
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def dashboard():
    stats = load_dashboard_stats()
    return render_template("dashboard.html", stats=stats, model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        if not MODEL_LOADED:
            flash("Model not found. Please run train_model.py first.", "danger")
            return redirect(url_for("predict"))
        try:
            features = {
                "age": float(request.form["age"]),
                "monthly_fee": float(request.form["monthly_fee"]),
                "subscription_months": float(request.form["subscription_months"]),
                "login_frequency": float(request.form["login_frequency"]),
                "session_duration": float(request.form["session_duration"]),
                "support_tickets": float(request.form["support_tickets"]),
                "payment_failures": float(request.form["payment_failures"]),
                "last_active_days": float(request.form["last_active_days"]),
                "customer_satisfaction": float(request.form["customer_satisfaction"]),
                "feature_usage_score": float(request.form["feature_usage_score"]),
                "referrals": float(request.form["referrals"]),
                "engagement_score": float(request.form["engagement_score"]),
            }
            risk_score, level, recs = predict_single(features)
            result = {
                "risk_score": risk_score,
                "risk_level": level,
                "recommendations": recs,
                "inputs": features,
            }
        except (KeyError, ValueError) as e:
            flash(f"Invalid input: {e}", "danger")

    return render_template("prediction.html", result=result, model_loaded=MODEL_LOADED)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if not MODEL_LOADED:
            flash("Model not found. Please run train_model.py first.", "danger")
            return redirect(url_for("upload"))

        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "warning")
            return redirect(url_for("upload"))

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing_cols:
                flash(
                    f"CSV is missing required columns: {', '.join(missing_cols)}",
                    "danger",
                )
                return redirect(url_for("upload"))

            results_df = predict_batch(df)
            results_df.to_csv(LAST_RESULTS_PATH, index=False)

            table_html = results_df.to_html(
                classes="table table-striped table-hover align-middle",
                index=False,
                border=0,
            )
            return render_template(
                "results.html",
                table_html=table_html,
                total=len(results_df),
                high_risk=int((results_df["risk_level"] == "High Risk").sum()),
                medium_risk=int((results_df["risk_level"] == "Medium Risk").sum()),
                safe=int((results_df["risk_level"] == "Safe").sum()),
            )
        except Exception as e:  # noqa: BLE001
            flash(f"Error processing file: {e}", "danger")
            return redirect(url_for("upload"))

    return render_template("upload.html", model_loaded=MODEL_LOADED)


@app.route("/download")
def download():
    if os.path.exists(LAST_RESULTS_PATH):
        return send_file(LAST_RESULTS_PATH, as_attachment=True, download_name="predicted_results.csv")
    flash("No results available yet. Please upload a CSV first.", "warning")
    return redirect(url_for("upload"))


@app.route("/analytics")
def analytics():
    if not os.path.exists(DATASET_PATH):
        flash("Dataset not found. Please run dataset_generator.py first.", "danger")
        return render_template("analytics.html", charts={})

    df = pd.read_csv(DATASET_PATH)

    charts = {}

    # Churn distribution pie (Plotly)
    churn_counts = df["churn"].value_counts().rename({0: "Active", 1: "Churned"})
    fig1 = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        title="Churn Distribution",
        color=churn_counts.index,
        color_discrete_map={"Active": "#2E86DE", "Churned": "#EE5253"},
    )
    charts["churn_distribution"] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Risk distribution (if model loaded)
    if MODEL_LOADED:
        sample = df.sample(min(2000, len(df)), random_state=7)
        preds = predict_batch(sample)
        risk_counts = preds["risk_level"].value_counts()
        fig2 = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Distribution (sample)",
            labels={"x": "Risk Level", "y": "Customers"},
            color=risk_counts.index,
            color_discrete_map={
                "Safe": "#2ECC71",
                "Medium Risk": "#F39C12",
                "High Risk": "#E74C3C",
            },
        )
        charts["risk_distribution"] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(FEATURE_COLUMNS))

        fig3 = px.bar(
            x=importances,
            y=FEATURE_COLUMNS,
            orientation="h",
            title="Feature Importance",
            labels={"x": "Importance", "y": "Feature"},
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        charts["feature_importance"] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Monthly trend (synthetic trend based on subscription_months bucket)
    trend_df = (
        df.groupby(df["subscription_months"] // 6 * 6)["churn"]
        .mean()
        .reset_index()
        .rename(columns={"subscription_months": "month_bucket", "churn": "churn_rate"})
    )
    fig4 = px.line(
        trend_df,
        x="month_bucket",
        y="churn_rate",
        title="Churn Rate by Subscription Tenure (6-month buckets)",
        markers=True,
    )
    charts["monthly_trends"] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    # Customer segmentation: engagement vs satisfaction colored by churn
    sample_seg = df.sample(min(1500, len(df)), random_state=3)
    fig5 = px.scatter(
        sample_seg,
        x="engagement_score",
        y="customer_satisfaction",
        color=sample_seg["churn"].map({0: "Active", 1: "Churned"}),
        title="Customer Segmentation: Engagement vs Satisfaction",
        color_discrete_map={"Active": "#2E86DE", "Churned": "#EE5253"},
    )
    charts["segmentation"] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("analytics.html", charts=charts)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
