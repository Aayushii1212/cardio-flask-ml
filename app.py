from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
import json
import os

def load_dataframe():
    return pd.read_csv(
        os.path.join(BASE_DIR, "cardio_train.csv"),
        sep=";"
    )

# =========================
# APP SETUP
# =========================
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD MODEL & DATA SAFELY
# =========================
model = joblib.load(os.path.join(BASE_DIR, "cardio_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# FIXED: relative CSV path (DEPLOYMENT SAFE)
# df = pd.read_csv(
#     os.path.join(BASE_DIR, "cardio_train.csv"),
#     sep=";"
# )

# =========================
# FEATURE CONFIGURATION
# =========================
FEATURE_ORDER = [
    "gender", "weight", "ap_hi", "ap_lo", "cholesterol",
    "gluc", "smoke", "alco", "active", "ageyears",
    "heightm", "bmi", "pulse_pressure"
]

NUMERIC_COLS = [
    "ageyears", "heightm", "weight", "ap_hi",
    "ap_lo", "bmi", "pulse_pressure"
]

# =========================
# PREDICTION STATS
# =========================
prediction_stats = {
    "total": 0,
    "high_risk": 0,
    "low_risk": 0
}

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/analytics")
def analytics():
    df = load_dataframe()

    stats = {
        'total_records': len(df),
        'cardio_positive': int(df['cardio'].sum()),
        'cardio_negative': int(len(df) - df['cardio'].sum()),
        'avg_age': float(df['age'].mean() / 365),
        'gender_dist': df['gender'].value_counts().to_dict(),
        'cholesterol_dist': df['cholesterol'].value_counts().to_dict(),
        'prediction_stats': prediction_stats
    }
    return render_template("analytics.html", stats=stats)


@app.route("/api/analytics-data")
def get_analytics_data():
    df=load_dataframe()
    df['ageyears'] = (df['age'] / 365).astype(int)

    age_groups = pd.cut(
        df['ageyears'],
        bins=[0, 40, 50, 60, 70, 120],
        labels=['<40', '40-50', '50-60', '60-70', '70+']
    )
    age_dist = age_groups.value_counts().sort_index().to_dict()

    df['heightm'] = df['height'] / 100
    df['bmi'] = df['weight'] / (df['heightm'] ** 2)

    bmi_groups = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    bmi_dist = bmi_groups.value_counts().to_dict()

    bp_high = len(df[(df['ap_hi'] > 140) | (df['ap_lo'] > 90)])
    bp_normal = len(df) - bp_high

    data = {
        'age_distribution': {k: int(v) for k, v in age_dist.items()},
        'bmi_distribution': {k: int(v) for k, v in bmi_dist.items()},
        'bp_distribution': {'Normal': bp_normal, 'High': bp_high},
        'cardio_by_age': df.groupby('ageyears')['cardio'].mean().to_dict(),
        'risk_factors': {
            'smoking': int(df['smoke'].sum()),
            'alcohol': int(df['alco'].sum()),
            'inactive': int((df['active'] == 0).sum())
        }
    }
    return jsonify(data)

# =========================
# FEATURE ENGINEERING
# =========================
def engineer_features(form):
    ageyears = float(form["age"])
    height = float(form["height"])
    weight = float(form["weight"])
    ap_hi = float(form["ap_hi"])
    ap_lo = float(form["ap_lo"])

    heightm = height / 100
    bmi = weight / (heightm ** 2)
    pulse_pressure = ap_hi - ap_lo

    features = {
        "gender": int(form["gender"]),
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": int(form["cholesterol"]),
        "gluc": int(form["gluc"]),
        "smoke": int(form["smoke"]),
        "alco": int(form["alco"]),
        "active": int(form["active"]),
        "ageyears": ageyears,
        "heightm": heightm,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure
    }

    return features, bmi

# =========================
# BUILD & SCALE INPUT
# =========================
def build_model_input(features):
    X = np.array([[features[f] for f in FEATURE_ORDER]])
    numeric_values = np.array([[features[c] for c in NUMERIC_COLS]])

    scaled_numeric = scaler.transform(numeric_values)

    for i, col in enumerate(NUMERIC_COLS):
        idx = FEATURE_ORDER.index(col)
        X[0, idx] = scaled_numeric[0, i]

    return X

# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            features, bmi = engineer_features(request.form)
            model_input = build_model_input(features)

            pred = model.predict(model_input)[0]
            probability = (
                model.predict_proba(model_input)[0]
                if hasattr(model, "predict_proba")
                else None
            )

            result = (
                "High Risk of Cardiovascular Disease"
                if pred == 1
                else "Low Risk of Cardiovascular Disease"
            )

            risk_level = "high" if pred == 1 else "low"

            prediction_stats["total"] += 1
            prediction_stats["high_risk"] += int(pred == 1)
            prediction_stats["low_risk"] += int(pred == 0)

            risk_factors = []
            if bmi > 30:
                risk_factors.append("BMI indicates obesity (BMI > 30)")
            elif bmi > 25:
                risk_factors.append("BMI indicates overweight (BMI 25–30)")

            if features["ap_hi"] > 140 or features["ap_lo"] > 90:
                risk_factors.append("High blood pressure (Hypertension)")

            if features["cholesterol"] >= 2:
                risk_factors.append("Elevated cholesterol levels")

            if features["gluc"] >= 2:
                risk_factors.append("Elevated glucose levels")

            if features["smoke"] == 1:
                risk_factors.append("Smoking habit")

            if features["alco"] == 1:
                risk_factors.append("Alcohol consumption")

            if features["active"] == 0:
                risk_factors.append("Low physical activity")

            confidence = (
                round(max(probability) * 100, 1)
                if probability is not None
                else None
            )

            return render_template(
                "result.html",
                prediction=result,
                risk_level=risk_level,
                risk_factors=risk_factors if risk_factors else None,
                bmi=round(bmi, 2),
                confidence=confidence,
                error=False
            )

        except Exception as e:
            return render_template(
                "result.html",
                prediction=f"Error processing data: {str(e)}",
                error=True
            )

    return render_template("predict.html")

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
