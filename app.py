"""
Diet AI - Flask Web Application
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, send_file, session
import io

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import analyze_user
from pdf_report import generate_pdf
from body_classifier import BODY_MODEL_METRICS
from progress_forecast import forecast_progress
from nlp_advisor import answer_question, INTENT_MODEL_METRICS

app = Flask(__name__)
app.secret_key = "nutriai-secret-2024"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-metrics")
def model_metrics():
    """Expose the body-composition classifier's evaluation metrics
    (train/test accuracy, cross-validation, confusion matrix, feature
    importance) so the ML rigor behind the app is visible, not hidden."""
    return jsonify({"success": True, "metrics": BODY_MODEL_METRICS})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        # Validate required fields
        required = ["age", "height", "weight", "activity", "goal"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Run analysis
        result = analyze_user(data)

        # Store in session for PDF generation
        session["last_result"] = result

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    """Fit a regression to logged weight entries and project the trend
    against the physiologically expected rate for the user's calorie target."""
    try:
        data = request.get_json()
        entries = data.get("entries", [])
        target_calories = float(data.get("target_calories"))
        tdee = float(data.get("tdee"))
        weeks_ahead = int(data.get("weeks_ahead", 4))

        result = forecast_progress(entries, target_calories, tdee, weeks_ahead)
        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        return jsonify({"success": True, "forecast": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/nlp-metrics")
def nlp_metrics():
    """Expose the diet Q&A intent classifier's evaluation metrics."""
    return jsonify({"success": True, "metrics": INTENT_MODEL_METRICS})


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a free-text diet question by classifying its intent (TF-IDF +
    Naive Bayes) and generating a personalized response from the user's
    latest profile/targets."""
    try:
        data = request.get_json()
        question = (data.get("question") or "").strip()
        context = data.get("context") or {}

        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        result = answer_question(question, context)
        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    try:
        data = request.get_json()
        result = data.get("result")

        if not result:
            return jsonify({"error": "No result data"}), 400

        pdf_bytes = generate_pdf(result)

        name = result["user"].get("name", "User").replace(" ", "_")
        filename = f"NutriAI_Diet_Report_{name}.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
