"""
NutriAI - Progress Forecaster
Fits a linear regression to a user's logged weight entries to project their
trend forward, and compares the observed rate of change against what their
calorie target *should* produce physiologically (~7700 kcal ≈ 1 kg of body
weight), flagging whether they're on track for their stated goal.

This is a genuine (if simple) regression problem: the input is noisy,
irregularly-spaced real-world data (people don't weigh in perfectly daily),
and the output - a slope with a confidence measure (R^2) - is exactly the
kind of quantity a linear model is suited to estimate, rather than something
you'd hand-derive from two points.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

KCAL_PER_KG = 7700  # commonly used approximation for adipose tissue energy density


def forecast_progress(entries: list, target_calories: float, tdee: float, weeks_ahead: int = 4) -> dict:
    """
    entries: list of {"day": <int offset from first entry>, "weight": <float kg>}
             must be at least 2 points, ideally spread over 1-2+ weeks for a
             meaningful trend.
    """
    if not entries or len(entries) < 2:
        return {"error": "Need at least 2 weight entries (on different days) to fit a trend."}

    entries = sorted(entries, key=lambda e: e["day"])
    days = np.array([e["day"] for e in entries], dtype=float)
    weights = np.array([e["weight"] for e in entries], dtype=float)

    if days[-1] == days[0]:
        return {"error": "All entries are on the same day - log at least two different days."}

    X = days.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, weights)
    predicted = model.predict(X)

    r2 = float(r2_score(weights, predicted)) if len(entries) > 2 else None

    slope_per_day = float(model.coef_[0])
    slope_per_week = slope_per_day * 7

    last_day = float(days[-1])
    future_day = last_day + weeks_ahead * 7
    projected_weight = float(model.predict([[future_day]])[0])

    daily_cal_diff = target_calories - tdee
    expected_kg_per_week = (daily_cal_diff * 7) / KCAL_PER_KG

    on_track, status = _assess_trend(slope_per_week, expected_kg_per_week)

    return {
        "slope_kg_per_week": round(slope_per_week, 3),
        "expected_kg_per_week": round(expected_kg_per_week, 3),
        "projected_weight": round(projected_weight, 1),
        "weeks_ahead": weeks_ahead,
        "r2": round(r2, 3) if r2 is not None else None,
        "on_track": on_track,
        "status": status,
        "n_entries": len(entries),
        "current_weight": round(float(weights[-1]), 1),
        "starting_weight": round(float(weights[0]), 1),
    }


def _assess_trend(actual_kg_per_week: float, expected_kg_per_week: float):
    """Compare the fitted real-world trend to the physiologically expected one."""
    tolerance = 0.12  # kg/week - within this of zero counts as "roughly flat"

    if abs(expected_kg_per_week) < tolerance:
        # Maintenance goal: expect the trend to be roughly flat.
        if abs(actual_kg_per_week) < tolerance * 1.5:
            return True, "Your weight is holding steady, matching a maintenance goal."
        direction = "gaining" if actual_kg_per_week > 0 else "losing"
        return False, f"You're {direction} faster than expected for a maintenance goal - consider adjusting intake."

    same_direction = (actual_kg_per_week > 0) == (expected_kg_per_week > 0)
    if not same_direction:
        return False, "Your logged trend is moving the opposite direction from your goal."

    ratio = actual_kg_per_week / expected_kg_per_week
    if 0.4 <= ratio <= 2.0:
        return True, "Your logged trend is tracking in line with your calorie target."
    if ratio < 0.4:
        return False, "Progress is slower than your calorie target implies - check adherence to the plan."
    return False, "Progress is faster than expected - worth double-checking your logged weights and intake."
