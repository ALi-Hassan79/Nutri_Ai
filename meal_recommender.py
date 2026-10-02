"""
NutriAI - ML Meal Recommender
Ranks meal candidates using a k-Nearest-Neighbors similarity search in
macro-profile space, instead of a hand-written "closest calorie count" rule.

Design note: hard constraints (allergies, diet type, cuisine) stay rule-based
in model.py's _filter_meals - those are safety/preference constraints that
must never be "soft", so ML has no business overriding them. What the ML
model contributes is the *ranking* among already-safe candidates: given the
user's target macro split (protein/carb/fat percentages) and a calorie
budget for the slot, which meals' nutritional profile is closest to what the
user actually needs right now? That's a genuine similarity-search problem,
not something a single sort key expresses well once more than one factor
(calories AND protein-density AND carb-density AND fat-density) matters at once.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Feature weights: calorie fit matters most (must roughly hit the slot
# budget), macro-ratio fit is secondary personalization on top of that.
FEATURE_WEIGHTS = np.array([2.2, 1.0, 1.0, 1.0])


def _meal_vector(meal: dict, slot_budget: float) -> list:
    cals = max(meal["calories"], 1)
    cal_fit = meal["calories"] / slot_budget if slot_budget > 0 else 1.0
    protein_ratio = (meal["protein"] * 4) / cals
    carbs_ratio = (meal["carbs"] * 4) / cals
    fat_ratio = (meal["fat"] * 9) / cals
    return [cal_fit, protein_ratio, carbs_ratio, fat_ratio]


def rank_candidates(
    candidates: list,
    slot_budget: float,
    target_ratios: tuple,
    top_k: int = 3,
) -> list:
    """Return up to top_k candidates ordered by nearest-neighbor distance to
    the user's ideal macro/calorie profile for this meal slot.

    target_ratios: (protein_ratio, carbs_ratio, fat_ratio) derived from the
    user's overall daily macro percentages (e.g. 0.30, 0.45, 0.25).
    """
    if not candidates:
        return []

    n = len(candidates)
    k = min(top_k, n)

    X = np.array([_meal_vector(m, slot_budget) for m in candidates]) * FEATURE_WEIGHTS
    target_vec = (np.array([1.0, *target_ratios]) * FEATURE_WEIGHTS).reshape(1, -1)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(target_vec)

    ranked = []
    for dist, idx in zip(distances[0], indices[0]):
        # Convert distance to an intuitive 0-100 "match score" via exponential
        # decay (calibrated against real meal-database distances so a strong
        # match reads ~85-95% and a poor one drops below ~40%).
        match_score = round(min(100 * np.exp(-dist * 0.55), 99.9), 1)
        meal = dict(candidates[idx])
        meal["match_score"] = match_score
        ranked.append(meal)
    return ranked
