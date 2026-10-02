"""
NutriAI - Intelligent Nutrition Engine
Calorie science, body composition analysis, smart meal planning, and personalized guidance.
"""
import hashlib
import numpy as np

from food_database import (
    FOOD_DATABASE,
    CALORIE_SPLIT,
    MEAL_SLOTS,
    DIET_COMPATIBILITY,
    GOAL_FAMILY,
    DAYS,
)
from nutrition_advisor import generate_guidance
from body_classifier import (
    classify_body,
    BODY_MODEL_METRICS,
    BODY_LABELS,
    BODY_EMOJIS,
)
from meal_recommender import rank_candidates


# ─── Core Calculations ───────────────────────────────────────────────────────

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor Equation"""
    if gender.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calculate_tdee(bmr: float, activity_level: str) -> float:
    return round(bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.55), 0)


def get_target_calories(tdee: float, goal: str, gender: str = "male") -> float:
    adjustments = {
        "lose_weight": -500,
        "lose_fast": -750,
        "maintain": 0,
        "gain_muscle": +300,
        "gain_fast": +500,
    }
    floor = 1400 if gender.lower() == "male" else 1200
    return max(floor, tdee + adjustments.get(goal, 0))


# ─── Smart Meal Planner ───────────────────────────────────────────────────────

def _user_seed(data: dict) -> int:
    key = f"{data.get('name','')}-{data['age']}-{data['height']}-{data['weight']}-{data.get('diet','')}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _filter_meals(data: dict, goal_family: str) -> dict:
    diet = data.get("diet", "omnivore")
    allowed_tags = DIET_COMPATIBILITY.get(diet, DIET_COMPATIBILITY["omnivore"])
    allergies = set(data.get("allergies", []))
    cuisine_pref = data.get("cuisine", "any")

    pools = {slot: [] for slot in MEAL_SLOTS}
    for meal in FOOD_DATABASE:
        if meal["goal_fit"] != goal_family:
            continue
        if not allowed_tags.intersection(meal["diet_tags"]):
            continue
        if allergies.intersection(meal["allergens"]):
            continue
        if cuisine_pref != "any" and meal["cuisine"] != cuisine_pref:
            continue
        pools[meal["meal"]].append(meal)

    # Relax cuisine filter if pool too small
    for slot in MEAL_SLOTS:
        if len(pools[slot]) < 3 and cuisine_pref != "any":
            for meal in FOOD_DATABASE:
                if meal["goal_fit"] != goal_family or meal["meal"] != slot:
                    continue
                if not allowed_tags.intersection(meal["diet_tags"]):
                    continue
                if allergies.intersection(meal["allergens"]):
                    continue
                if meal not in pools[slot]:
                    pools[slot].append(meal)
    return pools


def _scale_meal(meal: dict, factor: float, note: bool = True) -> dict:
    if abs(factor - 1.0) < 0.03:
        return dict(meal)
    scaled = dict(meal)
    scaled["calories"] = int(round(meal["calories"] * factor))
    scaled["protein"] = int(round(meal["protein"] * factor))
    scaled["carbs"] = int(round(meal["carbs"] * factor))
    scaled["fat"] = int(round(meal["fat"] * factor))
    if note and factor != 1.0:
        pct = int(round((factor - 1) * 100))
        sign = "+" if pct > 0 else ""
        scaled["portion_note"] = f"Portion adjusted {sign}{pct}% to match your calorie target."
    return scaled


def _day_totals(meals: dict) -> dict:
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for meal in meals.values():
        for k in totals:
            totals[k] += meal[k]
    return totals


def _rebalance_day(meals: dict, target_calories: float) -> dict:
    """Scale all meals proportionally so the day lands near the calorie target."""
    totals = _day_totals(meals)
    if totals["calories"] <= 0:
        return meals
    diff = abs(totals["calories"] - target_calories)
    if diff <= 60:
        return meals
    factor = target_calories / totals["calories"]
    factor = max(0.75, min(2.0, factor))
    rebalanced = {}
    for slot, meal in meals.items():
        rebalanced[slot] = _scale_meal(meal, factor)
    return rebalanced


def _pick_meal(candidates: list, slot_budget: float, rng, used_names: set, target_ratios: tuple) -> dict:
    if not candidates:
        raise ValueError(f"No meals available for slot with budget {slot_budget}")

    fresh = [m for m in candidates if m["name"] not in used_names]
    pool = fresh if len(fresh) >= 3 else candidates

    ranked = rank_candidates(pool, slot_budget, target_ratios, top_k=3)

    # Pick from the top matches for variety, weighted toward the best match
    choice = ranked[rng.integers(0, len(ranked))]
    used_names.add(choice["name"])

    factor = slot_budget / choice["calories"] if choice["calories"] > 0 else 1.0
    factor = max(0.85, min(1.45, factor))
    return _scale_meal(choice, factor, note=False)


def generate_weekly_plan(data: dict, target_calories: float, target_ratios: tuple) -> list:
    """Build a 7-day plan that hits calorie targets, respects dietary preferences,
    and ranks meal candidates via ML-based macro-profile similarity."""
    goal = data.get("goal", "maintain")
    family = GOAL_FAMILY.get(goal, "maintain")
    pools = _filter_meals(data, family)
    rng = np.random.default_rng(seed=_user_seed(data))
    used_names: set = set()
    plan = []

    for day in DAYS:
        day_meals = {}
        for slot in MEAL_SLOTS:
            budget = target_calories * CALORIE_SPLIT[slot]
            meal = _pick_meal(pools[slot], budget, rng, used_names, target_ratios)
            day_meals[slot] = meal

        day_meals = _rebalance_day(day_meals, target_calories)
        day_total = _day_totals(day_meals)

        plan.append({
            "day": day,
            "meals": day_meals,
            "total_calories": day_total["calories"],
            "total_protein": day_total["protein"],
            "total_carbs": day_total["carbs"],
            "total_fat": day_total["fat"],
            "target_calories": int(target_calories),
        })
    return plan


# ─── Main Analysis ────────────────────────────────────────────────────────────

def analyze_user(data: dict) -> dict:
    age = int(data["age"])
    height = float(data["height"])
    weight = float(data["weight"])
    gender = data.get("gender", "male")
    activity = data["activity"]
    goal = data["goal"]

    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity)
    target_cals = get_target_calories(tdee, goal, gender)
    body_type = classify_body(bmi, activity, age)

    if goal in ("lose_weight", "lose_fast"):
        protein_pct, carb_pct, fat_pct = 0.35, 0.40, 0.25
    elif goal in ("gain_muscle", "gain_fast"):
        protein_pct, carb_pct, fat_pct = 0.30, 0.45, 0.25
    else:
        protein_pct, carb_pct, fat_pct = 0.25, 0.45, 0.30

    macros = {
        "protein_g": round(target_cals * protein_pct / 4),
        "carbs_g": round(target_cals * carb_pct / 4),
        "fat_g": round(target_cals * fat_pct / 9),
        "protein_pct": int(protein_pct * 100),
        "carbs_pct": int(carb_pct * 100),
        "fat_pct": int(fat_pct * 100),
    }

    weekly_plan = generate_weekly_plan(data, target_cals, (protein_pct, carb_pct, fat_pct))

    metrics = {
        "bmi": bmi,
        "bmr": round(bmr),
        "tdee": int(tdee),
        "target_calories": int(target_cals),
        "macros": macros,
    }
    guidance = generate_guidance(data, {**metrics, "macros": macros})

    return {
        "user": data,
        "bmi": bmi,
        "bmr": round(bmr),
        "tdee": int(tdee),
        "target_calories": int(target_cals),
        "body_type": body_type,
        "macros": macros,
        "weekly_plan": weekly_plan,
        "guidance": guidance,
        "model_metrics": BODY_MODEL_METRICS,
    }
