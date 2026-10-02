"""
NutriAI - Intelligent Nutrition Advisor
Rule-based coaching engine that delivers personalized, evidence-informed guidance.
"""

GOAL_LABELS = {
    "lose_weight": "steady fat loss",
    "lose_fast": "accelerated fat loss",
    "maintain": "weight maintenance",
    "gain_muscle": "lean muscle gain",
    "gain_fast": "calorie surplus bulking",
}

ACTIVITY_LABELS = {
    "sedentary": "mostly sedentary",
    "light": "lightly active",
    "moderate": "moderately active",
    "active": "very active",
    "very_active": "extremely active",
}


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def _hydration_target(weight_kg: float, activity: str) -> int:
    base = weight_kg * 35
    boost = {"sedentary": 0, "light": 200, "moderate": 400, "active": 600, "very_active": 800}
    return int(round(base + boost.get(activity, 400)))


def generate_guidance(data: dict, metrics: dict) -> dict:
    """Build personalized coaching messages from user profile and calculated metrics."""
    age = int(data["age"])
    gender = data.get("gender", "male")
    activity = data["activity"]
    goal = data["goal"]
    diet = data.get("diet", "omnivore")
    bmi = metrics["bmi"]
    target_cals = metrics["target_calories"]
    macros = metrics["macros"]
    category = _bmi_category(bmi)

    goal_label = GOAL_LABELS.get(goal, "your goal")
    activity_label = ACTIVITY_LABELS.get(activity, activity)

    # ── Opening summary ─────────────────────────────────────────────────────
    if goal in ("lose_weight", "lose_fast"):
        summary = (
            f"Based on your profile ({activity_label}, BMI {bmi}), your plan targets "
            f"{target_cals:,} kcal/day for {goal_label}. Focus on protein "
            f"({macros['protein_g']}g) and whole foods — not extreme restriction."
        )
    elif goal in ("gain_muscle", "gain_fast"):
        summary = (
            f"You're set up for {goal_label} at {target_cals:,} kcal/day with "
            f"{macros['protein_g']}g protein. Consistency with meals and progressive "
            f"training matters more than any single 'superfood'."
        )
    else:
        summary = (
            f"Your maintenance target is {target_cals:,} kcal/day. With a {activity_label} "
            f"lifestyle and BMI {bmi}, the priority is balanced meals and sustainable habits."
        )

    # ── Priority actions (top 3) ────────────────────────────────────────────
    priorities = []

    if category == "underweight" and goal in ("lose_weight", "lose_fast"):
        priorities.append(
            "Your BMI is below the healthy range — avoid aggressive deficits. "
            "Consider switching to 'Maintain' and consult a healthcare provider."
        )
    elif category == "obese" and goal in ("gain_muscle", "gain_fast"):
        priorities.append(
            "With a higher BMI, prioritize strength training and protein while "
            "keeping surplus moderate to minimize unnecessary fat gain."
        )

    if goal in ("lose_weight", "lose_fast"):
        priorities.append(
            f"Hit {macros['protein_g']}g protein daily — it preserves muscle and "
            "reduces hunger during a calorie deficit."
        )
        priorities.append(
            "Fill half your plate with vegetables at lunch and dinner for volume without excess calories."
        )
    elif goal in ("gain_muscle", "gain_fast"):
        priorities.append(
            f"Spread {macros['protein_g']}g protein across 4–5 meals for better absorption and recovery."
        )
        priorities.append(
            "Eat a carb + protein meal within 1–2 hours after training to support muscle repair."
        )
    else:
        priorities.append(
            "Track intake 3–4 days per week to stay aware without obsessing over every meal."
        )
        priorities.append(
            "Keep protein near 1.6g/kg body weight to support long-term health and satiety."
        )

    if age >= 50:
        priorities.append(
            "After 50, prioritize calcium, vitamin D, and resistance training to protect bone and muscle."
        )

    if diet == "vegan":
        priorities.append(
            "Combine legumes + grains (e.g. dal + rice) and consider B12 supplementation."
        )
    elif diet == "vegetarian":
        priorities.append(
            "Use eggs, dairy, legumes, and paneer strategically to reach protein targets."
        )

    priorities = priorities[:4]

    # ── Daily habits ────────────────────────────────────────────────────────
    water_ml = _hydration_target(float(data["weight"]), activity)
    habits = [
        f"Drink ~{water_ml:,} ml water daily ({water_ml // 250} glasses).",
        "Eat at consistent meal times to regulate hunger hormones.",
        "Sleep 7–9 hours — poor sleep increases cravings and reduces willpower.",
        "Walk 20–30 minutes daily, even on rest days, to support metabolism.",
    ]

    if activity in ("active", "very_active"):
        habits.append("Schedule rest days — recovery is when muscle and hormones rebuild.")

    if goal in ("lose_weight", "lose_fast"):
        habits.append("Weigh yourself weekly (same time, same conditions) — not daily.")

    # ── Meal timing guide ───────────────────────────────────────────────────
    if goal in ("gain_muscle", "gain_fast"):
        timing = {
            "breakfast": "Within 1 hour of waking — kickstart protein synthesis.",
            "lunch": "Balanced carbs + protein to fuel afternoon activity.",
            "dinner": "Largest meal if training in evening; include carbs for recovery.",
            "snack": "Post-workout or before bed — casein-rich options aid overnight recovery.",
        }
    elif goal in ("lose_weight", "lose_fast"):
        timing = {
            "breakfast": "High protein to reduce mid-morning snacking.",
            "lunch": "Biggest meal if you're most active midday.",
            "dinner": "Lighter but still protein-rich — finish 2–3 hours before sleep.",
            "snack": "Only if genuinely hungry — choose protein + fiber.",
        }
    else:
        timing = {
            "breakfast": "Balanced start — don't skip; it stabilizes blood sugar.",
            "lunch": "Mix protein, carbs, and vegetables.",
            "dinner": "Moderate portions; avoid heavy late-night eating.",
            "snack": "Optional — fit within your daily calorie budget.",
        }

    # ── Weekly focus ────────────────────────────────────────────────────────
    weekly_focus = []
    if goal in ("lose_weight", "lose_fast"):
        weekly_focus = [
            "Week 1: Follow the plan 80% — build the habit, not perfection.",
            "Week 2: Add 10 min daily walking; notice energy and sleep changes.",
            "Week 3: Prep 2–3 meals ahead to avoid impulsive food choices.",
            "Week 4: Review progress — adjust portions if weight isn't moving as expected.",
        ]
    elif goal in ("gain_muscle", "gain_fast"):
        weekly_focus = [
            "Week 1: Hit protein target every day — track if needed.",
            "Week 2: Progressive overload in training (add reps or weight).",
            "Week 3: Ensure you're eating enough on rest days too.",
            "Week 4: Assess strength gains and adjust calories ±100–200 if needed.",
        ]
    else:
        weekly_focus = [
            "Week 1: Eat mindfully — notice hunger vs boredom eating.",
            "Week 2: Include 5 different colored vegetables across the week.",
            "Week 3: Limit ultra-processed snacks to 2–3 times.",
            "Week 4: Maintain consistency — sustainability beats perfection.",
        ]

    # ── Warnings / safety ───────────────────────────────────────────────────
    warnings = []
    if target_cals <= 1300 and gender == "female":
        warnings.append(
            "Your calorie target is quite low. Ensure adequate protein and micronutrients; "
            "consult a dietitian if you feel fatigued or unwell."
        )
    if target_cals <= 1500 and gender == "male" and goal in ("lose_weight", "lose_fast"):
        warnings.append(
            "Very low calorie targets can slow metabolism. Monitor energy levels closely."
        )
    if category == "underweight":
        warnings.append(
            "BMI below 18.5 may indicate undernutrition. Professional guidance is recommended."
        )
    if category == "obese" and goal in ("lose_weight", "lose_fast"):
        warnings.append(
            "Gradual loss (0.5–1 kg/week) is safer and more sustainable than rapid cuts."
        )

    return {
        "summary": summary,
        "priorities": priorities,
        "daily_habits": habits,
        "meal_timing": timing,
        "weekly_focus": weekly_focus,
        "hydration_ml": water_ml,
        "warnings": warnings,
        "health_score": _health_score(category, activity, age, goal),
    }


def _health_score(category: str, activity: str, age: int, goal: str) -> dict:
    """Simple wellness score for dashboard display (educational, not medical)."""
    score = 70
    if category == "normal":
        score += 15
    elif category == "overweight":
        score += 5
    elif category == "underweight":
        score += 0
    else:
        score -= 5

    activity_bonus = {"sedentary": 0, "light": 5, "moderate": 10, "active": 15, "very_active": 18}
    score += activity_bonus.get(activity, 5)

    if goal == "maintain":
        score += 5
    if age < 30:
        score += 5
    elif age > 60:
        score -= 3

    score = max(40, min(98, score))
    label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 55 else "Needs attention"
    return {"score": score, "label": label}
