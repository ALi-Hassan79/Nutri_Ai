"""
NutriAI - Body Composition Classifier
A genuine ML component: a RandomForestClassifier trained on a realistically
simulated population, evaluated with a proper train/test split, k-fold
cross-validation, and standard classification metrics.

Why this isn't just "if bmi < 25" wearing an ML costume:
The old version generated bmi values and then derived the label with the exact
same threshold the classifier was supposed to learn, so the model could only
ever be trivially "perfect" and added nothing over the rule itself. Here, the
label depends on an *effective* body-composition score that BMI only partially
predicts - activity level shifts it (an athletic build reads lower than BMI
alone would suggest), age shifts it slightly, and each person gets measurement
/lifestyle noise. The model has to learn a real, non-trivial decision boundary
from three correlated-but-imperfect features, which is what makes evaluating
it (accuracy, F1, confusion matrix) a meaningful exercise instead of a formality.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BODY_LABELS = ["Underweight", "Normal Weight", "Overweight", "Obese"]
BODY_EMOJIS = ["⚠️", "✅", "🔶", "🔴"]

RANDOM_STATE = 42


def _generate_population(n: int, seed: int = RANDOM_STATE):
    """Simulate a population where the label is a *noisy* function of BMI,
    activity, and age - not a clean threshold on BMI alone."""
    rng = np.random.default_rng(seed)

    ages = rng.integers(16, 75, n)
    activities = rng.integers(1, 6, n)          # 1 sedentary .. 5 very active
    bmis = np.clip(rng.normal(24.5, 6.0, n), 14, 48)

    # Higher activity -> more lean mass -> "true" body-fat category reads
    # lower than raw BMI suggests. Effect is real but noisy (not everyone
    # active is lean), and older age nudges the other way slightly.
    activity_effect = (activities - 3) * rng.normal(0.9, 0.35, n)
    age_effect = np.where(ages > 50, rng.normal(0.6, 0.4, n), 0.0)
    measurement_noise = rng.normal(0, 1.3, n)

    effective_score = bmis - activity_effect + age_effect + measurement_noise

    labels = np.select(
        [effective_score < 18.5, effective_score < 25.0, effective_score < 30.0],
        [0, 1, 2],
        default=3,
    )

    X = np.column_stack([bmis, activities, ages]).astype(float)
    y = labels.astype(int)
    return X, y


def _train_and_evaluate():
    X, y = _generate_population(n=6000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    importances = model.feature_importances_

    metrics = {
        "test_accuracy": round(float(accuracy) * 100, 1),
        "macro_precision": round(float(precision) * 100, 1),
        "macro_recall": round(float(recall) * 100, 1),
        "macro_f1": round(float(f1) * 100, 1),
        "cv_accuracy_mean": round(float(cv_scores.mean()) * 100, 1),
        "cv_accuracy_std": round(float(cv_scores.std()) * 100, 1),
        "confusion_matrix": cm.tolist(),
        "labels": BODY_LABELS,
        "feature_importance": {
            "bmi": round(float(importances[0]) * 100, 1),
            "activity": round(float(importances[1]) * 100, 1),
            "age": round(float(importances[2]) * 100, 1),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_estimators": model.n_estimators,
    }

    # Refit on the full dataset for the production model - standard practice
    # once the held-out test has already told us how well it generalizes.
    final_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    )
    final_model.fit(X, y)

    return final_model, metrics


BODY_CLASSIFIER, BODY_MODEL_METRICS = _train_and_evaluate()

ACTIVITY_SCORES = {
    "sedentary": 1,
    "light": 2,
    "moderate": 3,
    "active": 4,
    "very_active": 5,
}


def classify_body(bmi: float, activity_level: str, age: int) -> dict:
    activity_score = ACTIVITY_SCORES.get(activity_level, 3)
    X = np.array([[bmi, activity_score, age]])
    label_idx = int(BODY_CLASSIFIER.predict(X)[0])
    proba = BODY_CLASSIFIER.predict_proba(X)[0]
    confidence = round(float(proba[label_idx]) * 100, 1)
    return {
        "label": BODY_LABELS[label_idx],
        "emoji": BODY_EMOJIS[label_idx],
        "confidence": confidence,
        "index": label_idx,
    }
