"""
Diet AI - PDF Report Generator
Generates a professional PDF diet report using ReportLab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io


# ─── Color Palette ───────────────────────────────────────────────────────────

GREEN       = colors.HexColor("#22c55e")
DARK_GREEN  = colors.HexColor("#16a34a")
LIGHT_GREEN = colors.HexColor("#dcfce7")
GRAY_900    = colors.HexColor("#111827")
GRAY_600    = colors.HexColor("#4b5563")
GRAY_100    = colors.HexColor("#f3f4f6")
WHITE       = colors.white
ACCENT      = colors.HexColor("#0ea5e9")
ORANGE      = colors.HexColor("#f97316")


# ─── Style Setup ─────────────────────────────────────────────────────────────

def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "cover_title": ParagraphStyle("cover_title",
            fontSize=32, leading=40, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "cover_sub": ParagraphStyle("cover_sub",
            fontSize=14, leading=20, textColor=LIGHT_GREEN,
            fontName="Helvetica", alignment=TA_CENTER),
        "section_head": ParagraphStyle("section_head",
            fontSize=14, leading=18, textColor=DARK_GREEN,
            fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body",
            fontSize=10, leading=14, textColor=GRAY_600,
            fontName="Helvetica"),
        "label": ParagraphStyle("label",
            fontSize=9, leading=12, textColor=GRAY_600,
            fontName="Helvetica"),
        "metric_val": ParagraphStyle("metric_val",
            fontSize=22, leading=26, textColor=GRAY_900,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "metric_lbl": ParagraphStyle("metric_lbl",
            fontSize=8, leading=10, textColor=GRAY_600,
            fontName="Helvetica", alignment=TA_CENTER),
        "day_head": ParagraphStyle("day_head",
            fontSize=11, leading=14, textColor=DARK_GREEN,
            fontName="Helvetica-Bold"),
        "meal_name": ParagraphStyle("meal_name",
            fontSize=9, leading=12, textColor=GRAY_900,
            fontName="Helvetica-Bold"),
        "small": ParagraphStyle("small",
            fontSize=8, leading=10, textColor=GRAY_600,
            fontName="Helvetica"),
        "footer": ParagraphStyle("footer",
            fontSize=8, leading=10, textColor=GRAY_600,
            fontName="Helvetica", alignment=TA_CENTER),
    }
    return custom


# ─── Helper: Stat Card Table ─────────────────────────────────────────────────

def stat_cards(metrics: list, styles: dict):
    """metrics = [(value, label, color), ...]"""
    cells = []
    for val, lbl, col in metrics:
        cell = [
            Paragraph(str(val), ParagraphStyle("v", fontSize=20, leading=24,
                textColor=col, fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph(lbl, styles["metric_lbl"]),
        ]
        cells.append(cell)

    # Flatten to table row
    row_vals = [Paragraph(str(v), ParagraphStyle("v", fontSize=20, leading=24,
        textColor=c, fontName="Helvetica-Bold", alignment=TA_CENTER)) for v, _, c in metrics]
    row_lbls = [Paragraph(l, styles["metric_lbl"]) for _, l, _ in metrics]

    t = Table([row_vals, row_lbls], colWidths=[4.2 * cm] * len(metrics))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GRAY_100),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [GRAY_100, GRAY_100]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


# ─── Main PDF Builder ─────────────────────────────────────────────────────────

def generate_pdf(result: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = get_styles()
    S = []  # Story

    user     = result["user"]
    macros   = result["macros"]
    body_t   = result["body_type"]
    plan     = result["weekly_plan"]
    guidance = result.get("guidance", {})

    GOAL_LABELS = {
        "lose_weight": "Lose Weight",
        "lose_fast": "Lose Weight Fast",
        "maintain": "Maintain Weight",
        "gain_muscle": "Build Muscle",
        "gain_fast": "Bulk Up Fast",
    }
    ACTIVITY_LABELS = {
        "sedentary": "Sedentary",
        "light": "Lightly Active",
        "moderate": "Moderately Active",
        "active": "Very Active",
        "very_active": "Extremely Active",
    }

    # ── Cover Banner ─────────────────────────────────────────────────────────
    banner_data = [
        [Paragraph("🥗 NutriAI Diet Report", styles["cover_title"])],
        [Paragraph(f"Generated for {user.get('name', 'User')} · {datetime.now().strftime('%B %d, %Y')}", styles["cover_sub"])],
    ]
    banner = Table(banner_data, colWidths=[17 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    S.append(banner)
    S.append(Spacer(1, 18))

    # ── User Profile ─────────────────────────────────────────────────────────
    S.append(Paragraph("📋 Your Profile", styles["section_head"]))
    S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=8))

    profile_data = [
        ["Age", f"{user['age']} years", "Height", f"{user['height']} cm"],
        ["Weight", f"{user['weight']} kg", "Gender", user.get("gender", "—").title()],
        ["Activity Level", ACTIVITY_LABELS.get(user["activity"], user["activity"]),
         "Goal", GOAL_LABELS.get(user["goal"], user["goal"])],
    ]
    pt = Table(profile_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY_600),
        ("TEXTCOLOR", (2, 0), (2, -1), GRAY_600),
        ("TEXTCOLOR", (1, 0), (1, -1), GRAY_900),
        ("TEXTCOLOR", (3, 0), (3, -1), GRAY_900),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, GRAY_100]),
    ]))
    S.append(pt)
    S.append(Spacer(1, 16))

    # ── Key Metrics ───────────────────────────────────────────────────────────
    S.append(Paragraph("📊 Key Metrics", styles["section_head"]))
    S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=8))

    bmi_color = GREEN if result["body_type"]["index"] == 1 else ORANGE
    metrics = [
        (result["bmi"], "BMI", bmi_color),
        (result["bmr"], "BMR (kcal/day)", ACCENT),
        (result["tdee"], "TDEE (kcal/day)", ACCENT),
        (result["target_calories"], "Target Calories", DARK_GREEN),
    ]
    S.append(stat_cards(metrics, styles))
    S.append(Spacer(1, 10))

    # Body type badge
    bt_color = [GREEN, DARK_GREEN, ORANGE, colors.HexColor("#ef4444")][body_t["index"]]
    badge_data = [[
        Paragraph(f"{body_t['emoji']}  Body Composition: {body_t['label']}   "
                  f"(ML Model Confidence: {body_t['confidence']}%)",
                  ParagraphStyle("badge", fontSize=10, leading=14,
                                 textColor=bt_color, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER))
    ]]
    badge_t = Table(badge_data, colWidths=[17 * cm])
    badge_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
        ("BOX", (0, 0), (-1, -1), 1, bt_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    S.append(badge_t)
    S.append(Spacer(1, 16))

    # ── Macro Targets ─────────────────────────────────────────────────────────
    S.append(Paragraph("⚖️ Daily Macro Targets", styles["section_head"]))
    S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=8))

    macro_data = [
        ["Macronutrient", "Grams / Day", "% of Calories", "Calories"],
        ["🥩 Protein", f"{macros['protein_g']}g", f"{macros['protein_pct']}%",
         f"{macros['protein_g'] * 4} kcal"],
        ["🍞 Carbohydrates", f"{macros['carbs_g']}g", f"{macros['carbs_pct']}%",
         f"{macros['carbs_g'] * 4} kcal"],
        ["🥑 Fats", f"{macros['fat_g']}g", f"{macros['fat_pct']}%",
         f"{macros['fat_g'] * 9} kcal"],
        ["TOTAL", "—", "100%", f"{result['target_calories']} kcal"],
    ]
    mt = Table(macro_data, colWidths=[5*cm, 4*cm, 4*cm, 4*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT_GREEN),
        ("TEXTCOLOR", (0, -1), (-1, -1), DARK_GREEN),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [WHITE, GRAY_100]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (0, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
    ]))
    S.append(mt)
    S.append(Spacer(1, 16))

    # ── Personalized Guidance ─────────────────────────────────────────────────
    if guidance:
        S.append(Paragraph("💡 Your Nutrition Coach", styles["section_head"]))
        S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=8))
        S.append(Paragraph(guidance.get("summary", ""), styles["body"]))
        S.append(Spacer(1, 8))

        hs = guidance.get("health_score", {})
        if hs:
            S.append(Paragraph(
                f"Wellness Score: {hs.get('score', '—')}/100 ({hs.get('label', '')}) · "
                f"Hydration: {guidance.get('hydration_ml', 0):,} ml/day",
                ParagraphStyle("hs", fontSize=9, leading=12, textColor=DARK_GREEN, fontName="Helvetica-Bold")
            ))
            S.append(Spacer(1, 6))

        S.append(Paragraph("Top Priorities", ParagraphStyle("sub", fontSize=10, leading=14,
            textColor=GRAY_900, fontName="Helvetica-Bold", spaceBefore=6)))
        for p in guidance.get("priorities", []):
            S.append(Paragraph(f"• {p}", styles["body"]))
            S.append(Spacer(1, 2))

        S.append(Paragraph("Daily Habits", ParagraphStyle("sub2", fontSize=10, leading=14,
            textColor=GRAY_900, fontName="Helvetica-Bold", spaceBefore=8)))
        for h in guidance.get("daily_habits", []):
            S.append(Paragraph(f"• {h}", styles["body"]))
            S.append(Spacer(1, 2))

        for w in guidance.get("warnings", []):
            S.append(Paragraph(f"⚠ {w}", ParagraphStyle("warn", fontSize=9, leading=12,
                textColor=ORANGE, fontName="Helvetica")))

    S.append(PageBreak())

    # ── Weekly Meal Plan ──────────────────────────────────────────────────────
    S.append(Paragraph("📅 7-Day Personalized Meal Plan", styles["section_head"]))
    S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=10))

    MEAL_ICONS = {"breakfast": "🌅", "lunch": "☀️", "dinner": "🌙", "snack": "🍎"}

    for day_data in plan:
        day = day_data["day"]
        meals = day_data["meals"]
        total = day_data["total_calories"]
        target = day_data["target_calories"]
        diff = total - target

        # Day header
        day_header = Table([[
            Paragraph(f"📆 {day}", styles["day_head"]),
            Paragraph(f"{total} kcal  {'▲' if diff > 0 else '▼'} {abs(diff)} vs target",
                      ParagraphStyle("dh_r", fontSize=9, leading=12,
                                     textColor=ORANGE if abs(diff) > 100 else GREEN,
                                     fontName="Helvetica-Bold", alignment=TA_RIGHT)),
        ]], colWidths=[9*cm, 8*cm])
        day_header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), GRAY_100),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        S.append(day_header)

        # Meals row
        meal_rows = []
        for meal_type, meal in meals.items():
            icon = MEAL_ICONS.get(meal_type, "🍽️")
            macro_str = (f"P:{meal['protein']}g  C:{meal['carbs']}g  F:{meal['fat']}g")
            tip = meal.get("tip", "")
            portion = meal.get("portion_note", "")
            detail = meal["name"]
            if tip:
                detail += f"<br/><font size='7' color='#16a34a'>{tip}</font>"
            if portion:
                detail += f"<br/><font size='7' color='#f97316'>{portion}</font>"
            meal_rows.append([
                Paragraph(f"{icon} {meal_type.title()}", styles["label"]),
                Paragraph(detail, styles["meal_name"]),
                Paragraph(f"{meal['calories']} kcal", ParagraphStyle("cal",
                    fontSize=9, leading=12, textColor=ACCENT, fontName="Helvetica-Bold")),
                Paragraph(macro_str, styles["small"]),
            ])

        meals_t = Table(meal_rows, colWidths=[2.5*cm, 6.5*cm, 2.5*cm, 5.5*cm])
        meals_t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor("#f9fafb")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f3f4f6")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        S.append(meals_t)
        S.append(Spacer(1, 6))

    # ── Tips ─────────────────────────────────────────────────────────────────
    S.append(Spacer(1, 10))
    S.append(Paragraph("💡 Nutrition Tips", styles["section_head"]))
    S.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceAfter=8))

    GOAL_TIPS = {
        "lose_weight": [
            "Drink 8-10 glasses of water daily — it boosts metabolism by up to 30%.",
            "Eat slowly and mindfully; it takes 20 minutes for your brain to feel full.",
            "Prioritise protein at every meal to preserve muscle while in a calorie deficit.",
            "Avoid liquid calories — sodas, juices, and alcohol add up fast.",
        ],
        "maintain": [
            "Track your calories 3-4 days a week to stay aware without obsessing.",
            "Balance treats within your weekly calorie budget, not daily.",
            "Sleep 7-9 hours — poor sleep increases hunger hormones significantly.",
        ],
        "gain_muscle": [
            "Eat within 30-60 minutes post-workout to maximise muscle protein synthesis.",
            "Distribute protein evenly across 4-5 meals for optimal absorption.",
            "Progressive overload in training combined with sufficient calories = growth.",
            "Creatine monohydrate is the most evidence-backed muscle supplement.",
        ],
    }
    from model import GOAL_FAMILY
    tips = GOAL_TIPS.get(GOAL_FAMILY.get(user["goal"], "maintain"), [])
    for tip in tips:
        S.append(Paragraph(f"• {tip}", styles["body"]))
        S.append(Spacer(1, 3))

    # ── Footer ────────────────────────────────────────────────────────────────
    S.append(Spacer(1, 20))
    S.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    S.append(Spacer(1, 6))
    S.append(Paragraph(
        "This report is generated by NutriAI and is for informational purposes only. "
        "Consult a registered dietitian or physician before making significant dietary changes.",
        styles["footer"]
    ))

    doc.build(S)
    buf.seek(0)
    return buf.read()
