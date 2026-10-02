# NutriAI — Intelligent Nutrition Advisor

NutriAI is a Flask-based nutrition planning application that combines calorie calculations, machine learning, meal recommendations, nutrition guidance, progress forecasting, and downloadable PDF reports.

## Features

- **Personalized calorie analysis:** Estimates Basal Metabolic Rate (BMR) using the Mifflin–St Jeor equation and calculates Total Daily Energy Expenditure (TDEE) using activity level and goal.
- **Body classification:** Uses a Random Forest model to classify body-related categories from user information.
- **Smart meal planning:** Recommends meals based on dietary preference, allergies, cuisine, and calorie targets.
- **Nutrition advisor:** Provides practical nutrition guidance and daily habit suggestions.
- **Diet Q&A:** Uses a text-classification model to identify question intent and generate a relevant response.
- **Progress forecasting:** Estimates future weight trends from logged entries.
- **PDF reports:** Generates a downloadable nutrition report.

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **PDF Generation:** ReportLab

## Project Structure

```text
Diet_ai/
├── app.py                  # Flask application and API routes
├── body_classifier.py      # Body classification model
├── food_database.py        # Meal and food data
├── meal_recommender.py     # Meal recommendation logic
├── model.py                # Main nutrition analysis
├── nlp_advisor.py          # Diet question intent classification
├── nutrition_advisor.py    # Nutrition guidance
├── pdf_report.py           # PDF report generation
├── progress_forecast.py    # Progress trend forecasting
├── requirements.txt        # Python dependencies
└── templates/
    └── index.html          # Web interface
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ALi-Hassan79/Diet_ai.git
cd Diet_ai
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bat
py -3.12 -m venv venv
venv\Scripts\activate
```

If Python 3.12 is not installed, use a supported Python version available on your system.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Loads the web interface |
| `/analyze` | POST | Analyzes a user's nutrition profile |
| `/forecast` | POST | Forecasts progress from weight entries |
| `/ask` | POST | Answers a nutrition-related question |
| `/model-metrics` | GET | Returns body-classifier evaluation metrics |
| `/nlp-metrics` | GET | Returns NLP intent-model evaluation metrics |
| `/download-pdf` | POST | Creates a downloadable PDF report |

## Meal Planning

The meal recommendation system considers dietary preferences, allergies, and cuisine choices. It allocates the daily calorie target across meals and adjusts portions to better match the target.

## Disclaimer

NutriAI is intended for educational and informational purposes only. It is not a substitute for medical advice, diagnosis, or treatment. Consult a qualified healthcare professional or registered dietitian before making significant dietary changes.

## License

No license has been specified for this project. Add a `LICENSE` file if you want to make the project available for reuse.
