"""
Adult Income Classifier - Backend API
======================================
Stateless FastAPI service. Loads the trained preprocessor + Decision Tree
model (adult_income_model.pkl) once at startup, exposes a /predict endpoint
that accepts raw human-readable form fields, reconstructs the exact feature
matrix the model was trained on, and returns a prediction.

No database. No file/session storage. Every request is processed independently.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then POST to http://localhost:8000/predict
"""

import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model artifact once at process startup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "adult_income_model.pkl"

with open(MODEL_PATH, "rb") as f:
    _artifact = pickle.load(f)

PREPROCESSOR = _artifact["preprocessor"]
MODEL = _artifact["model"]
TRAINED_SPLIT = _artifact.get("split", "unknown")

# ---------------------------------------------------------------------------
# Feature schema — mirrors the notebook's encoding exactly
# ---------------------------------------------------------------------------
CONT_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

RELATIONSHIP_VALUES = [
    "Husband", "Wife", "Not-in-family",
    "Own-child", "Unmarried", "Other-relative",
]
MARITAL_STATUS_VALUES = [
    "Married-civ-spouse", "Never-married", "Divorced",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]
OCCUPATION_VALUES = [
    "Exec-managerial", "Prof-specialty", "Craft-repair", "Adm-clerical",
    "Sales", "Other-service", "Machine-op-inspct", "Transport-moving",
    "Handlers-cleaners", "Farming-fishing", "Tech-support",
    "Protective-serv", "Priv-house-serv", "Armed-Forces",
]
WORKCLASS_VALUES = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
]

# Exact column order the ColumnTransformer/model was fitted on
FEATURE_ORDER = (
    CONT_COLS
    + ["sex"]
    + [f"relationship_{v}" for v in RELATIONSHIP_VALUES]
    + [f"marital-status_{v}" for v in MARITAL_STATUS_VALUES]
    + [f"occupation_{v}" for v in OCCUPATION_VALUES]
    + [f"workclass_{v}" for v in WORKCLASS_VALUES]
)

# Education label -> education-num, taken from the standard Adult dataset mapping
EDUCATION_NUM_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

# Winsorisation caps used during training (99th percentile of the combined
# train+test data, per the notebook's preprocessing cell). Applied here too so
# extreme user-entered values are capped the same way the model expects.
CAPITAL_GAIN_CAP = 15024.0
CAPITAL_LOSS_CAP = 2001.0


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=17, le=90, description="Age in years")
    workclass: Literal[tuple(WORKCLASS_VALUES)] = Field(..., description="Employment category")
    education: Literal[tuple(EDUCATION_NUM_MAP.keys())] = Field(..., description="Highest education level")
    marital_status: Literal[tuple(MARITAL_STATUS_VALUES)] = Field(..., alias="maritalStatus")
    occupation: Literal[tuple(OCCUPATION_VALUES)]
    relationship: Literal[tuple(RELATIONSHIP_VALUES)]
    sex: Literal["Male", "Female"]
    capital_gain: float = Field(0, ge=0, alias="capitalGain")
    capital_loss: float = Field(0, ge=0, alias="capitalLoss")
    hours_per_week: int = Field(..., ge=1, le=99, alias="hoursPerWeek")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction: Literal["<=50K", ">50K"]
    prediction_label: int
    probability_above_50k: float
    probability_at_or_below_50k: float
    model_trained_on_split: str


# ---------------------------------------------------------------------------
# Feature engineering — replicates the notebook's preprocessing pipeline
# ---------------------------------------------------------------------------
def build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_ORDER}

    row["age"] = req.age
    row["education-num"] = EDUCATION_NUM_MAP[req.education]
    row["capital-gain"] = min(req.capital_gain, CAPITAL_GAIN_CAP)
    row["capital-loss"] = min(req.capital_loss, CAPITAL_LOSS_CAP)
    row["hours-per-week"] = req.hours_per_week

    # LabelEncoder on ['Female', 'Male'] (alphabetical) -> Female=0, Male=1
    row["sex"] = 1 if req.sex == "Male" else 0

    row[f"relationship_{req.relationship}"] = 1
    row[f"marital-status_{req.marital_status}"] = 1
    row[f"occupation_{req.occupation}"] = 1
    row[f"workclass_{req.workclass}"] = 1

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Adult Income Classifier API",
    description="Stateless prediction API for the CCS23213 Adult Income Decision Tree model.",
    version="1.0.0",
)

# Wide-open CORS since this is a public-facing demo API called from a separate
# frontend (Streamlit / Vercel). Tighten allow_origins for production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "adult-income-classifier",
        "status": "ok",
        "model_trained_on_split": TRAINED_SPLIT,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/schema")
def schema():
    """Lets a frontend build its form dynamically instead of hardcoding options."""
    return {
        "workclass": WORKCLASS_VALUES,
        "education": list(EDUCATION_NUM_MAP.keys()),
        "marital_status": MARITAL_STATUS_VALUES,
        "occupation": OCCUPATION_VALUES,
        "relationship": RELATIONSHIP_VALUES,
        "sex": ["Male", "Female"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        X = build_feature_row(req)
        X_transformed = PREPROCESSOR.transform(X)

        pred = int(MODEL.predict(X_transformed)[0])
        proba = MODEL.predict_proba(X_transformed)[0]  # [P(<=50K), P(>50K)]

        return PredictionResponse(
            prediction=">50K" if pred == 1 else "<=50K",
            prediction_label=pred,
            probability_above_50k=round(float(proba[1]), 4),
            probability_at_or_below_50k=round(float(proba[0]), 4),
            model_trained_on_split=TRAINED_SPLIT,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
