"""
Adult Income Classifier - Backend API
======================================
Stateless FastAPI service. Loads the trained preprocessor + Decision Tree
model (adult_income_model.pkl) once at startup, exposes a /predict endpoint
that accepts raw human-readable form fields, reconstructs the exact feature
matrix the model was trained on, and returns a prediction.

Everything the API needs to rebuild the feature row (column order, category
values, winsorisation caps, education mapping) is read from the model
artifact written by the notebook, so retraining never requires editing this
file by hand.

No database. No file/session storage. Every request is processed independently.

Run locally (Python 3.11 or 3.12):
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then POST to http://localhost:8000/predict
"""

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
logger = logging.getLogger("adult-income-api")

# ---------------------------------------------------------------------------
# Load model artifact once at process startup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "adult_income_model.pkl"
REQUIRED_KEYS = {"preprocessor", "model", "winsor_caps", "education_num_map"}

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11+ is required (scikit-learn 1.8 does not support older versions).")

with open(MODEL_PATH, "rb") as f:
    # Turn scikit-learn's "unpickling estimator from version X" warning into a
    # hard error: a mismatched version can load fine yet fail on every predict.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=getattr(sklearn.exceptions, "InconsistentVersionWarning", UserWarning))
        _artifact = pickle.load(f)

_missing = REQUIRED_KEYS - set(_artifact)
if _missing:
    raise RuntimeError(
        f"{MODEL_PATH.name} is missing {sorted(_missing)}. "
        "Re-run the final cell of 'machine learning phase/ML_model_notebook.ipynb' to regenerate it."
    )

PREPROCESSOR = _artifact["preprocessor"]
MODEL = _artifact["model"]
TRAINED_SPLIT = _artifact.get("split", "unknown")
WINSOR_CAPS: dict[str, float] = _artifact["winsor_caps"]
EDUCATION_NUM_MAP: dict[str, int] = _artifact["education_num_map"]

logger.info(
    "Loaded model (trained with scikit-learn %s, running %s, split %s)",
    _artifact.get("sklearn_version", "unknown"), sklearn.__version__, TRAINED_SPLIT,
)

# ---------------------------------------------------------------------------
# Feature schema — derived from the columns the preprocessor was fitted on
# ---------------------------------------------------------------------------
FEATURE_ORDER: list[str] = list(PREPROCESSOR.feature_names_in_)


def _categories(prefix: str) -> list[str]:
    return [c[len(prefix):] for c in FEATURE_ORDER if c.startswith(prefix)]


RELATIONSHIP_VALUES = _categories("relationship_")
MARITAL_STATUS_VALUES = _categories("marital-status_")
OCCUPATION_VALUES = _categories("occupation_")
WORKCLASS_VALUES = _categories("workclass_")
EDUCATION_VALUES = sorted(EDUCATION_NUM_MAP, key=EDUCATION_NUM_MAP.get)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., ge=17, le=90, description="Age in years")
    workclass: Literal[tuple(WORKCLASS_VALUES)] = Field(..., description="Employment category")
    education: Literal[tuple(EDUCATION_VALUES)] = Field(..., description="Highest education level")
    marital_status: Literal[tuple(MARITAL_STATUS_VALUES)] = Field(..., alias="maritalStatus")
    occupation: Literal[tuple(OCCUPATION_VALUES)]
    relationship: Literal[tuple(RELATIONSHIP_VALUES)]
    sex: Literal["Male", "Female"]
    capital_gain: float = Field(0, ge=0, alias="capitalGain")
    capital_loss: float = Field(0, ge=0, alias="capitalLoss")
    hours_per_week: int = Field(..., ge=1, le=99, alias="hoursPerWeek")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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
    row["capital-gain"] = min(req.capital_gain, WINSOR_CAPS["capital-gain"])
    row["capital-loss"] = min(req.capital_loss, WINSOR_CAPS["capital-loss"])
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
    version="1.1.0",
)

# Wide-open CORS since this is a public-facing demo API called from a separate
# static frontend. Tighten allow_origins for production use.
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
        "education": EDUCATION_VALUES,
        "marital_status": MARITAL_STATUS_VALUES,
        "occupation": OCCUPATION_VALUES,
        "relationship": RELATIONSHIP_VALUES,
        "sex": ["Male", "Female"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Input validation already happened in Pydantic (invalid input -> 422).
    # Anything failing past this point is a server-side problem, not a bad request.
    try:
        X_transformed = PREPROCESSOR.transform(build_feature_row(req))
        proba = MODEL.predict_proba(X_transformed)[0]  # [P(<=50K), P(>50K)]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

    pred = int(proba[1] > proba[0])
    return PredictionResponse(
        prediction=">50K" if pred == 1 else "<=50K",
        prediction_label=pred,
        probability_above_50k=round(float(proba[1]), 4),
        probability_at_or_below_50k=round(float(proba[0]), 4),
        model_trained_on_split=TRAINED_SPLIT,
    )
