"""Shared paths, feature lists and the sklearn preprocessing pipeline."""

from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# One seed for every random operation so results are reproducible.
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
ASSETS_DIR = ROOT / "assets"
SQL_DIR = ROOT / "sql"

RAW_CSV = DATA_DIR / "telco_customer_churn.csv"
DB_PATH = DATA_DIR / "churn.db"
QUERIES_PATH = SQL_DIR / "queries.sql"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

ID_COL = "customerID"
TARGET = "churn_flag"

# gender is deliberately excluded: it is not actionable and carries no useful signal.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    """Create every output directory the project writes to."""
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR, TABLES_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a query against the project database and return a DataFrame."""
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def build_preprocessor() -> ColumnTransformer:
    """Scale numbers and one-hot encode categoricals, unknown categories ignored."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_pipeline(estimator) -> Pipeline:
    """Pair the shared preprocessing with a classifier."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def load_customers() -> pd.DataFrame:
    """Load the cleaned customer table."""
    return read_sql("SELECT * FROM customers")
