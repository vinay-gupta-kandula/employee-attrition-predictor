"""
Data preprocessing pipeline and transformation utilities for Employee Attrition Prediction.
"""

from typing import List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Uninformative, constant, or identifier columns to exclude
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
TARGET_COLUMNS = ["Attrition", "Attrition_Yes"]

# Canonical feature lists
NUMERICAL_FEATURES: List[str] = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_FEATURES: List[str] = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove target and non-informative ID / constant columns if present in data.
    """
    cols_to_drop = [c for c in DROP_COLUMNS + TARGET_COLUMNS if c in data.columns]
    return data.drop(columns=cols_to_drop)


def build_preprocessing_pipeline(
    data: Optional[pd.DataFrame] = None,
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for one-hot encoding categorical features
    and standard-scaling numerical features.

    If data is provided, column names are dynamically derived (excluding target
    and drop columns). Otherwise, canonical lists are used.
    """
    if data is not None:
        cleaned = clean_dataframe(data)
        num_cols = cleaned.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = cleaned.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        num_cols = list(NUMERICAL_FEATURES)
        cat_cols = list(CATEGORICAL_FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve clean feature names from a fitted ColumnTransformer.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
        # Clean prefix if scikit-learn added any
        cleaned = []
        for name in names:
            if name.startswith("num__") or name.startswith("cat__"):
                cleaned.append(name.split("__", 1)[1])
            else:
                cleaned.append(name)
        return cleaned
    raise ValueError("Preprocessor has not been fitted or feature names are unavailable.")
