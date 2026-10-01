"""
Unit tests for data preprocessing module (app/preprocessing.py).
Tests data cleaning, column transformations, pipeline serialization compatibility,
and handling of known/unknown categorical levels.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from app.preprocessing import (
    clean_dataframe,
    build_preprocessing_pipeline,
    get_feature_names,
    DROP_COLUMNS,
    TARGET_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
)


@pytest.fixture
def sample_raw_dataframe():
    """
    Constructs a minimal DataFrame reflecting the IBM HR dataset schema.
    """
    data = {
        "Age": [30, 45, 25],
        "Attrition": ["Yes", "No", "No"],
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        "DailyRate": [800, 1200, 500],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "DistanceFromHome": [5, 10, 2],
        "Education": [3, 4, 2],
        "EducationField": ["Life Sciences", "Medical", "Technical Degree"],
        "EmployeeCount": [1, 1, 1],
        "EmployeeNumber": [101, 102, 103],
        "EnvironmentSatisfaction": [3, 2, 4],
        "Gender": ["Male", "Female", "Male"],
        "HourlyRate": [65, 80, 50],
        "JobInvolvement": [3, 2, 4],
        "JobLevel": [2, 3, 1],
        "JobRole": ["Sales Executive", "Research Scientist", "Human Resources"],
        "JobSatisfaction": [3, 4, 2],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "MonthlyIncome": [5000, 8500, 3000],
        "MonthlyRate": [15000, 20000, 10000],
        "NumCompaniesWorked": [2, 4, 1],
        "Over18": ["Y", "Y", "Y"],
        "OverTime": ["Yes", "No", "No"],
        "PercentSalaryHike": [12, 15, 11],
        "PerformanceRating": [3, 3, 3],
        "RelationshipSatisfaction": [3, 4, 2],
        "StandardHours": [80, 80, 80],
        "StockOptionLevel": [0, 1, 2],
        "TotalWorkingYears": [8, 15, 3],
        "TrainingTimesLastYear": [2, 3, 2],
        "WorkLifeBalance": [3, 2, 4],
        "YearsAtCompany": [5, 10, 2],
        "YearsInCurrentRole": [3, 7, 1],
        "YearsSinceLastPromotion": [1, 3, 0],
        "YearsWithCurrManager": [3, 7, 1],
    }
    return pd.DataFrame(data)


def test_clean_dataframe_drops_uninformative_columns(sample_raw_dataframe):
    """
    Verifies that EmployeeCount, EmployeeNumber, Over18, StandardHours,
    and target columns are dropped by clean_dataframe.
    """
    cleaned = clean_dataframe(sample_raw_dataframe)

    for col in DROP_COLUMNS:
        assert col not in cleaned.columns, f"Column {col} should have been dropped"

    for col in TARGET_COLUMNS:
        assert col not in cleaned.columns, f"Target column {col} should have been dropped"

    # Verify that standard feature columns remain
    assert "Age" in cleaned.columns
    assert "MonthlyIncome" in cleaned.columns
    assert "OverTime" in cleaned.columns
    assert "Department" in cleaned.columns
    assert cleaned.shape[1] == len(ALL_FEATURE_COLUMNS)


def test_clean_dataframe_idempotence():
    """
    Verifies that calling clean_dataframe on already cleaned data is a safe no-op.
    """
    df = pd.DataFrame({"Age": [30], "DailyRate": [1000], "Department": ["Sales"]})
    cleaned = clean_dataframe(df)
    assert list(cleaned.columns) == ["Age", "DailyRate", "Department"]


def test_build_preprocessing_pipeline_structure():
    """
    Verifies that the returned pipeline is a ColumnTransformer with both
    numerical and categorical transformers.
    """
    pipeline = build_preprocessing_pipeline()
    assert isinstance(pipeline, ColumnTransformer)

    transformer_names = [t[0] for t in pipeline.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_build_preprocessing_pipeline_with_dataframe(sample_raw_dataframe):
    """
    Verifies build_preprocessing_pipeline when supplied with a DataFrame.
    """
    pipeline = build_preprocessing_pipeline(sample_raw_dataframe)
    assert isinstance(pipeline, ColumnTransformer)

    cleaned = clean_dataframe(sample_raw_dataframe)
    pipeline.fit(cleaned)
    transformed = pipeline.transform(cleaned)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_raw_dataframe)
    assert transformed.shape[1] >= len(NUMERICAL_FEATURES)


def test_preprocessing_pipeline_transform_and_feature_names(sample_raw_dataframe):
    """
    Tests fitting and transforming with the pipeline and extracting valid feature names.
    """
    cleaned = clean_dataframe(sample_raw_dataframe)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(cleaned)

    transformed = pipeline.transform(cleaned)
    feature_names = get_feature_names(pipeline)

    assert len(feature_names) == transformed.shape[1]
    assert all(isinstance(f, str) for f in feature_names)

    # Numerical features should retain their names
    for num_col in NUMERICAL_FEATURES:
        assert num_col in feature_names


def test_preprocessing_handles_unknown_categories(sample_raw_dataframe):
    """
    Verifies that OneHotEncoder with handle_unknown='ignore' handles novel categories
    without raising an exception.
    """
    cleaned = clean_dataframe(sample_raw_dataframe)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(cleaned)

    # Create test record with an unobserved department and role
    unseen_df = cleaned.iloc[[0]].copy()
    unseen_df.at[0, "Department"] = "Unknown Dept"
    unseen_df.at[0, "JobRole"] = "Unknown Role"

    transformed = pipeline.transform(unseen_df)
    assert transformed.shape[1] == len(get_feature_names(pipeline))
    assert not np.isnan(transformed).any()
