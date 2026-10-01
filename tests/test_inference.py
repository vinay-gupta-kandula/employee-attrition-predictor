"""
Unit tests for model inference, artifact loading, and SHAP explainability engine
(app/model_inference.py).
"""

import os
import pytest
import pandas as pd
import numpy as np

import app.model_inference as mi
from app.model_inference import (
    load_artifacts,
    check_artifacts_loaded,
    compute_risk_tier,
    predict_attrition,
    explain_prediction,
)


@pytest.fixture(autouse=True)
def ensure_artifacts_loaded():
    """
    Ensure artifacts are loaded before each inference test.
    """
    if not check_artifacts_loaded():
        load_artifacts()
    yield


@pytest.fixture
def sample_employee_df():
    """
    Valid sample employee input record as a single-row DataFrame.
    """
    return pd.DataFrame([
        {
            "Age": 38,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 1200,
            "Department": "Research & Development",
            "DistanceFromHome": 4,
            "Education": 3,
            "EducationField": "Medical",
            "EnvironmentSatisfaction": 3,
            "Gender": "Male",
            "HourlyRate": 75,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 5200,
            "MonthlyRate": 16000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 13,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 6,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 4,
        }
    ])


@pytest.fixture
def high_risk_employee_df():
    """
    High-risk profile employee record (young, low tenure, high overtime, low satisfaction).
    """
    return pd.DataFrame([
        {
            "Age": 22,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 300,
            "Department": "Sales",
            "DistanceFromHome": 25,
            "Education": 1,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 40,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2100,
            "MonthlyRate": 8000,
            "NumCompaniesWorked": 1,
            "OverTime": "Yes",
            "PercentSalaryHike": 10,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 1,
            "TrainingTimesLastYear": 1,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
        }
    ])


def test_load_artifacts_success():
    """
    Verifies that load_artifacts correctly populates global objects.
    """
    load_artifacts()
    assert check_artifacts_loaded() is True
    assert mi.model is not None
    assert mi.preprocessor is not None
    assert mi.explainer is not None
    assert mi.feature_names is not None
    assert len(mi.feature_names) == 51


def test_load_artifacts_missing_files():
    """
    Verifies that load_artifacts raises FileNotFoundError when passed nonexistent paths.
    """
    with pytest.raises(FileNotFoundError):
        load_artifacts(model_path="nonexistent_model.joblib")

    with pytest.raises(FileNotFoundError):
        load_artifacts(preprocessor_path="nonexistent_preprocessor.joblib")


def test_compute_risk_tier():
    """
    Tests risk tier threshold categorization.
    """
    assert compute_risk_tier(0.10) == "Low"
    assert compute_risk_tier(0.249) == "Low"
    assert compute_risk_tier(0.25) == "Medium"
    assert compute_risk_tier(0.499) == "Medium"
    assert compute_risk_tier(0.50) == "High"
    assert compute_risk_tier(0.95) == "High"


def test_predict_attrition_output(sample_employee_df):
    """
    Tests predict_attrition returns expected class and probability bounds.
    """
    prediction, probability = predict_attrition(sample_employee_df)

    assert prediction in (0, 1)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_explain_prediction_structure(sample_employee_df):
    """
    Tests explain_prediction returns valid SHAP dict, top risk, and top retention factors.
    """
    shap_dict, top_risk, top_retention = explain_prediction(sample_employee_df)

    assert isinstance(shap_dict, dict)
    assert len(shap_dict) == len(mi.feature_names)

    # All values in shap_dict should be numeric
    for k, v in shap_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, float)

    # Top risk factors should have positive SHAP values (pushing towards attrition)
    assert isinstance(top_risk, dict)
    for feat, val in top_risk.items():
        assert val > 0, f"Expected positive SHAP value for risk factor {feat}, got {val}"

    # Top retention factors should have negative SHAP values (pushing towards retention)
    assert isinstance(top_retention, dict)
    for feat, val in top_retention.items():
        assert val < 0, f"Expected negative SHAP value for retention factor {feat}, got {val}"


def test_predict_unloaded_raises_runtime_error(sample_employee_df, monkeypatch):
    """
    Verifies RuntimeError is raised if prediction is attempted without loaded artifacts.
    """
    monkeypatch.setattr(mi, "model", None)
    with pytest.raises(RuntimeError):
        predict_attrition(sample_employee_df)


def test_explain_unloaded_raises_runtime_error(sample_employee_df, monkeypatch):
    """
    Verifies RuntimeError is raised if explanation is attempted without loaded artifacts.
    """
    monkeypatch.setattr(mi, "explainer", None)
    with pytest.raises(RuntimeError):
        explain_prediction(sample_employee_df)
