"""
Unit and integration tests for FastAPI REST API endpoints (app/main.py).
Tests /health, /predict, /explain, / root, schema validation, and error scenarios.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.model_inference as mi
from app.model_inference import load_artifacts


@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture with pre-loaded ML artifacts.
    """
    load_artifacts()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_employee_payload():
    """
    Standard valid employee attributes matching EmployeeFeatures schema.
    """
    return {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1100,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Medical",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }


@pytest.fixture
def high_attrition_employee_payload():
    """
    Employee payload with high attrition characteristics (OverTime, low income, single).
    """
    return {
        "Age": 22,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 300,
        "Department": "Sales",
        "DistanceFromHome": 25,
        "Education": 1,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 1,
        "Gender": "Female",
        "HourlyRate": 40,
        "JobInvolvement": 1,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2100,
        "MonthlyRate": 7000,
        "NumCompaniesWorked": 1,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
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


def test_root_endpoint(client):
    """
    Verifies the root information endpoint returns service metadata.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "documentation" in data
    assert data["endpoints"]["predict"] == "POST /predict"
    assert data["endpoints"]["explain"] == "POST /explain"


def test_health_check_healthy(client):
    """
    Verifies that /health returns 200 OK when artifacts are loaded.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


import app.main as main_mod


def test_health_check_unhealthy(client, monkeypatch):
    """
    Verifies that /health returns 503 when model artifacts are missing.
    """
    monkeypatch.setattr(main_mod, "check_artifacts_loaded", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503
    assert "Model artifacts are not loaded" in response.json()["detail"]


def test_predict_endpoint_valid_input(client, valid_employee_payload):
    """
    Tests /predict endpoint with a valid employee payload.
    """
    response = client.post("/predict", json=valid_employee_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_endpoint_high_risk_employee(client, high_attrition_employee_payload):
    """
    Tests /predict endpoint with a high attrition risk employee profile.
    """
    response = client.post("/predict", json=high_attrition_employee_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    # High-risk profile should reflect elevated probability compared to baseline
    assert data["risk_level"] in ["Medium", "High"]


def test_predict_validation_error_age_too_low(client, valid_employee_payload):
    """
    Tests that age < 18 triggers 422 Unprocessable Entity.
    """
    payload = valid_employee_payload.copy()
    payload["Age"] = 16  # Minimum age is 18
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any("Age" in str(err) for err in response.json()["detail"])


def test_predict_validation_error_invalid_category(client, valid_employee_payload):
    """
    Tests that unlisted categorical option triggers 422 Unprocessable Entity.
    """
    payload = valid_employee_payload.copy()
    payload["BusinessTravel"] = "Frequent_Space_Flight"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any("BusinessTravel" in str(err) for err in response.json()["detail"])


def test_predict_validation_error_missing_field(client):
    """
    Tests that missing required fields trigger 422 Unprocessable Entity.
    """
    incomplete_payload = {"Age": 30, "MonthlyIncome": 5000}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_explain_endpoint_valid_input(client, valid_employee_payload):
    """
    Tests /explain endpoint returning local SHAP values and driver rankings.
    """
    response = client.post("/explain", json=valid_employee_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]

    # Verify SHAP contributions
    assert "shap_values" in data
    assert isinstance(data["shap_values"], dict)
    assert len(data["shap_values"]) == 51

    # Verify top risk and retention factor sections
    assert "top_risk_factors" in data
    assert isinstance(data["top_risk_factors"], dict)

    assert "top_retention_factors" in data
    assert isinstance(data["top_retention_factors"], dict)


def test_explain_validation_error(client, valid_employee_payload):
    """
    Tests that invalid input to /explain triggers 422 Unprocessable Entity.
    """
    payload = valid_employee_payload.copy()
    payload["MonthlyIncome"] = 500  # Less than min 1000
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_predict_unloaded_model_service_unavailable(client, valid_employee_payload, monkeypatch):
    """
    Tests that /predict returns 503 if artifacts are unloaded.
    """
    monkeypatch.setattr(main_mod, "check_artifacts_loaded", lambda: False)
    response = client.post("/predict", json=valid_employee_payload)
    assert response.status_code == 503
    assert "Model is not ready" in response.json()["detail"]


def test_explain_unloaded_model_service_unavailable(client, valid_employee_payload, monkeypatch):
    """
    Tests that /explain returns 503 if artifacts are unloaded.
    """
    monkeypatch.setattr(main_mod, "check_artifacts_loaded", lambda: False)
    response = client.post("/explain", json=valid_employee_payload)
    assert response.status_code == 503
    assert "Model is not ready" in response.json()["detail"]


def test_predict_internal_error_handling(client, valid_employee_payload, monkeypatch):
    """
    Tests that unexpected exceptions during inference return 500 Internal Server Error.
    """
    def raise_err(df):
        raise ValueError("Simulated unexpected inference failure")

    monkeypatch.setattr(main_mod, "predict_attrition", raise_err)
    response = client.post("/predict", json=valid_employee_payload)
    assert response.status_code == 500
    assert "Inference computation error" in response.json()["detail"]


def test_explain_internal_error_handling(client, valid_employee_payload, monkeypatch):
    """
    Tests that unexpected exceptions during explanation return 500 Internal Server Error.
    """
    def raise_err(df):
        raise ValueError("Simulated unexpected explanation failure")

    monkeypatch.setattr(main_mod, "explain_prediction", raise_err)
    response = client.post("/explain", json=valid_employee_payload)
    assert response.status_code == 500
    assert "SHAP explanation computation error" in response.json()["detail"]

