"""
FastAPI application entry point for Explainable Employee Attrition Prediction.
Provides /predict, /explain, /health, and root API documentation endpoints.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api_models import (
    EmployeeFeatures,
    PredictionResponse,
    ExplanationResponse,
    HealthResponse,
)
from app.model_inference import (
    load_artifacts,
    predict_attrition,
    explain_prediction,
    check_artifacts_loaded,
    compute_risk_tier,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown resource handling.
    """
    model_path = os.getenv("MODEL_PATH", os.path.join("models", "attrition_model.joblib"))
    preprocessor_path = os.getenv(
        "PREPROCESSOR_PATH", os.path.join("models", "preprocessor.joblib")
    )
    try:
        load_artifacts(model_path=model_path, preprocessor_path=preprocessor_path)
        print("ML artifacts successfully loaded during startup.")
    except Exception as e:
        print(f"Warning: Failed to load ML artifacts at startup: {e}")
    yield


app = FastAPI(
    title="Employee Attrition Predictor & Explainer API",
    description=(
        "Production-ready FastAPI service providing real-time employee attrition "
        "predictions and local SHAP (SHapley Additive exPlanations) interpretability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for external integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """
    Root information endpoint.
    """
    return {
        "service": "Employee Attrition Predictor & Explainer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "explain": "POST /explain",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Service health check endpoint.
    Returns 200 OK if model and preprocessor are loaded, 503 if not ready.
    """
    if not check_artifacts_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded into memory.",
        )
    return HealthResponse(status="ok", model_loaded=True, version="1.0.0")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict(features: EmployeeFeatures):
    """
    Predict whether an employee is at risk of attrition based on HR attributes.
    Returns the binary prediction ('Yes'/'No'), probability, and risk tier.
    """
    if not check_artifacts_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference. Artifacts missing.",
        )

    try:
        # Convert Pydantic features to DataFrame
        data_dict = (
            features.model_dump()
            if hasattr(features, "model_dump")
            else features.dict()
        )
        df = pd.DataFrame([data_dict])

        prediction_raw, probability = predict_attrition(df)
        prediction_label = "Yes" if prediction_raw == 1 else "No"
        risk_tier = compute_risk_tier(probability)

        return PredictionResponse(
            prediction=prediction_label,
            probability=probability,
            risk_level=risk_tier,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}",
        )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Interpretability"],
)
async def explain(features: EmployeeFeatures):
    """
    Explain individual attrition prediction with local SHAP values.
    Returns prediction, probability, full SHAP value dictionary,
    top risk-increasing drivers, and top retention-promoting drivers.
    """
    if not check_artifacts_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for explanation. Artifacts missing.",
        )

    try:
        data_dict = (
            features.model_dump()
            if hasattr(features, "model_dump")
            else features.dict()
        )
        df = pd.DataFrame([data_dict])

        prediction_raw, probability = predict_attrition(df)
        prediction_label = "Yes" if prediction_raw == 1 else "No"
        risk_tier = compute_risk_tier(probability)

        shap_values, top_risk, top_retention = explain_prediction(df)

        return ExplanationResponse(
            prediction=prediction_label,
            probability=probability,
            risk_level=risk_tier,
            shap_values=shap_values,
            top_risk_factors=top_risk,
            top_retention_factors=top_retention,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation computation error: {str(e)}",
        )
