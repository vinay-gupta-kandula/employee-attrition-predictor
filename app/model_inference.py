"""
Model inference, artifact loading, and SHAP explainability engine.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import shap
from app.preprocessing import get_feature_names, clean_dataframe

# Global state for loaded ML artifacts
model: Optional[Any] = None
preprocessor: Optional[Any] = None
explainer: Optional[shap.TreeExplainer] = None
feature_names: Optional[List[str]] = None


def load_artifacts(
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
) -> None:
    """
    Load serialized model, preprocessor, and initialize SHAP TreeExplainer.
    Environment variables MODEL_PATH and PREPROCESSOR_PATH take precedence.
    """
    global model, preprocessor, explainer, feature_names

    resolved_model_path = model_path or os.getenv(
        "MODEL_PATH", os.path.join("models", "attrition_model.joblib")
    )
    resolved_preprocessor_path = preprocessor_path or os.getenv(
        "PREPROCESSOR_PATH", os.path.join("models", "preprocessor.joblib")
    )

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at {resolved_model_path}"
        )
    if not os.path.exists(resolved_preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {resolved_preprocessor_path}"
        )

    model = joblib.load(resolved_model_path)
    preprocessor = joblib.load(resolved_preprocessor_path)

    # Initialize TreeExplainer for the loaded model
    explainer = shap.TreeExplainer(model)

    # Extract feature names from fitted preprocessor
    feature_names = get_feature_names(preprocessor)


def check_artifacts_loaded() -> bool:
    """
    Verify whether all required artifacts are properly loaded into memory.
    """
    return (
        model is not None
        and preprocessor is not None
        and explainer is not None
        and feature_names is not None
    )


def compute_risk_tier(probability: float) -> str:
    """
    Assign risk tier based on predicted attrition probability.
    """
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


def predict_attrition(data_point: pd.DataFrame) -> Tuple[int, float]:
    """
    Generate attrition prediction (class) and probability for input record.
    """
    if not check_artifacts_loaded():
        raise RuntimeError("ML model and preprocessor artifacts have not been loaded.")

    cleaned_input = clean_dataframe(data_point)
    processed_data = preprocessor.transform(cleaned_input)

    prediction = model.predict(processed_data)
    proba = model.predict_proba(processed_data)[:, 1]

    return int(prediction[0]), float(proba[0])


def explain_prediction(
    data_point: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute local SHAP explanation values for an individual data point.
    Returns:
      - shap_values: dictionary of all feature names to SHAP values
      - top_risk_factors: features pushing attrition risk up (positive SHAP)
      - top_retention_factors: features reducing attrition risk (negative SHAP)
    """
    if not check_artifacts_loaded():
        raise RuntimeError("ML model and preprocessor artifacts have not been loaded.")

    cleaned_input = clean_dataframe(data_point)
    processed_data = preprocessor.transform(cleaned_input)

    raw_shap = explainer.shap_values(processed_data)

    # Handle different SHAP output formats across versions (list vs 3D array vs 2D array)
    if isinstance(raw_shap, list):
        # List of [class_0_values, class_1_values]
        sample_shap = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
    elif hasattr(raw_shap, "shape") and len(raw_shap.shape) == 3:
        # Array shape: (n_samples, n_features, n_classes)
        sample_shap = (
            raw_shap[0, :, 1] if raw_shap.shape[2] > 1 else raw_shap[0, :, 0]
        )
    elif hasattr(raw_shap, "shape") and len(raw_shap.shape) == 2:
        sample_shap = raw_shap[0]
    else:
        sample_shap = np.array(raw_shap).flatten()

    # Map SHAP values to feature names
    shap_dict = {
        name: round(float(val), 5)
        for name, val in zip(feature_names, sample_shap)
    }

    # Extract top factors for business interpretability
    risk_factors = {k: v for k, v in shap_dict.items() if v > 0}
    sorted_risk = dict(
        sorted(risk_factors.items(), key=lambda item: item[1], reverse=True)[:5]
    )

    retention_factors = {k: v for k, v in shap_dict.items() if v < 0}
    sorted_retention = dict(
        sorted(retention_factors.items(), key=lambda item: item[1])[:5]
    )

    return shap_dict, sorted_risk, sorted_retention
