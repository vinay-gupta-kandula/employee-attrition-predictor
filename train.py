"""
Model Training, Evaluation, and Serialization for Employee Attrition Prediction.
Trains classification models with class-imbalance mitigation, evaluates metrics,
and saves the trained model and preprocessor artifacts to models/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from app.preprocessing import (
    build_preprocessing_pipeline,
    clean_dataframe,
    get_feature_names,
)


def load_and_prepare_data(filepath: str = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Encode target: Yes -> 1, No -> 0
    y = (df["Attrition"].str.strip() == "Yes").astype(int)

    # Clean features: drops ID and constant columns
    X = clean_dataframe(df)
    print(f"Features after cleaning: {X.shape[1]} columns")

    return X, y, df


def train_and_evaluate():
    os.makedirs("models", exist_ok=True)
    X, y, df = load_and_prepare_data()

    # Stratified 70/30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples (Attrition: {y_train.mean():.2%})")
    print(f"Test set:  {X_test.shape[0]} samples (Attrition: {y_test.mean():.2%})")

    # Build and fit preprocessing pipeline on training data ONLY
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"Transformed features count: {len(feature_names)}")

    # Candidate models
    models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    }

    results = {}
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (TEST SET: 30% Stratified)")
    print("=" * 60)

    for name, model in models.items():
        # 5-fold Stratified Cross-Validation on Train
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="roc_auc")

        # Fit model on full training set
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc), 4),
            "CV_ROC_AUC_Mean": round(float(cv_scores.mean()), 4),
            "CV_ROC_AUC_Std": round(float(cv_scores.std()), 4),
            "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
            "ClassificationReport": classification_report(y_test, y_pred, output_dict=True),
        }

        print(f"\n--- {name} ---")
        print(f"  Accuracy:         {acc:.4f}")
        print(f"  Precision:        {prec:.4f}")
        print(f"  Recall:           {rec:.4f}")
        print(f"  F1-Score:         {f1:.4f}")
        print(f"  ROC-AUC:          {roc:.4f}")
        print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Select RandomForestClassifier as the primary serving model
    # (provides tree structure for fast exact SHAP TreeExplainer and balanced weights)
    primary_model_name = "RandomForestClassifier"
    primary_model = models[primary_model_name]

    print("\n" + "=" * 60)
    print(f"Primary Serving Model: {primary_model_name}")
    print("=" * 60)

    # Validate SHAP TreeExplainer on sample
    print("Validating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(primary_model)
    sample_trans = X_test_trans[:1]
    shap_vals = explainer.shap_values(sample_trans)
    print(f"SHAP explanation successfully generated. Shape: {getattr(shap_vals, 'shape', len(shap_vals))}")

    # Save artifacts
    model_path = os.path.join("models", "attrition_model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    metrics_path = os.path.join("models", "evaluation_metrics.json")

    joblib.dump(primary_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    summary_metadata = {
        "selected_model": primary_model_name,
        "metrics": results,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }

    with open(metrics_path, "w") as f:
        json.dump(summary_metadata, f, indent=2)

    print(f"Artifacts successfully saved to:")
    print(f" - {model_path}")
    print(f" - {preprocessor_path}")
    print(f" - {metrics_path}")

    return primary_model, preprocessor, results


if __name__ == "__main__":
    train_and_evaluate()
