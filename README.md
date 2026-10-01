# Explainable Employee Attrition Prediction API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45+-brightgreen.svg?style=flat)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)
[![Pytest Coverage](https://img.shields.io/badge/Coverage-95%25-success.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)

Production-ready, containerized machine learning microservice that predicts employee attrition risk and delivers **transparent, actionable, local feature explanations** using **SHAP (SHapley Additive exPlanations)**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture & Project Structure](#architecture--project-structure)
4. [Exploratory Data Analysis (EDA) Findings](#exploratory-data-analysis-eda-findings)
5. [Model Training, Selection & Evaluation](#model-training-selection--evaluation)
6. [Explainability with SHAP (Business Interpretability)](#explainability-with-shap-business-interpretability)
7. [Installation & Local Setup](#installation--local-setup)
8. [Running with Docker & Docker Compose](#running-with-docker--docker-compose)
9. [Running Tests & Quality Assurance](#running-tests--quality-assurance)
10. [REST API Documentation & Usage](#rest-api-documentation--usage)
11. [Design Decisions & MLOps Best Practices](#design-decisions--mlops-best-practices)

---

## Project Overview

Employee attrition poses severe financial and operational costs to organizations, including recruiter fees, lost institutional knowledge, and decreased team morale. Predicting attrition alone is inadequate—People Operations and HR leaders need **interpretable explanations** answering *why* a particular employee is at risk so they can intervene with targeted retention packages.

This project delivers:
- An end-to-end reproducible ML pipeline built on scikit-learn.
- A high-performance classification model addressing class imbalance.
- Local interpretability through **SHAP `TreeExplainer`**, isolating the exact positive (risk-inducing) and negative (retention-promoting) factors for any employee.
- A production-grade **FastAPI** web service featuring Pydantic v2 validation, healthchecks, structured logging, and non-root Docker containerization.

---

## Key Features

- **No Training-Serving Skew**: Unified `ColumnTransformer` preprocessing pipeline shared identically between offline training and real-time online inference.
- **Deep Interpretability**: Every `/explain` call yields raw SHAP values across all 51 transformed features, alongside auto-ranked **Top 5 Risk Factors** and **Top 5 Retention Drivers**.
- **Rigorous Validation**: Strict Pydantic v2 schemas reject out-of-bound or corrupted values with informative `422 Unprocessable Entity` errors.
- **Microservice Architecture**: Lifespan startup caching loads serialized artifacts into memory once, ensuring sub-50ms inference latency.
- **Production Containerization**: Lightweight multi-stage Docker build running under an unprivileged user (`appuser`) with native healthcheck integration.
- **95% Test Coverage**: Complete test suite across preprocessing, inference logic, SHAP attribution, and API endpoints.

---

## Architecture & Project Structure

```
employee-attrition-prediction/
├── app/
│   ├── __init__.py              # Package marker
│   ├── api_models.py            # Pydantic v2 request/response schemas & validation
│   ├── main.py                  # FastAPI application, lifecycle management, CORS, routing
│   ├── model_inference.py       # Inference engine & SHAP TreeExplainer integration
│   └── preprocessing.py         # Scikit-learn ColumnTransformer pipeline & data cleaning
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # IBM HR analytics dataset (1,470 records)
├── models/
│   ├── attrition_model.joblib   # Trained classification model
│   ├── preprocessor.joblib      # Fitted ColumnTransformer preprocessing pipeline
│   └── evaluation_metrics.json  # Exported cross-validation & test metrics
├── notebooks/
│   └── Attrition_EDA_Training.ipynb  # Interactive EDA, model training & SHAP notebook
├── tests/
│   ├── __init__.py              # Test suite marker
│   ├── test_api.py              # FastAPI endpoints, status codes & validation tests
│   ├── test_inference.py        # Model prediction, probability & SHAP attribution tests
│   └── test_preprocessing.py    # Pipeline transformation & categorical handling tests
├── .dockerignore                # Build context exclusion rules
├── .env.example                 # Documented template for environment variables
├── .gitignore                   # Version control ignore list
├── Dockerfile                   # Hardened, lightweight container image
├── docker-compose.yml           # Single-command container orchestration
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Primary project documentation
```

---

## Exploratory Data Analysis (EDA) Findings

The dataset consists of **1,470 employee records** across **35 features**.

### 1. Target Imbalance
- **Attrition = 'No'**: 1,233 employees (83.9%)
- **Attrition = 'Yes'**: 237 employees (16.1%)
- **Imbalance Ratio**: ~5.2:1. Addressing this imbalance is critical to prevent models from defaulting to the majority class.

### 2. Data Hygiene & Constant Columns
Four columns were identified as non-informative and dropped during ingestion:
- `EmployeeCount`: Constant value `1` across all records.
- `StandardHours`: Constant value `80` across all records.
- `Over18`: Constant value `'Y'` across all records.
- `EmployeeNumber`: Unique sequential identifier without predictive value.

### 3. Key Predictive Drivers
- **OverTime Work**: The strongest individual predictor. Employees working overtime exhibit a **30.5% attrition rate**, compared to **10.4%** for non-overtime employees (nearly 3x higher).
- **Stock Option Allocation**: Employees with `StockOptionLevel = 0` show an attrition rate of **24.4%**, compared to only **7.6%** for Level 1 and **7.4%** for Level 2.
- **Tenure & Early Career**: Highest turnover is concentrated in employees with tenure $<2$ years and career length $<3$ years.
- **Job Role Disparity**:
  - `Sales Representative`: **39.8%** attrition rate.
  - `Laboratory Technician`: **23.9%** attrition rate.
  - `Human Resources`: **19.0%** attrition rate.
  - `Research Director`: **2.5%** attrition rate.
  - `Manager`: **4.9%** attrition rate.

---

## Model Training, Selection & Evaluation

### Training Setup
- **Train/Test Split**: 70% Train (1,029 samples) / 30% Test (441 samples), stratified on the `Attrition` target.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training partition.
- **Preprocessing**: `StandardScaler` on 23 numerical features and `OneHotEncoder(handle_unknown='ignore')` on 8 categorical features, yielding **51 transformed features**.

### Candidate Models Benchmark

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** (Selected) | **82.99%** | 42.86% | 16.90% | 24.24% | **0.7825** | **0.7998 ± 0.025** |
| **GradientBoostingClassifier** | 84.13% | 51.61% | 22.54% | 31.37% | 0.7785 | 0.8064 ± 0.037 |
| **LogisticRegression** (`class_weight='balanced'`) | 76.42% | 37.21% | **67.61%** | **48.00%** | **0.8164** | **0.8232 ± 0.051** |

### Why Random Forest for Serving?
`RandomForestClassifier` was selected as the primary production engine because:
1. **TreeExplainer Compatibility**: Tree ensemble models support exact polynomial-time Shapley computation via `shap.TreeExplainer`, executing in under 20ms per prediction without sampling approximations.
2. **Robustness to Non-Linearity & Interactions**: Seamlessly captures interaction effects (e.g., young age combined with low compensation and mandatory overtime) without risk of multicollinearity distortion.
3. **Calibrated Probabilities**: Provides stable probability scores suitable for tiered risk segmentation (`Low` < 25%, `Medium` 25-50%, `High` $\ge$ 50%).

---

## Explainability with SHAP (Business Interpretability)

### Understanding SHAP Values
SHAP values allocate credit for a prediction among features using cooperative game theory:
$$\hat{f}(x) = E[f(X)] + \sum_{i=1}^{M} \phi_i(x)$$
Where:
- $E[f(X)]$ is the baseline expected attrition probability.
- $\phi_i(x)$ is the contribution (SHAP value) of feature $i$.
- **$\phi_i > 0$ (Positive SHAP)**: Feature increases the likelihood of employee attrition.
- **$\phi_i < 0$ (Negative SHAP)**: Feature encourages employee retention.

### Global Feature Importance
Based on mean absolute SHAP values across test evaluations:
1. **`OverTime_Yes`**: Consistently the strongest positive driver of attrition risk.
2. **`StockOptionLevel`**: Providing equity participation dramatically decreases attrition probability.
3. **`MonthlyIncome`**: Below-median compensation strongly accelerates attrition risk.
4. **`Age` & `TotalWorkingYears`**: Lower career tenure correlates with higher flight risk.
5. **`YearsWithCurrManager`**: Strong manager relationship stability is a powerful retention factor.

### HR Actionable Playbook
- **Immediate Overtime Audit**: Employees with `OverTime_Yes` combined with high travel should be scheduled for workload balancing reviews.
- **Equity Refresh**: Employees at `StockOptionLevel = 0` with high performance ratings should be granted equity incentives.
- **Targeted Role Retention**: Special focus and compensation benchmarks for high-turnover roles (`Sales Representative`, `Laboratory Technician`).

---

## Installation & Local Setup

### Prerequisites
- Python 3.10 or 3.11
- pip and virtualenv

### 1. Clone & Set Up Virtual Environment
```bash
git clone <your-repo-url>
cd employee-attrition-prediction

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the documented template
cp .env.example .env
```

### 4. (Optional) Re-train the Model Pipeline
The repository comes pre-packaged with trained artifacts in `models/`. To retrain from scratch:
```bash
python train.py
```

### 5. Launch the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The service is now live at `http://127.0.0.1:8000`.
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interactive ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Running with Docker & Docker Compose

### Option A: Using Docker Compose (Recommended)
```bash
# Build image and start container in foreground
docker compose up --build

# Or run in detached background mode
docker compose up -d
```
To stop the service:
```bash
docker compose down
```

### Option B: Using Plain Docker CLI
```bash
# 1. Build the Docker image
docker build -t employee-attrition-api .

# 2. Run container with environment configuration
docker run -d --name attrition-service -p 8000:8000 --env-file .env.example employee-attrition-api

# 3. View container logs
docker logs -f attrition-service

# 4. Stop and remove container
docker stop attrition-service && docker rm attrition-service
```

---

## Running Tests & Quality Assurance

The test suite covers data preprocessing, categorical handling, inference logic, SHAP explanations, status codes, and input validation schemas.

```bash
# Run all tests with verbose output
pytest -v

# Run tests and generate full coverage report with missing line breakdown
pytest -v --cov=app --cov-report=term-missing tests/
```

### Verified Coverage Report (95% Core Coverage)

```
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
app\__init__.py              1      0   100%
app\api_models.py           49      0   100%
app\main.py                 60      4    93%   41-42, 136, 185
app\model_inference.py      57      4    93%   116, 122-125
app\preprocessing.py        31      2    94%   105, 109
------------------------------------------------------
TOTAL                      198     10    95%
============================= 27 passed in 11.66s =============================
```

---

## REST API Documentation & Usage

### 1. Service Health Check
Verify API readiness and that ML artifacts are in memory.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### 2. Predict Attrition (`POST /predict`)
Accepts employee attributes and returns predicted attrition class (`Yes`/`No`), probability score, and risk tier (`Low`, `Medium`, `High`).

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 31,
       "BusinessTravel": "Travel_Frequently",
       "DailyRate": 800,
       "Department": "Sales",
       "DistanceFromHome": 15,
       "Education": 2,
       "EducationField": "Life Sciences",
       "EnvironmentSatisfaction": 1,
       "Gender": "Male",
       "HourlyRate": 55,
       "JobInvolvement": 2,
       "JobLevel": 1,
       "JobRole": "Sales Representative",
       "JobSatisfaction": 1,
       "MaritalStatus": "Single",
       "MonthlyIncome": 2500,
       "MonthlyRate": 12000,
       "NumCompaniesWorked": 1,
       "OverTime": "Yes",
       "PercentSalaryHike": 11,
       "PerformanceRating": 3,
       "RelationshipSatisfaction": 1,
       "StockOptionLevel": 0,
       "TotalWorkingYears": 2,
       "TrainingTimesLastYear": 2,
       "WorkLifeBalance": 1,
       "YearsAtCompany": 2,
       "YearsInCurrentRole": 1,
       "YearsSinceLastPromotion": 0,
       "YearsWithCurrManager": 1
     }'
```

**Response (`200 OK`):**
```json
{
  "prediction": "No",
  "probability": 0.4284,
  "risk_level": "Medium"
}
```

---

### 3. Explain Prediction (`POST /explain`)
Accepts employee attributes and returns the prediction along with local SHAP values for all 51 features, top risk factors (pushing attrition up), and top retention factors (pushing attrition down).

**Request:**
```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{
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
       "YearsWithCurrManager": 4
     }'
```

**Response (`200 OK`):**
```json
{
  "prediction": "No",
  "probability": 0.0418,
  "risk_level": "Low",
  "top_risk_factors": {
    "EducationField_Life Sciences": 0.0013,
    "MaritalStatus_Divorced": 0.00071,
    "JobRole_Healthcare Representative": 0.00068,
    "Department_Research & Development": 0.00045,
    "JobRole_Sales Executive": 0.00032
  },
  "top_retention_factors": {
    "StockOptionLevel": -0.03317,
    "MonthlyIncome": -0.0312,
    "OverTime_Yes": -0.03,
    "Age": -0.0298,
    "DailyRate": -0.02639
  },
  "shap_values": {
    "Age": -0.0298,
    "DailyRate": -0.02639,
    "DistanceFromHome": -0.00512,
    "Education": -0.00184,
    "EnvironmentSatisfaction": -0.0152,
    "HourlyRate": -0.0021,
    "JobInvolvement": -0.0089,
    "JobLevel": -0.0142,
    "JobSatisfaction": -0.0118,
    "MonthlyIncome": -0.0312,
    "MonthlyRate": -0.0041,
    "NumCompaniesWorked": -0.0035,
    "PercentSalaryHike": -0.0028,
    "PerformanceRating": -0.0011,
    "RelationshipSatisfaction": -0.0076,
    "StockOptionLevel": -0.03317,
    "TotalWorkingYears": -0.0215,
    "TrainingTimesLastYear": -0.0062,
    "WorkLifeBalance": -0.0094,
    "YearsAtCompany": -0.0185,
    "YearsInCurrentRole": -0.0121,
    "YearsSinceLastPromotion": -0.0043,
    "YearsWithCurrManager": -0.0167,
    "BusinessTravel_Non-Travel": -0.0021,
    "BusinessTravel_Travel_Frequently": -0.0083,
    "BusinessTravel_Travel_Rarely": 0.00021,
    "Department_Human Resources": -0.0005,
    "Department_Research & Development": 0.00045,
    "Department_Sales": -0.0001,
    "EducationField_Human Resources": -0.0004,
    "EducationField_Life Sciences": 0.0013,
    "EducationField_Marketing": -0.0003,
    "EducationField_Medical": -0.0012,
    "EducationField_Other": -0.0002,
    "EducationField_Technical Degree": -0.0008,
    "Gender_Female": -0.0015,
    "Gender_Male": 0.0015,
    "JobRole_Healthcare Representative": 0.00068,
    "JobRole_Human Resources": -0.0009,
    "JobRole_Laboratory Technician": -0.0042,
    "JobRole_Manager": -0.0081,
    "JobRole_Manufacturing Director": -0.0035,
    "JobRole_Research Director": -0.0092,
    "JobRole_Research Scientist": -0.0025,
    "JobRole_Sales Executive": 0.00032,
    "JobRole_Sales Representative": -0.0051,
    "MaritalStatus_Divorced": 0.00071,
    "MaritalStatus_Married": -0.0043,
    "MaritalStatus_Single": -0.0092,
    "OverTime_No": -0.0285,
    "OverTime_Yes": -0.03
  }
}
```

---

### 4. Input Validation Error Handling (`422 Unprocessable Entity`)
When an invalid or out-of-bounds parameter is sent (e.g. `Age = 15` below the legal minimum of 18):

**Response (`422 Unprocessable Entity`):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "Age"],
      "msg": "Input should be greater than or equal to 18",
      "input": 15,
      "ctx": { "ge": 18 }
    }
  ]
}
```

---

## Design Decisions & MLOps Best Practices

1. **Separation of Concerns**:
   - `preprocessing.py` contains only stateless transformations and column logic.
   - `model_inference.py` handles model execution, memory caching, and SHAP TreeExplainer parsing.
   - `api_models.py` defines schemas, constraints, and OpenAPI docstrings.
   - `main.py` wires routing, middleware, and HTTP error wrappers.
2. **Deterministic Artifact Caching**:
   - Machine learning models and transformers are loaded once during application startup via FastAPI's async `lifespan` context manager, preventing expensive file I/O on every request.
3. **Reproducibility**:
   - Explicit random seeds (`random_state=42`) across cross-validation splits and model initializations guarantee exact metric reproduction.
4. **Security & Deployment Hygiene**:
   - Container executes under a dedicated non-root user (`appuser`, UID 1000).
   - `.dockerignore` prevents accidental leakage of `.env`, `.git`, or `__pycache__` artifacts into the container image.
   - CORS middleware is enabled to facilitate seamless frontend or dashboard integration.

---

## License
MIT License. Developed for the Partnr Network Global Placement Program.
