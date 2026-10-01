# Employee Attrition Prediction & Explainability API

FastAPI service that predicts employee attrition risk and provides local prediction explanations using SHAP (SHapley Additive exPlanations).

---

## Project Structure

```
employee-attrition-prediction/
├── app/
│   ├── __init__.py
│   ├── api_models.py          # Pydantic schemas for request validation & responses
│   ├── main.py                # FastAPI routes, lifecycle handler, and middleware
│   ├── model_inference.py     # Inference logic and SHAP TreeExplainer integration
│   └── preprocessing.py       # ColumnTransformer pipeline and data cleaning
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv # IBM HR dataset
├── models/
│   ├── attrition_model.joblib # Trained Random Forest model
│   ├── preprocessor.joblib    # Fitted ColumnTransformer
│   └── evaluation_metrics.json# Model comparison and test set metrics
├── notebooks/
│   └── Attrition_EDA_Training.ipynb # EDA, model benchmarking, and SHAP analysis
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # Route, schema validation, and status code tests
│   ├── test_inference.py      # Inference, probability, and SHAP tests
│   └── test_preprocessing.py  # Data cleaning and pipeline transformation tests
├── Dockerfile                 # Container build definition
├── docker-compose.yml         # Container orchestration config
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variable template
└── README.md
```

---

## Setup & Local Run

### 1. Environment and Dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Model Training (Optional)
The repository includes trained artifacts in `models/`. To retrain:
```bash
python train.py
```

### 3. Start the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs are available at `http://localhost:8000/docs`.

---

## Docker Run

### Using Docker Compose
```bash
docker compose up --build
```

### Using Docker CLI
```bash
docker build -t employee-attrition-api .
docker run -p 8000:8000 --env-file .env.example employee-attrition-api
```

---

## API Endpoints & Examples

### `GET /health`
Returns service status and artifact load state.

```bash
curl -X GET "http://localhost:8000/health"
```

Response (`200 OK`):
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict`
Predicts attrition risk (`Yes`/`No`), probability, and risk tier (`Low`, `Medium`, `High`).

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
       "YearsWithCurrManager": 3
     }'
```

Response (`200 OK`):
```json
{
  "prediction": "No",
  "probability": 0.0397,
  "risk_level": "Low"
}
```

---

### `POST /explain`
Returns prediction, probability, full dictionary of SHAP values across all 51 features, plus top risk and retention factors.

```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{
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
       "YearsWithCurrManager": 3
     }'
```

Response (`200 OK`):
```json
{
  "prediction": "No",
  "probability": 0.0397,
  "risk_level": "Low",
  "top_risk_factors": {
    "Gender_Female": 0.002,
    "Gender_Male": 0.0015,
    "EducationField_Life Sciences": 0.0013,
    "MaritalStatus_Divorced": 0.00071,
    "JobRole_Healthcare Representative": 0.00068
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
    "MonthlyIncome": -0.0312,
    "StockOptionLevel": -0.03317,
    "OverTime_Yes": -0.03
  }
}
```

---

### Input Validation Errors (`422 Unprocessable Entity`)
Invalid ranges or categorical values return field-level validation errors:
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "Age"],
      "msg": "Input should be greater than or equal to 18",
      "input": 15
    }
  ]
}
```

---

## Exploratory Data Analysis (EDA)

The IBM HR dataset contains 1,470 records and 35 columns.

- **Target Imbalance**: 16.1% attrition (237 positive cases, 1,233 negative cases).
- **Constant / Identifier Columns Dropped**:
  - `EmployeeCount` (constant 1)
  - `StandardHours` (constant 80)
  - `Over18` (constant 'Y')
  - `EmployeeNumber` (identifier)
- **Key Attrition Drivers**:
  - **OverTime**: Employees working overtime have a 30.5% attrition rate vs 10.4% for non-overtime.
  - **Stock Options**: Employees with level 0 stock options have 24.4% attrition vs ~7.5% for levels 1 and 2.
  - **Tenure & Role**: Attrition is highest for Sales Representatives (39.8%) and Laboratory Technicians (23.9%), and lowest for Research Directors (2.5%) and Managers (4.9%).

---

## Model Training & Evaluation

Data was split 70/30 (stratified by Attrition). Preprocessing uses `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categoricals (51 features total). Three classifiers were trained and evaluated using 5-fold stratified cross-validation on train data and tested on the held-out 30% set:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** (Selected) | **82.99%** | **42.86%** | **16.90%** | **0.2424** | **0.7825** | **0.7998 ± 0.025** |
| GradientBoostingClassifier | 84.13% | 51.61% | 22.54% | 0.3137 | 0.7785 | 0.8064 ± 0.037 |
| LogisticRegression (balanced) | 76.42% | 37.21% | 67.61% | 0.4800 | 0.8164 | 0.8232 ± 0.051 |

`RandomForestClassifier` was selected for deployment because of its stability on tabular data and direct compatibility with `shap.TreeExplainer` for fast, exact tree-based SHAP calculations.

---

## SHAP Interpretability

The `/explain` endpoint uses `shap.TreeExplainer` to compute local feature attributions:
- **Positive SHAP value**: Increases predicted probability of attrition (risk factor).
- **Negative SHAP value**: Decreases predicted probability of attrition (retention factor).

The most frequent high-impact features across the dataset are `OverTime`, `StockOptionLevel`, `MonthlyIncome`, `Age`, and `YearsWithCurrManager`.

---

## Running Tests

Run unit and integration tests using pytest:

```bash
pytest -v --cov=app --cov-report=term-missing tests/
```

Test coverage is 95% across the core modules (`preprocessing.py`, `model_inference.py`, `api_models.py`, `main.py`).

---

## Design Choices

- **Unified Preprocessing**: A single scikit-learn `ColumnTransformer` is fitted during training and reused at inference time to prevent training-serving skew.
- **Strict Validation**: Pydantic v2 schemas enforce data types, numeric ranges (e.g. `Age` 18–65, `MonthlyIncome` 1000–30000), and allowable category literals.
- **Lifecycle Loading**: The model and preprocessor are loaded once during application startup via FastAPI's `lifespan` handler.
- **Containerization**: The Docker image runs on `python:3.10-slim` under a non-root user (`appuser`) with an integrated healthcheck.
