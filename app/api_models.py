"""
Pydantic data models for FastAPI request validation and response serialization.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class EmployeeFeatures(BaseModel):
    """
    Employee attributes schema with strict type validation and realistic domain bounds.
    """
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "Age": 41,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 1102,
                "Department": "Sales",
                "DistanceFromHome": 1,
                "Education": 2,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 2,
                "Gender": "Female",
                "HourlyRate": 94,
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobRole": "Sales Executive",
                "JobSatisfaction": 4,
                "MaritalStatus": "Single",
                "MonthlyIncome": 5993,
                "MonthlyRate": 19479,
                "NumCompaniesWorked": 8,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 1,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 0,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 6,
                "YearsInCurrentRole": 4,
                "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 5,
            }
        },
    )

    Age: int = Field(..., ge=18, le=65, description="Employee age in years (18-65)")
    BusinessTravel: Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"] = Field(
        ..., description="Frequency of business travel"
    )
    DailyRate: int = Field(..., ge=100, le=2000, description="Daily rate compensation")
    Department: Literal["Human Resources", "Research & Development", "Sales"] = Field(
        ..., description="Department within organization"
    )
    DistanceFromHome: int = Field(
        ..., ge=1, le=100, description="Distance from home to work in miles"
    )
    Education: int = Field(
        ...,
        ge=1,
        le=5,
        description="Education level: 1='Below College', 2='College', 3='Bachelor', 4='Master', 5='Doctor'",
    )
    EducationField: Literal[
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ] = Field(..., description="Field of educational study")
    EnvironmentSatisfaction: int = Field(
        ..., ge=1, le=4, description="Work environment satisfaction rating (1-4)"
    )
    Gender: Literal["Female", "Male"] = Field(..., description="Employee gender")
    HourlyRate: int = Field(..., ge=20, le=150, description="Hourly compensation rate")
    JobInvolvement: int = Field(
        ..., ge=1, le=4, description="Level of job involvement (1-4)"
    )
    JobLevel: int = Field(..., ge=1, le=5, description="Job seniority level (1-5)")
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ] = Field(..., description="Job role designation")
    JobSatisfaction: int = Field(
        ..., ge=1, le=4, description="Job satisfaction rating (1-4)"
    )
    MaritalStatus: Literal["Divorced", "Married", "Single"] = Field(
        ..., description="Marital status"
    )
    MonthlyIncome: int = Field(
        ..., ge=1000, le=30000, description="Monthly income in USD"
    )
    MonthlyRate: int = Field(
        ..., ge=1000, le=40000, description="Monthly rate in USD"
    )
    NumCompaniesWorked: int = Field(
        ..., ge=0, le=20, description="Number of prior companies worked for"
    )
    OverTime: Literal["No", "Yes"] = Field(
        ..., description="Whether employee works overtime"
    )
    PercentSalaryHike: int = Field(
        ..., ge=0, le=50, description="Percentage salary hike received"
    )
    PerformanceRating: int = Field(
        ..., ge=1, le=4, description="Performance appraisal rating (1-4)"
    )
    RelationshipSatisfaction: int = Field(
        ..., ge=1, le=4, description="Relationship satisfaction rating (1-4)"
    )
    StockOptionLevel: int = Field(
        ..., ge=0, le=3, description="Stock option level (0-3)"
    )
    TotalWorkingYears: int = Field(
        ..., ge=0, le=50, description="Total career working years"
    )
    TrainingTimesLastYear: int = Field(
        ..., ge=0, le=10, description="Training sessions attended last year"
    )
    WorkLifeBalance: int = Field(
        ..., ge=1, le=4, description="Work-life balance rating (1-4)"
    )
    YearsAtCompany: int = Field(
        ..., ge=0, le=50, description="Tenure at current company in years"
    )
    YearsInCurrentRole: int = Field(
        ..., ge=0, le=30, description="Years in current role"
    )
    YearsSinceLastPromotion: int = Field(
        ..., ge=0, le=30, description="Years elapsed since last promotion"
    )
    YearsWithCurrManager: int = Field(
        ..., ge=0, le=30, description="Years working with current manager"
    )


class PredictionResponse(BaseModel):
    """
    Response schema for /predict endpoint.
    """
    prediction: str = Field(..., description="Attrition prediction ('Yes' or 'No')")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of employee attrition"
    )
    risk_level: Optional[str] = Field(
        None, description="Attrition risk tier ('Low', 'Medium', 'High')"
    )


class ExplanationResponse(BaseModel):
    """
    Response schema for /explain endpoint including SHAP feature contributions.
    """
    prediction: str = Field(..., description="Attrition prediction ('Yes' or 'No')")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of employee attrition"
    )
    risk_level: Optional[str] = Field(
        None, description="Attrition risk tier ('Low', 'Medium', 'High')"
    )
    shap_values: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping each transformed feature name to its SHAP contribution value",
    )
    top_risk_factors: Optional[Dict[str, float]] = Field(
        None,
        description="Top features driving higher probability of attrition (positive SHAP values)",
    )
    top_retention_factors: Optional[Dict[str, float]] = Field(
        None,
        description="Top features driving retention / lowering attrition (negative SHAP values)",
    )


class HealthResponse(BaseModel):
    """
    Response schema for /health endpoint.
    """
    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    model_loaded: bool = Field(
        ..., description="Flag indicating if model artifacts are loaded"
    )
    version: str = Field("1.0.0", description="API version")
