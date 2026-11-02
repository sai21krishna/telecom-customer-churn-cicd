
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import preprocess_data


# Initialize FastAPI
app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="ML-powered customer churn prediction system",
    version="1.0.0"
)


# Load trained model
MODEL_PATH = "models/churn_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    raise RuntimeError(
        f"Unable to load trained model: {error}"
    )


# Input request schema
class CustomerData(BaseModel):

    Gender: str = "Male"
    Senior_Citizen: str = "No"

    Tenure_Months: float = Field(12, ge=0, le=72)

    Contract_Type: str = "Month-to-month"
    Internet_Service: str = "Fiber"
    Payment_Method: str = "UPI"

    Monthly_Charges_INR: float = Field(
        1000, ge=0, le=5000
    )

    Total_Charges_INR: float = Field(
        12000, ge=0
    )

    Tech_Support: str = "No"
    Online_Security: str = "No"

    Support_Tickets_6M: float = Field(
        1, ge=0, le=20
    )

    Avg_Data_Usage_GB: float = Field(
        100, ge=0, le=500
    )


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs"
    }


# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "application": "Customer Churn Prediction"
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerData):

    try:

        input_data = pd.DataFrame([
            customer.model_dump()
        ])

        # Apply the same feature engineering
        processed_data = preprocess_data(input_data)

        # Predict class
        prediction = int(
            model.predict(processed_data)[0]
        )

        # Predict probability
        probability = model.predict_proba(
            processed_data
        )[0][1]

        result = (
            "Churn" if prediction == 1
            else "Not Churn"
        )

        return {
            "prediction": result,
            "churn_probability": round(
                float(probability), 4
            ),
            "message": (
                "Customer is likely to discontinue services"
                if prediction == 1
                else "Customer is likely to continue services"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
