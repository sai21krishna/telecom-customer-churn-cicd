
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert "Customer Churn" in response.json()["message"]


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint():

    customer = {
        "Gender": "Male",
        "Senior_Citizen": "No",
        "Tenure_Months": 12,
        "Contract_Type": "Month-to-month",
        "Internet_Service": "Fiber",
        "Payment_Method": "UPI",
        "Monthly_Charges_INR": 1200,
        "Total_Charges_INR": 14400,
        "Tech_Support": "No",
        "Online_Security": "No",
        "Support_Tickets_6M": 5,
        "Avg_Data_Usage_GB": 100
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [
        "Churn",
        "Not Churn"
    ]

    assert 0 <= data["churn_probability"] <= 1


def test_invalid_tenure():

    customer = {
        "Gender": "Male",
        "Senior_Citizen": "No",
        "Tenure_Months": 100,
        "Contract_Type": "Month-to-month",
        "Internet_Service": "Fiber",
        "Payment_Method": "UPI",
        "Monthly_Charges_INR": 1200,
        "Total_Charges_INR": 14400,
        "Tech_Support": "No",
        "Online_Security": "No",
        "Support_Tickets_6M": 5,
        "Avg_Data_Usage_GB": 100
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422
