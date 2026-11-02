
import pandas as pd
import numpy as np

# Original input columns
FEATURE_COLUMNS = [
    "Gender",
    "Senior_Citizen",
    "Tenure_Months",
    "Contract_Type",
    "Internet_Service",
    "Payment_Method",
    "Monthly_Charges_INR",
    "Total_Charges_INR",
    "Tech_Support",
    "Online_Security",
    "Support_Tickets_6M",
    "Avg_Data_Usage_GB"
]


def preprocess_data(df):

    df = df.copy()

    # Remove customer identifier and target if present
    df = df.drop(
        columns=["Customer_ID", "Churn"],
        errors="ignore"
    )

    # Keep only required input columns
    df = df[FEATURE_COLUMNS]

    # Clean categorical columns
    categorical_columns = [
        "Gender",
        "Senior_Citizen",
        "Contract_Type",
        "Internet_Service",
        "Payment_Method",
        "Tech_Support",
        "Online_Security"
    ]

    for col in categorical_columns:
        df[col] = df[col].apply(
            lambda x: str(x).strip() if pd.notna(x) else np.nan
        )

    # Convert numerical columns
    numerical_columns = [
        "Tenure_Months",
        "Monthly_Charges_INR",
        "Total_Charges_INR",
        "Support_Tickets_6M",
        "Avg_Data_Usage_GB"
    ]

    for col in numerical_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Correct invalid values
    df.loc[
        (df["Tenure_Months"] < 0) |
        (df["Tenure_Months"] > 72),
        "Tenure_Months"
    ] = np.nan

    df.loc[
        (df["Monthly_Charges_INR"] < 0) |
        (df["Monthly_Charges_INR"] > 5000),
        "Monthly_Charges_INR"
    ] = np.nan

    df.loc[
        df["Total_Charges_INR"] < 0,
        "Total_Charges_INR"
    ] = np.nan

    df.loc[
        (df["Support_Tickets_6M"] < 0) |
        (df["Support_Tickets_6M"] > 20),
        "Support_Tickets_6M"
    ] = np.nan

    df.loc[
        (df["Avg_Data_Usage_GB"] < 0) |
        (df["Avg_Data_Usage_GB"] > 500),
        "Avg_Data_Usage_GB"
    ] = np.nan

    # Normalize categorical values
    category_map = {
        "Gender": {
            "male": "Male",
            "female": "Female"
        },
        "Senior_Citizen": {
            "yes": "Yes",
            "no": "No"
        },
        "Tech_Support": {
            "yes": "Yes",
            "no": "No"
        },
        "Online_Security": {
            "yes": "Yes",
            "no": "No"
        },
        "Contract_Type": {
            "month-to-month": "Month-to-month",
            "one year": "One year",
            "two year": "Two year"
        },
        "Internet_Service": {
            "dsl": "DSL",
            "fiber": "Fiber"
        },
        "Payment_Method": {
            "cash": "Cash",
            "upi": "UPI",
            "bank transfer": "Bank Transfer",
            "credit card": "Credit Card"
        }
    }

    for col, mapping in category_map.items():
        df[col] = df[col].apply(
            lambda x: mapping.get(x.lower(), np.nan)
            if isinstance(x, str) else np.nan
        )

    # Feature Engineering 1: Tenure Category
    df["Tenure_Category"] = pd.cut(
        df["Tenure_Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Short-term",
            "Medium-term",
            "Long-term"
        ]
    ).astype("object")

    # Feature Engineering 2: Average Monthly Spending
    df["Average_Monthly_Spending"] = (
        df["Total_Charges_INR"] /
        df["Tenure_Months"].replace(0, np.nan)
    )

    # Feature Engineering 3: Service Usage Indicator
    df["Service_Usage_Indicator"] = np.where(
        df["Avg_Data_Usage_GB"].fillna(0) > 0,
        1,
        0
    )

    return df
