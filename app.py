import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("house_price_prediction_model.pkl")
st.title("🏠 House Price Prediction")
st.markdown("### Predict the selling price of a house using Machine Learning")

st.markdown("---")

st.sidebar.title("🏠 House Price Prediction")
st.sidebar.info(
    """
    This application predicts the selling price of a house using a trained
    Linear Regression model.
    """
)

# -------------------------------
# Create Two Columns
# -------------------------------
col1, col2 = st.columns(2)

# ===============================
# LEFT COLUMN
# ===============================
with col1:

    st.subheader("📍 Property Details")

    overall_qual = st.slider(
        "Overall Quality",
        1,
        10,
        5
    )

    overall_cond = st.slider(
        "Overall Condition",
        1,
        10,
        5
    )

    year_built = st.number_input(
        "Year Built",
        1870,
        2025,
        2000
    )

    gr_liv_area = st.number_input(
        "Ground Living Area (sq ft)",
        300,
        6000,
        1500
    )

# ===============================
# RIGHT COLUMN
# ===============================
with col2:

    st.subheader("🚗 Garage Details")

    garage_cars = st.number_input(
        "Garage Capacity",
        0,
        5,
        2
    )

    garage_area = st.number_input(
        "Garage Area",
        0,
        2000,
        500
    )

    lot_area = st.number_input(
        "Lot Area",
        1000,
        100000,
        8000
    )

    lot_frontage = st.number_input(
        "Lot Frontage",
        20,
        300,
        70
    )

st.markdown("---")

# -------------------------------
# Predict Button
# -------------------------------
predict = st.button(
    "🔍 Predict House Price",
    use_container_width=True
)

if predict:

    st.success("✅ UI is ready! Prediction logic will be added in the next step.")

import pandas as pd
import json

train_df = pd.read_csv("house price data/train.csv")   # Adjust path if needed

X = train_df.drop("SalePrice", axis=1)

default_values = X.mode().iloc[0].to_dict()

with open("default_values.json", "w") as f:
    json.dump(default_values, f, indent=4)



import json

with open("default_values.json", "r") as f:
    default_values = json.load(f)

if predict:

    # Copy default values
    input_data = default_values.copy()

    # Update values entered by user
    input_data["OverallQual"] = overall_qual
    input_data["OverallCond"] = overall_cond
    input_data["YearBuilt"] = year_built
    input_data["GrLivArea"] = gr_liv_area
    input_data["GarageCars"] = garage_cars
    input_data["GarageArea"] = garage_area
    input_data["LotArea"] = lot_area
    input_data["LotFrontage"] = lot_frontage

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)

    # Convert back from log scale
    predicted_price = np.expm1(prediction[0])

    st.success(f"🏠 Estimated House Price: ${predicted_price:,.2f}")