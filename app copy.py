# Hamilton County Property Value Predictor
# Educational demonstration only

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# App Title and Description
st.title("Hamilton County Property Value Predictor")
st.write("""
This app predicts **APPRAISED_VALUE** for residential properties in Hamilton County, TN
using a machine learning regression model.""")


# Load Data
import zipfile
with zipfile.ZipFile("AssessorExportCSV.zip") as z:
    df = pd.read_csv(z.open("AssessorExportCSV.csv"))

# Data Cleaning
df = df[df["APPRAISED_VALUE"].notna()]
df = df[df["APPRAISED_VALUE"] > 0]

# Filter residential properties
df = df[df["PROPERTY_TYPE_CODE_DESC"] == "Residential"]

# Select features
features = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]

df = df[features + ["APPRAISED_VALUE"]].dropna()

X = df[features]
y = df["APPRAISED_VALUE"]

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# User Input Section
st.header("Enter Property Information")

land_value = st.number_input("Land Value ($)", min_value=0.0, value=50000.0, step=1000.0)
build_value = st.number_input("Building Value ($)", min_value=0.0, value=150000.0, step=1000.0)
yard_value = st.number_input("Yard Items Value ($)", min_value=0.0, value=5000.0, step=500.0)
acres = st.number_input("Lot Size (Acres)", min_value=0.0, value=0.25, step=0.01)

input_data = pd.DataFrame([[land_value, build_value, yard_value, acres]],
                          columns=features)

st.subheader("User Input Data")
st.write(input_data)


# Prediction
if st.button("Predict Appraised Value"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Appraised Value: ${prediction:,.2f}")
