# Hamilton County Property Value Predictor
# Educational demonstration only

import streamlit as st
import pandas as pd
import zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------
# App Title
# ----------------------------------------------------

st.title("Hamilton County Property Value Predictor")

st.write("""
This application predicts **APPRAISED_VALUE** for residential properties in
Hamilton County, Tennessee using a machine learning regression model.
""")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile("AssessorExportCSV.zip") as z:
            filename = [f for f in z.namelist() if f.endswith(".csv")][0]
            df = pd.read_csv(z.open(filename), low_memory=False)
        return df
    except Exception as e:
        st.error("Error loading dataset. Make sure AssessorExportCSV.zip is uploaded.")
        st.stop()

df = load_data()

# Show dataset info for debugging
st.write("Dataset Shape:", df.shape)
st.write("Available Columns:", list(df.columns))

# ----------------------------------------------------
# Convert Columns to Numeric
# ----------------------------------------------------

columns_to_convert = [
    "APPRAISED_VALUE",
    "LAND_VALUE",
    "BUILD_VALUE",
    "YARDITEMS_VALUE",
    "CALC_ACRES"
]

for col in columns_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------
# Clean Data
# ----------------------------------------------------

if "APPRAISED_VALUE" not in df.columns:
    st.error("APPRAISED_VALUE column not found in dataset.")
    st.stop()

df = df[df["APPRAISED_VALUE"].notna()]
df = df[df["APPRAISED_VALUE"] > 0]

# Features used in the model
features = [
    "LAND_VALUE",
    "BUILD_VALUE",
    "YARDITEMS_VALUE",
    "CALC_ACRES"
]

# Only keep features that exist in dataset
available_features = [f for f in features if f in df.columns]

if len(available_features) == 0:
    st.error("None of the required feature columns were found in the dataset.")
    st.stop()

df = df[available_features + ["APPRAISED_VALUE"]].dropna()

if df.empty:
    st.error("Dataset became empty after cleaning. Please check missing values.")
    st.stop()

# ----------------------------------------------------
# Prepare Model Data
# ----------------------------------------------------

X = df[available_features]
y = df["APPRAISED_VALUE"]

# Ensure dataset is large enough
if len(df) < 10:
    st.error("Not enough data available to train the model.")
    st.stop()

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------------------
# Model Evaluation
# ----------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"Mean Absolute Error (MAE): **${mae:,.2f}**")
st.write(f"R² Score: **{r2:.3f}**")

# ----------------------------------------------------
# User Input Section
# ----------------------------------------------------

st.header("Enter Property Information")

land_value = st.number_input(
    "Land Value ($)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

build_value = st.number_input(
    "Building Value ($)",
    min_value=0.0,
    value=150000.0,
    step=1000.0
)

yard_value = st.number_input(
    "Yard Items Value ($)",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

acres = st.number_input(
    "Lot Size (Acres)",
    min_value=0.0,
    value=0.25,
    step=0.01
)

# Create dataframe from user inputs
input_data = pd.DataFrame(
    [[land_value, build_value, yard_value, acres]],
    columns=features
)

st.subheader("User Input Data")
st.write(input_data)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if st.button("Predict Appraised Value"):
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Appraised Value: **${prediction:,.2f}**")
