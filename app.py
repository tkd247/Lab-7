# Hamilton County Property Value Predictor

# Educational demonstration only

import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# App Title

st.title("Hamilton County Property Value Predictor")

st.write("""
This application predicts **APPRAISED_VALUE** for properties in Hamilton County,
Tennessee using a machine learning regression model.

Educational use only.
""")

# -------------------------

# Load Dataset

# -------------------------

@st.cache_data
def load_data():

```
with zipfile.ZipFile("AssessorExportCSV.zip") as z:

    filename = [f for f in z.namelist() if f.endswith(".csv")][0]

    df = pd.read_csv(z.open(filename), low_memory=False)

return df
```

df = load_data()

st.write("Dataset Shape:", df.shape)

# -------------------------

# Convert Numeric Columns

# -------------------------

numeric_cols = [
"APPRAISED_VALUE",
"LAND_VALUE",
"BUILD_VALUE",
"YARDITEMS_VALUE",
"CALC_ACRES"
]

for col in numeric_cols:

```
if col in df.columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")
```

# -------------------------

# Data Cleaning

# -------------------------

df = df.dropna(subset=["APPRAISED_VALUE"])
df = df[df["APPRAISED_VALUE"] > 1000]

df["LAND_VALUE"] = df["LAND_VALUE"].fillna(0)
df["BUILD_VALUE"] = df["BUILD_VALUE"].fillna(0)
df["YARDITEMS_VALUE"] = df["YARDITEMS_VALUE"].fillna(0)
df["CALC_ACRES"] = df["CALC_ACRES"].fillna(0)

# -------------------------

# Feature Selection

# -------------------------

features = [
"LAND_VALUE",
"BUILD_VALUE",
"YARDITEMS_VALUE",
"CALC_ACRES"
]

X = df[features]
y = df["APPRAISED_VALUE"]

# -------------------------

# Train/Test Split

# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -------------------------

# Train Model

# -------------------------

model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

# -------------------------

# Model Evaluation

# -------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"Mean Absolute Error (MAE): **${mae:,.2f}**")
st.write(f"R² Score: **{r2:.3f}**")

# -------------------------

# Graph

# -------------------------

st.subheader("Distribution of Property Values")

fig, ax = plt.subplots()

ax.hist(df["APPRAISED_VALUE"], bins=50)

ax.set_xlabel("Appraised Value ($)")
ax.set_ylabel("Number of Properties")
ax.set_title("Hamilton County Property Value Distribution")

st.pyplot(fig)

# -------------------------

# User Inputs

# -------------------------

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

input_data = pd.DataFrame(
[[land_value, build_value, yard_value, acres]],
columns=features
)

st.subheader("User Input Data")
st.write(input_data)

# -------------------------

# Prediction

# -------------------------

if st.button("Predict Appraised Value"):

```
prediction = model.predict(input_data)[0]

st.success(f"Estimated Appraised Value: **${prediction:,.2f}**")
```
