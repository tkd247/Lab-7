# Hamilton County Property Value Predictor

import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.title("Hamilton County Property Value Predictor")

st.write(
    "This application predicts APPRAISED_VALUE for properties in Hamilton County, Tennessee using a machine learning regression model."
)


@st.cache_data
def load_data():
    with zipfile.ZipFile("AssessorExportCSV.zip") as z:
        filename = [f for f in z.namelist() if f.endswith(".csv")][0]
        df = pd.read_csv(z.open(filename), low_memory=False)
    return df


df = load_data()

st.write("Dataset Shape:", df.shape)


numeric_cols = [
    "APPRAISED_VALUE",
    "LAND_VALUE",
    "BUILD_VALUE",
    "YARDITEMS_VALUE",
    "CALC_ACRES"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(subset=["APPRAISED_VALUE"])

df["LAND_VALUE"] = df["LAND_VALUE"].fillna(0)
df["BUILD_VALUE"] = df["BUILD_VALUE"].fillna(0)
df["YARDITEMS_VALUE"] = df["YARDITEMS_VALUE"].fillna(0)
df["CALC_ACRES"] = df["CALC_ACRES"].fillna(0)


features = [
    "LAND_VALUE",
    "BUILD_VALUE",
    "YARDITEMS_VALUE",
    "CALC_ACRES"
]


df = df[features + ["APPRAISED_VALUE"]]

X = df[features]
y = df["APPRAISED_VALUE"]


if len(df) < 10:
    st.error("Dataset became too small after cleaning.")
    st.stop()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


st.subheader("Model Performance")

st.write(f"Mean Absolute Error (MAE): ${mae:,.2f}")
st.write(f"R² Score: {r2:.3f}")


st.subheader("Distribution of Property Values")

fig, ax = plt.subplots()

ax.hist(df["APPRAISED_VALUE"], bins=50)

ax.set_xlabel("Appraised Value ($)")
ax.set_ylabel("Number of Properties")

st.pyplot(fig)


st.header("Enter Property Information")

land_value = st.number_input("Land Value ($)", 0.0, 10000000.0, 50000.0)
build_value = st.number_input("Building Value ($)", 0.0, 10000000.0, 150000.0)
yard_value = st.number_input("Yard Items Value ($)", 0.0, 1000000.0, 5000.0)
acres = st.number_input("Lot Size (Acres)", 0.0, 100.0, 0.25)


input_data = pd.DataFrame(
    [[land_value, build_value, yard_value, acres]],
    columns=features
)


if st.button("Predict Appraised Value"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Appraised Value: ${prediction:,.2f}")
