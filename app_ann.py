# =========================================================
# PENICILLIN YIELD PREDICTOR — STREAMLIT APP (ANN VERSION)
# =========================================================
# Needs 3 files in the same folder as this script:
#   penicillin_ann_model.keras, scaler_X.pkl, scaler_y.pkl
# None of these require the original dataset at runtime.

import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# ---------------------------------------------------------
# Load model + scalers ONCE and cache them in memory.
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("penicillin_ann_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_artifacts()

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Penicillin Yield Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Penicillin Yield Predictor")
st.write(
    "Enter your fermentation conditions below to get an instant predicted "
    "penicillin yield, based on an Artificial Neural Network trained on "
    "100 simulated fermentation batches."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    substrate = st.number_input("Substrate concentration (S: g/L)", 0.0, 50.0, 15.0, 0.5)
    oxygen = st.number_input("Dissolved oxygen concentration (DO2: mg/L)", 0.0, 20.0, 8.0, 0.1)
    ph = st.number_input("pH", 3.0, 9.0, 6.5, 0.1)

with col2:
    temperature = st.number_input("Temperature (T: K)", 280.0, 320.0, 298.0, 0.5)
    time_h = st.number_input("Time (h)", 0.0, 400.0, 150.0, 1.0)

st.divider()

if st.button("Predict Yield", type="primary"):
    # Same column names used during training — must match exactly.
    input_df = pd.DataFrame([{
        "Substrate concentration(S:g/L)": substrate,
        "Dissolved oxygen concentration(DO2:mg/L)": oxygen,
        "pH(pH:pH)": ph,
        "Temperature(T:K)": temperature,
        "Time (h)": time_h,
    }])

    # The ANN was trained on SCALED inputs, so new inputs must be scaled
    # the same way before prediction, then the output un-scaled back to g/L.
    input_scaled = scaler_X.transform(input_df)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]

    st.metric("Predicted Penicillin Concentration", f"{prediction:.2f} g/L")
    st.caption(
        "This is a model estimate based on simulated fermentation data, "
        "not a guarantee of real-world yield."
    )
