import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "final_fraud_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Header
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter the transaction details below to determine "
    "whether the transaction is potentially fraudulent."
)

st.divider()

# -----------------------------
# Basic Transaction Details
# -----------------------------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0
    )

with col2:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

# -----------------------------
# V1 - V28 Features
# -----------------------------
st.subheader("Transaction Features")

features = {}

cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Predict Transaction", use_container_width=True):

    input_data = {
        "Time": time
    }

    input_data.update(features)

    input_data["Amount"] = amount

    input_df = pd.DataFrame([input_data])

    # Exact feature order used during training
    input_df = input_df[
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    ]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "This transaction has been classified as potentially fraudulent."
        )

    else:
        st.success("✅ LEGITIMATE TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "This transaction has been classified as legitimate."
        )
