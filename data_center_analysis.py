import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Failure Risk Predictor", layout="wide")

# Load the trained model and encoder
model = joblib.load('gb_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title("Data Center Failure Risk Predictor")

st.markdown("Enter the server's characteristics to predict its failure risk.")

# User inputs for the four features
col1, col2 = st.columns(2)
with col1:
    server_age = st.number_input("Server age (years)", 0.0, 15.0, 5.0, 0.5)
    battery_health = st.number_input("UPS battery health (%)", 0.0, 100.0, 80.0, 1.0)
with col2:
    psu_efficiency = st.number_input("Power supply efficiency (%)", 0.0, 100.0, 90.0, 1.0)
    exhaust_temp = st.number_input("Exhaust temperature (°C)", 0.0, 100.0, 40.0, 1.0)

# Predict button
if st.button("Predict Risk"):
    features = pd.DataFrame([{
        'server_age_years': server_age,
        'ups_battery_health': battery_health,
        'power_supply_efficiency': psu_efficiency,
        'exhaust_temperature_c': exhaust_temp
    }])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    risk = label_encoder.inverse_transform([prediction])[0]

    # Show result
    st.subheader("Prediction Result")
    st.metric("Predicted Risk", risk)

    # Show confidence for the predicted class
    classes = label_encoder.classes_
    idx = list(classes).index(risk)
    st.metric("Confidence", f"{proba[idx]:.2%}")

    # Probability breakdown
    st.write("**Probability breakdown:**")
    prob_df = pd.DataFrame({
        'Risk': classes,
        'Probability': [f"{p:.2%}" for p in proba]
    })
    st.dataframe(prob_df)
