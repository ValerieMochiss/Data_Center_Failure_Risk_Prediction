import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Failure Risk Predictor", layout="wide")


# Cache the model and encoder loading so they only load ONCE
@st.cache_resource
def load_artifacts():
  with open("gb_model.pkl", "rb") as f:
    model = pickle.load(f)
  with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
  return model, label_encoder


# Load artifacts using the cached function
model, label_encoder = load_artifacts()

st.title("Data Center Failure Risk Predictor")
st.markdown("Enter the server's characteristics to predict its failure risk.")

# User inputs for the four features
col1, col2 = st.columns(2)
with col1:
  server_age = st.number_input("Server age (years)", 0.0, 15.0, 5.0, 0.5)
  battery_health = st.number_input("UPS battery health (%)", 0.0, 100.0, 80.0, 1.0)
with col2:
  psu_efficiency = st.number_input(
      "Power supply efficiency (%)", 0.0, 100.0, 90.0, 1.0
  )
  exhaust_temp = st.number_input("Exhaust temperature (°C)", 0.0, 100.0, 40.0, 1.0)

# Predict button
if st.button("Predict Risk"):
  features = pd.DataFrame([{
      "server_age_years": server_age,
      "ups_battery_health": battery_health,
      "power_supply_efficiency": psu_efficiency,
      "exhaust_temperature_c": exhaust_temp,
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
  prob_df = pd.DataFrame(
      {"Risk": classes, "Probability": [f"{p:.2%}" for p in proba]}
  )
  st.dataframe(prob_df)
