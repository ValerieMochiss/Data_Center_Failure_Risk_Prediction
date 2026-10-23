import pandas as pd
import pickle
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Data Center Failure Risk Predictor", layout="wide"
)


# Cache the model and encoder loading so they only load ONCE
@st.cache_resource
def load_artifacts():
  with open("gb_model.pkl", "rb") as f:
    model = pickle.load(f)
  with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
  return model, label_encoder


# Load artifacts
model, label_encoder = load_artifacts()

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Server Parameters")
st.sidebar.markdown("Adjust the characteristics to evaluate failure risk.")

server_age = st.sidebar.number_input("Server age (years)", 0.0, 15.0, 5.0, 0.5)
battery_health = st.sidebar.number_input(
    "UPS battery health (%)", 0.0, 100.0, 80.0, 1.0
)
psu_efficiency = st.sidebar.number_input(
    "Power supply efficiency (%)", 0.0, 100.0, 90.0, 1.0
)
exhaust_temp = st.sidebar.number_input(
    "Exhaust temperature (°C)", 0.0, 100.0, 40.0, 1.0
)

predict_btn = st.sidebar.button(
    "Predict Risk", type="primary", use_container_width=True
)

# --- MAIN PAGE LAYOUT ---
st.title("🛡️ Data Center Failure Risk Predictor")
st.markdown(
    "Welcome to the server health monitoring dashboard. Use the sidebar to"
    " input telemetry data and run predictions."
)

if predict_btn:
  # Prepare features for model
  features = pd.DataFrame([{
      "server_age_years": server_age,
      "ups_battery_health": battery_health,
      "power_supply_efficiency": psu_efficiency,
      "exhaust_temperature_c": exhaust_temp,
  }])

  prediction = model.predict(features)[0]
  proba = model.predict_proba(features)[0]
  risk = label_encoder.inverse_transform([prediction])[0]

  # Layout results into columns
  col1, col2 = st.columns([1, 1])

  with col1:
    st.subheader("Prediction Result")

    # Color-coded risk indicator
    risk_lower = str(risk).lower()
    if any(keyword in risk_lower for keyword in ["high", "critical", "danger"]):
      st.error(f"🚨 Predicted Risk: **{str(risk).upper()}**")
    elif any(keyword in risk_lower for keyword in ["medium", "moderate"]):
      st.warning(f"⚠️ Predicted Risk: **{str(risk).capitalize()}**")
    else:
      st.success(f"✅ Predicted Risk: **{str(risk).capitalize()}**")

    # Show confidence
    classes = label_encoder.classes_
    idx = list(classes).index(risk)
    st.metric("Model Confidence", f"{proba[idx]:.2%}")

    # Probability breakdown table
    st.write("**Probability breakdown:**")
    prob_df = pd.DataFrame(
        {"Risk Level": classes, "Probability": [f"{p:.2%}" for p in proba]}
    )
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

  with col2:
    st.subheader("Server Health Profile")

    # Preparing an easily readable horizontal bar chart
    # We normalize/format metrics so 100% is always "Ideal/Healthy"
    chart_data = pd.DataFrame({
        "Metric": [
            "Server Age (Remaining life)",
            "Battery Health",
            "PSU Efficiency",
            "Exhaust Temp (Coolness)",
        ],
        "Score (%)": [
            max(
                0, 100 - (server_age / 15.0 * 100)
            ),  # Lower age is better -> 15 yrs = 0%
            battery_health,  # Higher is better
            psu_efficiency,  # Higher is better
            max(
                0, 100 - (exhaust_temp / 100.0 * 100)
            ),  # Lower temp is better -> 100°C = 0%
        ],
    })

    fig = px.bar(
        chart_data,
        x="Score (%)",
        y="Metric",
        orientation="h",
        range_x=[0, 100],
        text="Score (%)",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside", marker_color="#2563eb"
    )
    fig.update_layout(
        xaxis_title="Health / Ideal Score (%)",
        yaxis_title="",
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
  st.info(
      "👈 Configure your server parameters in the sidebar and click **Predict"
      " Risk** to see the analysis."
  )

  # Feature Impact / Contribution to the Prediction
st.subheader("Feature Impact on Prediction")

    # Get model feature importances (if supported by your model, e.g., GradientBoosting)
if hasattr(model, "feature_importances_"):
      importances = model.feature_importances_
      # Scale importances by current input values relative to their max limits to show directional weight
      normalized_inputs = [
          server_age / 15.0,
          (100 - battery_health) / 100.0,  # lower battery is worse
          (100 - psu_efficiency) / 100.0,  # lower psu is worse
          exhaust_temp / 100.0,  # higher temp is worse
      ]
      impact_scores = importances * normalized_inputs

      impact_df = pd.DataFrame({
          "Feature": [
              "Server Age",
              "Low Battery Health",
              "Low PSU Efficiency",
              "High Exhaust Temp",
          ],
          "Impact Score": impact_scores,
      })

      fig_impact = px.bar(
          impact_df,
          x="Impact Score",
          y="Feature",
          orientation="h",
          title="Relative Risk Drivers",
      )
      fig_impact.update_layout(
          xaxis_title="Contribution Weight",
          yaxis_title="",
          margin=dict(t=30, b=20, l=20, r=20),
          height=250,
      )
      st.plotly_chart(fig_impact, use_container_width=True)