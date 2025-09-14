# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64

# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="Climate Risk Guide Platform",
    layout="wide",
    page_icon="🌍"
)

# =========================================
# Load Models & Scalers
# =========================================
rf_flood = joblib.load("rf_flood.pkl")
scaler_flood = joblib.load("scaler_flood.pkl")

rf_heat = joblib.load("rf_heatwave.pkl")
scaler_heat = joblib.load("scaler_heatwave.pkl")

# =========================================
# Load External CSS
# =========================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# =========================================
# Title & Intro
# =========================================
st.markdown("<h1 style='text-align:center;'>🌍 Climate Risk Guide: Flood & Heatwave Hazards</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Predict <b>Flood</b> and <b>Heatwave</b> risks using environmental, climate, and social indicators."
    "</p>",
    unsafe_allow_html=True
)

# =========================================
# Sidebar Inputs
# =========================================
st.sidebar.header("⚙️ Input Parameters")

# --- Flood Inputs ---
with st.sidebar.expander("🌊 Flood Features", expanded=True):
    flood_input = {
        "MonsoonIntensity": st.slider("Monsoon Intensity", 1, 10, 5),
        "TopographyDrainage": st.slider("Topography & Drainage", 1, 10, 5),
        "RiverManagement": st.slider("River Management", 1, 10, 5),
        "Deforestation": st.slider("Deforestation Level", 1, 10, 5),
        "Urbanization": st.slider("Urbanization", 1, 10, 5),
        "ClimateChange": st.slider("Climate Change Impact", 1, 10, 5),
        "DamsQuality": st.slider("Dams Quality", 1, 10, 5),
        "Siltation": st.slider("Siltation", 1, 10, 5),
        "AgriculturalPractices": st.slider("Agricultural Practices", 1, 10, 5),
        "Encroachments": st.slider("Encroachments", 1, 10, 5),
        "IneffectiveDisasterPreparedness": st.slider("Disaster Preparedness", 1, 10, 5),
        "DrainageSystems": st.slider("Drainage Systems", 1, 10, 5),
        "CoastalVulnerability": st.slider("Coastal Vulnerability", 1, 10, 5),
        "Landslides": st.slider("Landslides Risk", 1, 10, 5),
        "Watersheds": st.slider("Watersheds Condition", 1, 10, 5),
        "DeterioratingInfrastructure": st.slider("Infrastructure Condition", 1, 10, 5),
        "PopulationScore": st.slider("Population Density Score", 1, 10, 5),
        "WetlandLoss": st.slider("Wetland Loss", 1, 10, 5),
        "InadequatePlanning": st.slider("Planning Quality", 1, 10, 5),
        "PoliticalFactors": st.slider("Political Factors", 1, 10, 5)
    }
flood_input_df = pd.DataFrame([flood_input])

# --- Heatwave Inputs ---
with st.sidebar.expander("🔥 Heatwave Features", expanded=True):
    heat_input = {
        "wind_speed": st.slider("Wind Speed (km/h)", 0, 100, 10),
        "cloud_cover": st.slider("Cloud Cover (%)", 0, 100, 50),
        "precipitation_probability": st.slider("Precipitation Probability (%)", 0, 100, 50),
        "pressure_surface_level": st.slider("Pressure Surface Level (hPa)", 900, 1100, 1013),
        "dew_point": st.slider("Dew Point (°C)", -10, 50, 25),
        "uv_index": st.slider("UV Index", 0, 15, 5),
        "visibility": st.slider("Visibility (km)", 0, 20, 10),
        "rainfall": st.slider("Rainfall (mm)", 0, 500, 50),
        "solar_radiation": st.slider("Solar Radiation (W/m²)", 0, 1200, 600),
        "snowfall": st.slider("Snowfall (mm)", 0, 50, 0),
        "max_temperature": st.slider("Max Temperature (°C)", 20, 50, 35),
        "min_temperature": st.slider("Min Temperature (°C)", 10, 40, 25),
        "max_humidity": st.slider("Max Humidity (%)", 0, 100, 70),
        "min_humidity": st.slider("Min Humidity (%)", 0, 100, 40)
    }
heat_input_df = pd.DataFrame([heat_input])

# =========================================
# Helper Function: Bigger Chart
# =========================================
def bigger_chart(title, labels, values, colors):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))  # slightly bigger
    ax.bar(labels, values, color=colors, width=0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def display_centered_image(buf):
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{img_base64}'></div>", unsafe_allow_html=True)

# =========================================
# Flood Prediction
# =========================================
st.markdown("---")
st.subheader("🌊 Flood Risk Prediction")

if st.button("Predict Flood Risk", type="primary"):
    X_f_scaled = scaler_flood.transform(flood_input_df)
    pred_f = rf_flood.predict(X_f_scaled)
    prob_f = rf_flood.predict_proba(X_f_scaled)[0]

    if pred_f[0] == 1:
        st.markdown("<div class='result-card flood'>⚠️ HIGH FLOOD RISK</div>", unsafe_allow_html=True)
        st.markdown("""
        **Safety Measures:**
        - Avoid low-lying areas and river banks.
        - Prepare emergency kit (food, water, medicines).
        - Keep updated with local weather alerts.
        - Ensure family & community evacuation plan.
        """)
    else:
        st.markdown("<div class='result-card flood'>✅ LOW FLOOD RISK</div>", unsafe_allow_html=True)
        st.markdown("No immediate flood risk. Stay alert and monitor weather updates.")

    buf = bigger_chart("Flood Risk", ["Low", "High"], [prob_f[0]*100, prob_f[1]*100], ["#a3cef1", "#6fa8dc"])
    display_centered_image(buf)

# =========================================
# Heatwave Prediction
# =========================================
st.markdown("---")
st.subheader("🔥 Heatwave Risk Prediction")

if st.button("Predict Heatwave Risk", type="primary"):
    heat_features = [
        "wind_speed", "cloud_cover", "precipitation_probability", "pressure_surface_level",
        "dew_point", "uv_index", "visibility", "rainfall", "solar_radiation", "snowfall",
        "max_temperature", "min_temperature", "max_humidity", "min_humidity"
    ]

    X_h_input = heat_input_df[heat_features]
    X_h_scaled = scaler_heat.transform(X_h_input)
    pred_h = rf_heat.predict(X_h_scaled)
    prob_h = rf_heat.predict_proba(X_h_scaled)[0]

    if pred_h[0] == 1:
        st.markdown("<div class='result-card heatwave'>⚠️ HIGH HEATWAVE RISK</div>", unsafe_allow_html=True)
        st.markdown("""
        **Safety Measures:**
        - Stay hydrated and avoid outdoor activity during peak heat hours.
        - Use sunscreen, hats, and protective clothing.
        - Check on vulnerable people (elderly, children).
        - Keep your home cool and ventilated.
        """)
    else:
        st.markdown("<div class='result-card heatwave'>✅ LOW HEATWAVE RISK</div>", unsafe_allow_html=True)
        st.markdown("No immediate heatwave risk. Stay safe and monitor weather alerts.")

    buf = bigger_chart("Heatwave Risk", ["Low", "High"], [prob_h[0]*100, prob_h[1]*100], ["#f4c2c2", "#f19292"])
    display_centered_image(buf)
