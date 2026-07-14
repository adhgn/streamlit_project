import streamlit as st
import requests
import plotly.express as px

st.set_page_config(page_title="UT Fleet Analytics", layout="wide")

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (FileNotFoundError, KeyError, UserWarning):
    BACKEND_URL = "http://localhost:8002"

st.title("🚜 United Tractors Machinery Health Simulator")
st.write("Adjust mechanical telemetry parameters derived from the Kaggle Predictive Maintenance dataset framework.")

col_inputs, col_outputs = st.columns([1, 1.2])

with col_inputs:
    st.subheader("🔧 Real-time Sensor Controls")
    air_temp = st.slider("Air Temperature (Kelvin)", 295.0, 305.0, 298.0, 0.1)
    proc_temp = st.slider("Process Temperature (Kelvin)", 305.0, 315.0, 308.0, 0.1)
    rot_speed = st.slider("Rotational Speed (RPM)", 1100, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    tool_wear = st.slider("Tool Wear (Minutes)", 0, 250, 40, 1)

with col_outputs:
    st.subheader("📊 Live Predictive Analytics")
    
    # Payload keys perfectly map to our lowercase Pydantic backend definitions
    payload = {
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["failure_probability_percent"]
            
            if result["failure_risk_detected"] or prob > 50:
                st.error(f"⚠️ HIGH BREAKDOWN RISK: {prob}% Probability calculated by model.")
            else:
                st.success(f"✅ SYSTEM STABLE: {prob}% Probability of Component Failure.")
            
            st.metric(label="Live Model Failure Risk Estimate", value=f"{prob}%")
            
            st.markdown("---")
            st.write("##### 🔍 Telemetry Stress Factor Breakdown")
            
            factors = result["risk_factors"]
            fig = px.bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation='h',
                labels={'x': 'Relative Stress Intensity', 'y': 'Sensor Domain'},
                color=list(factors.values()),
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("API connection established but model calculations are loading...")
            
    except requests.exceptions.ConnectionError:
        st.info("💡 Running in Local Development: Spin up the FastAPI server on port 8000 to feed live data inputs to the model.")