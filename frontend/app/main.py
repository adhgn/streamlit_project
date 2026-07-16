# import streamlit as st
# import requests
# import plotly.express as px
# import pandas as pd

# st.set_page_config(page_title="Fleet Analytics", layout="wide")

# try:
#     BACKEND_URL = st.secrets["BACKEND_URL"]
# except (FileNotFoundError, KeyError, UserWarning):
#     BACKEND_URL = "http://localhost:8000"

# st.title("🚜 Machinery Health Simulator")
# st.write("Adjust mechanical telemetry parameters derived from the Kaggle Predictive Maintenance dataset framework.")

# col_inputs, col_outputs = st.columns([1, 1.2])

# with col_inputs:
#     st.subheader("🔧 Real-time Sensor Controls")
#     air_temp = st.slider("Air Temperature (Kelvin)", 295.0, 305.0, 298.0, 0.1)
#     proc_temp = st.slider("Process Temperature (Kelvin)", 305.0, 315.0, 308.0, 0.1)
#     rot_speed = st.slider("Rotational Speed (RPM)", 1100, 2900, 1500, 10)
#     torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
#     tool_wear = st.slider("Tool Wear (Minutes)", 0, 250, 40, 1)

# with col_outputs:
#     st.subheader("📊 Live Predictive Analytics")
    
#     # Payload keys perfectly map to our lowercase Pydantic backend definitions
#     payload = {
#         "air_temperature": air_temp,
#         "process_temperature": proc_temp,
#         "rotational_speed": rot_speed,
#         "torque": torque,
#         "tool_wear": tool_wear
#     }
    
#     try:
#         response = requests.post(f"{BACKEND_URL}/predict", json=payload)
        
#         if response.status_code == 200:
#             result = response.json()
#             prob = result["failure_probability_percent"]
            
#             if result["failure_risk_detected"] or prob > 50:
#                 st.error(f"⚠️ HIGH BREAKDOWN RISK: {prob}% Probability calculated by model.")
#             else:
#                 st.success(f"✅ SYSTEM STABLE: {prob}% Probability of Component Failure.")
            
#             st.metric(label="Live Model Failure Risk Estimate", value=f"{prob}%")
            
#             st.markdown("---")
#             st.write("##### 🔍 Telemetry Stress Factor Breakdown")
            
#             factors = result["risk_factors"]
#             fig = px.bar(
#                 x=list(factors.values()),
#                 y=list(factors.keys()),
#                 orientation='h',
#                 labels={'x': 'Relative Stress Intensity', 'y': 'Sensor Domain'},
#                 color=list(factors.values()),
#                 color_continuous_scale='Reds'
#             )
#             fig.update_layout(showlegend=False, height=250, margin=dict(l=20, r=20, t=20, b=20))
#             st.plotly_chart(fig, use_container_width=True)
            
#         else:
#             st.warning("API connection established but model calculations are loading...")
            
#     except requests.exceptions.ConnectionError:
#         st.info("💡 Running in Local Development: Spin up the FastAPI server on port 8000 to feed live data inputs to the model.")

# st.write("---")
# st.header("📋 Recent Machinery Scan Logs")

# if st.button("🔄 Refresh Historical Logs"):
#     try:
#         # Call the new history endpoint
#         response = requests.get(f"{BACKEND_URL}/history?limit=10")
#         if response.status_code == 200:
#             history_data = response.json()
            
#             if not history_data:
#                 st.info("No scans recorded in the database yet. Adjust sliders to create some data!")
#             else:
#                 # Format into a clean Pandas DataFrame for display
#                 flat_history = []
#                 for record in history_data:
#                     row = {
#                         "Timestamp (UTC)": record["timestamp"],
#                         "Risk Detected": "⚠️ FAILURE" if record["risk_detected"] else "✅ NORMAL",
#                         "Probability": f"{record['probability']}%"
#                     }
#                     # Unpack telemetry items
#                     for k, v in record["telemetry"].items():
#                         row[k] = v
#                     flat_history.append(row)
                
#                 df_logs = pd.DataFrame(flat_history)
                
#                 # Render an elegant, sortable data table directly in the UI
#                 st.dataframe(df_logs, use_container_width=True, hide_index=True)
#         else:
#             st.error("Could not fetch log history from the server backend.")
#     except Exception as e:
#         st.error(f"Network error connecting to database history pipeline: {e}")

import streamlit as st
import requests
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Fleet Analytics", layout="wide")

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (FileNotFoundError, KeyError, UserWarning):
    BACKEND_URL = "http://localhost:8000"

# 🧠 1. Initialize Streamlit Session State keys so they persist across reruns
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "api_error" not in st.session_state:
    st.session_state.api_error = None

st.title("🚜 Machinery Health Simulator")
st.write("Adjust mechanical telemetry parameters derived from the Kaggle Predictive Maintenance dataset framework.")

col_inputs, col_outputs = st.columns([1, 1.2])

with col_inputs:
    st.subheader("🔧 Real-time Sensor Controls")
    air_temp = st.slider("Air Temperature (Kelvin)", 295.0, 305.0, 298.0, 0.1)
    proc_temp = st.slider("Process Temperature (Kelvin)", 305.0, 315.0, 308.0, 0.1)
    rot_speed = st.slider("Rotational Speed (RPM)", 1100, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    tool_wear = st.slider("Tool Wear (Minutes)", 0, 250, 40, 1)
    
    st.markdown("---")
    # 🎛️ 2. The explicit execution trigger button
    if st.button("🚀 Run Diagnostics", type="primary", use_container_width=True):
        payload = {
            "air_temperature": air_temp,
            "process_temperature": proc_temp,
            "rotational_speed": rot_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }
        
        with st.spinner("Communicating with ML core..."):
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                if response.status_code == 200:
                    st.session_state.prediction_result = response.json()
                    st.session_state.api_error = None
                else:
                    st.session_state.prediction_result = "loading"
                    st.session_state.api_error = None
            except requests.exceptions.ConnectionError:
                st.session_state.prediction_result = "offline"
                st.session_state.api_error = None

with col_outputs:
    st.subheader("📊 Live Predictive Analytics")
    
    # 📈 3. Render outputs strictly based on what is stored in Session State
    if st.session_state.prediction_result is None:
        st.info("💡 Adjust the sensor values on the left and click 'Run Diagnostics' to test machinery stability.")
        
    elif st.session_state.prediction_result == "loading":
        st.warning("API connection established but model calculations are loading...")
        
    elif st.session_state.prediction_result == "offline":
        st.info("💡 Running in Local Development: Spin up the FastAPI server on port 8000 to feed live data inputs to the model.")
        
    else:
        # We have a valid dictionary object saved in state memory
        result = st.session_state.prediction_result
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

# --- HISTORICAL LOGS ---
st.write("---")
st.header("📋 Recent Machinery Scan Logs")

if st.button("🔄 Refresh Historical Logs"):
    try:
        response = requests.get(f"{BACKEND_URL}/history?limit=10")
        if response.status_code == 200:
            history_data = response.json()
            
            if not history_data:
                st.info("No scans recorded in the database yet. Adjust sliders and run a diagnostic to log data!")
            else:
                flat_history = []
                for record in history_data:
                    row = {
                        "Timestamp (UTC)": record["timestamp"],
                        "Risk Detected": "⚠️ FAILURE" if record["risk_detected"] else "✅ NORMAL",
                        "Probability": f"{record['probability']}%"
                    }
                    for k, v in record["telemetry"].items():
                        row[k] = v
                    flat_history.append(row)
                
                df_logs = pd.DataFrame(flat_history)
                st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.error("Could not fetch log history from the server backend.")
    except Exception as e:
        st.error(f"Network error connecting to database history pipeline: {e}")