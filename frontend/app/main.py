import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import jwt
import io

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
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None


# --- LOGIN SCREEN INTERFACE ---
if st.session_state.auth_token is None:
    st.title("🔐 Fleet Analytics Portal")
    st.write("Please log in to access the Machinery Health Simulator.")
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Log In", type="primary")
        
        if submit_button:
            try:
                # OAuth2 forms expect standard form-data payloads, not JSON
                login_payload = {"username": username, "password": password}
                response = requests.post(f"{BACKEND_URL}/token", data=login_payload)
                
                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.auth_token = token_data["access_token"]
                    st.success("Login successful! Loading dashboard...")
                    st.rerun()  # Instantly re-render to display the actual app
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Could not connect to authentication services: {e}")
                
    st.stop() # Halts app rendering here if user isn't logged in!


# --- LOGGED IN ROUTING ---

# headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

# # Add a logout utility in the top right sidebar layout
# with st.sidebar:
#     st.write(f"Logged in as: **admin**")
#     if st.button("🚪 Log Out"):
#         st.session_state.auth_token = None
#         st.session_state.prediction_result = None
#         st.rerun()

# --- LOGGED IN ROUTING ---
headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

with st.sidebar:
    st.write(f"Logged in as: **admin**")
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.auth_token = None
        st.session_state.prediction_result = None
        st.rerun()
        
    st.markdown("---")
    
    # 🔐 Admin-Only Console Section inside the authenticated workspace
    with st.expander("👤 Team Access Management"):
        st.write("Provision new team accounts:")
        with st.form("sidebar_signup_form", clear_on_submit=True):
            new_user = st.text_input("Username", key="new_user_side")
            new_pass = st.text_input("Password", type="password", key="new_pass_side")
            create_btn = st.form_submit_button("Authorize User", type="secondary", use_container_width=True)
            
            if create_btn:
                if not new_user or not new_pass:
                    st.warning("All entry fields required.")
                elif len(new_pass) < 6:
                    st.warning("Password must be 6+ characters.")
                else:
                    try:
                        # 💡 CRITICAL FIX: We now pass `headers=headers` containing our admin token!
                        signup_payload = {"username": new_user, "password": new_pass}
                        reg_response = requests.post(
                            f"{BACKEND_URL}/register", 
                            json=signup_payload, 
                            headers=headers
                        )
                        
                        if reg_response.status_code == 201:
                            st.success(f"Access granted for '{new_user}'!")
                        else:
                            err_detail = reg_response.json().get("detail", "Request unauthorized.")
                            st.error(f"Error: {err_detail}")
                    except Exception as e:
                        st.error(f"Network pipeline failure: {e}")

tab1, tab2 = st.tabs(["Prediction", "Validation"])

with tab1:
    st.title("🚜 Machinery Health Simulator")
    st.write("Adjust mechanical telemetry parameters derived from the Kaggle Predictive Maintenance dataset framework.")
    
    # Fetch global fleet metrics from our new endpoint
    try:
        summary_res = requests.get(f"{BACKEND_URL}/analytics/summary", headers=headers)
        if summary_res.status_code == 200:
            metrics = summary_res.json()
            
            # Render a clean 3-column structural KPI row
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Scans Logged", metrics["total_scans"])
            kpi2.metric("Critical Risks Caught", metrics["total_failures_detected"], delta=f"{metrics['total_failures_detected']} critical", delta_color="inverse")
            kpi3.metric("Fleet Avg Failure Risk", f"{metrics['average_failure_probability']}%")
            st.markdown("---")
    except Exception:
        pass  # Gracefully fall back if analytics services are temporarily loading
    
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
                    # 🛠️ FIXED: Added the authentication headers parameter here!
                    response = requests.post(f"{BACKEND_URL}/predict", json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        st.session_state.prediction_result = response.json()
                        st.session_state.api_error = None
                    elif response.status_code == 401:
                        st.session_state.auth_token = None  # Clear invalid session
                        st.error("Session expired or token invalid. Please log in again.")
                        st.rerun()
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
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
            response = requests.get(f"{BACKEND_URL}/history?limit=10", headers=headers)
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
            elif response.status_code == 401:
                st.session_state.auth_token = None
                st.error("Unauthorized access session details. Resetting login screen...")
                st.rerun()
            else:
                st.error("Could not fetch log history from the server backend.")
        except Exception as e:
            st.error(f"Network error connecting to database history pipeline: {e}")
    
    # --- INTERACTIVE BREAKDOWN COMPONENT ---
    st.write("---")
    st.header("🗺️ Operational Stress Envelope Explorer")
    st.write(
        "This heat matrix simulates alternative combinations of **Torque** and **Rotational Speed (RPM)** "
        "under your current fixed temperature and tool wear thresholds to identify critical operation boundaries."
    )
    
    # Guard against executing grid logic if user isn't fully authenticated
    if st.session_state.auth_token:
        with st.spinner("Simulating full engineering envelope..."):
            try:
                # Query the sweep API using the active live slider settings from the left column
                params = {
                    "air_temp": air_temp,
                    "proc_temp": proc_temp,
                    "tool_wear": tool_wear
                }
                grid_res = requests.get(
                    f"{BACKEND_URL}/analytics/stress-grid", 
                    params=params, 
                    headers=headers
                )
                
                if grid_res.status_code == 200:
                    grid_data = grid_res.json()
                    df_grid = pd.DataFrame(grid_data)
                    
                    # Reshape raw records into a clean 2D layout matrix
                    df_pivot = df_grid.pivot(
                        index="Torque (Nm)", 
                        columns="RPM", 
                        values="Failure Probability (%)"
                    )
                    
                    # Render using Plotly Express Heatmap structures
                    fig_heatmap = px.imshow(
                        df_pivot,
                        labels=dict(x="Rotational Speed (RPM)", y="Torque (Nm)", color="Risk %"),
                        x=df_pivot.columns,
                        y=df_pivot.index,
                        color_continuous_scale="RdYlGn_r", # Red high risk, Green safe
                        origin="lower",
                        aspect="auto"
                    )
                    
                    # Fine-tune clarity presentation elements
                    fig_heatmap.update_layout(
                        margin=dict(l=40, r=40, t=20, b=40),
                        height=400,
                        coloraxis_colorbar=dict(title="Calculated Risk %")
                    )
                    
                    # Display to user interface canvas
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    st.caption(
                        "💡 **How to interpret:** Deep red blocks signal strict parameter intersections where structural breakdown is highly likely. "
                        "Adjust your temperature or tool wear sliders on the left to watch the danger envelope shift in real-time."
                    )
                    
                else:
                    st.warning("Could not calculate stress matrix trends from the engine infrastructure.")
            except Exception as e:
                st.error(f"Failed to securely render interactive matrix pipeline: {e}")
    
    # --- BATCH FILE UPLOAD COMPONENT ---
    st.write("---")
    st.header("📊 Bulk Operations Center")
    st.write("Upload a historical engineering sensor spreadsheet to calculate shift risks in mass.")
    
    if st.session_state.auth_token:
        # 📦 Provide a quick template reference so users match schema columns perfectly
        with st.popover("📋 View Required CSV Template Schema"):
            st.code("air_temperature,process_temperature,rotational_speed,torque,tool_wear\n298.1,308.6,1500,40.5,0\n302.4,311.2,2100,55.2,18")
        
        uploaded_file = st.file_uploader(
            "Choose a raw machinery log file", 
            type=["csv"], 
            help="Make sure columns exactly match the template keys above."
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Batch Ingest", type="primary"):
                with st.spinner("Executing high-speed inference sweep across shift data..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        
                        batch_res = requests.post(
                            f"{BACKEND_URL}/predict/batch", 
                            files=files, 
                            headers=headers
                        )
                        
                        if batch_res.status_code == 200:
                            st.success("Batch diagnostics completed and saved safely to local database records!")
                            csv_text = batch_res.text
                            
                            st.download_button(
                                label="📥 Download Annotated Risk Report",
                                data=csv_text,
                                file_name="processed_fleet_risk_report.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            df_processed = pd.read_csv(io.StringIO(csv_text))
                            st.dataframe(df_processed.head(10), use_container_width=True)
                            st.caption(f"Showing sample preview of {len(df_processed)} evaluated telemetry timestamps.")
                            
                        else:
                            try:
                                err_msg = batch_res.json().get("detail", "Error evaluating batch payload.")
                            except Exception:
                                err_msg = f"Server returned status code {batch_res.status_code}. Check backend console logs."
                            st.error(f"Processing Rejected: {err_msg}")
                            
                    except Exception as e:
                        st.error(f"Network transport error processing batch file: {e}")

with tab2:
    st.title("🛠️ Machine Maintenance Audit Panel")

    # Fetch recent unverified predictions from backend
    response = requests.get(f"{BACKEND_URL}/unreviewed-predictions", headers = headers)
    
    # if response["status"] == 1:
    logs = response.json()  # List of unreviewed prediction dicts
    
    for item in logs:
        with st.expander(f"Prediction ID: #{item['id']} - Sensor Batch"):
            # st.write(f"**Predicted Failure:** {item['failure_probability_percent']}")
            # st.write(f"**Torque:** {item['torque']} | **Air Temp:** {}")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Air Temperature (K)", item['air_temperature'])
            col2.metric("Process Temperature (K)", item["process_temperature"])
            col3.metric("Rotational Speed (RPM)", item["rotational_speed"])
            col4.metric("Torque (Nm)", item["torque"])
            col5.metric("Tool Wear (min)", item["tool_wear"])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm: Machine Operated Normally", key=f"norm_{item['id']}"):
                    requests.post(f"{BACKEND_URL}/feedback", json={"prediction_id": item['id'], "actual_failure": 0}, headers = headers)
                    st.success("Logged as Normal!")
                    st.rerun()
            with col2:
                if st.button("Confirm: Machine Failed / Needed Repair", key=f"fail_{item['id']}"):
                    requests.post(f"{BACKEND_URL}/feedback", json={"prediction_id": item['id'], "actual_failure": 1}, headers = headers)
                    st.warning("Logged as Failure!")
                    st.rerun()