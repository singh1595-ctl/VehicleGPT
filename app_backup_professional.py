
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #080f1d 0%, #101b30 55%, #0b1220 100%);
    color: #e8eef8;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111c2e 0%, #0b1220 100%);
    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] h2 {
    color: #7dd3fc;
    font-weight: 800;
}

/* Headings */
h1 {
    color: #f8fafc !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

h2, h3 {
    color: #dbeafe !important;
    font-weight: 700 !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #192942, #101a2b);
    border: 1px solid #304968;
    border-radius: 18px;
    padding: 23px 20px;
    min-height: 150px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #38bdf8;
}

.metric-label {
    color: #a8bbd3;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.7px;
    margin-bottom: 14px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.3;
}

.metric-note {
    color: #7dd3fc;
    font-size: 12px;
    margin-top: 12px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #7dd3fc;
    color: white;
    filter: brightness(1.12);
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox [data-baseweb="select"] {
    background-color: #111c2e;
    border-radius: 9px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #263750;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #293b54;
}

/* Captions */
.stCaption {
    color: #9aacc4;
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem;
    }

    .metric-card {
        padding: 16px 12px;
        min-height: 125px;
    }

    .metric-value {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🚗 VehicleGPT")
    st.caption("AI-Powered Vehicle Intelligence")
    st.divider()
    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Choose section",
        [
            "🏠 Dashboard",
            "📊 Sensor Readings",
            "🤖 ML Anomaly Detection",
            "🧠 Trained ML Model",
            "🔍 Analyze Readings",
            "🤖 AI Vehicle Assistant",
            "🩺 Vehicle Health Assessment",
            "📄 Health Report"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Educational prototype | Demo data")

# ==========================================
# SAMPLE VEHICLE DATA
# ==========================================

data = pd.DataFrame({
    "Time": pd.date_range(
        end=datetime.now(),
        periods=10,
        freq="min"
    ),
    "Engine Temperature (°C)": [
        88, 89, 90, 91, 92, 91, 93, 92, 93, 94
    ],
    "Battery Voltage (V)": [
        12.6, 12.5, 12.5, 12.4, 12.4,
        12.3, 12.4, 12.3, 12.3, 12.3
    ],
    "Speed (km/h)": [
        20, 25, 30, 35, 40, 38, 42, 36, 32, 35
    ]
})

latest = data.iloc[-1]
latest_temp = float(latest["Engine Temperature (°C)"])
latest_voltage = float(latest["Battery Voltage (V)"])
latest_speed = float(latest["Speed (km/h)"])

# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":
    st.title("🚗 VehicleGPT")
    st.subheader("AI-Powered Vehicle Intelligence System")
    st.write(
        "Vehicle sensor readings monitor karne aur possible problems "
        "identify karne ka smart dashboard."
    )

    st.info(
        "Demo mode: neeche dikhaya gaya data sample hai, "
        "real vehicle data nahi."
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ ENGINE TEMPERATURE</div>
            <div class="metric-value">{latest_temp:.1f} °C</div>
            <div class="metric-note">Latest sample reading</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔋 BATTERY VOLTAGE</div>
            <div class="metric-value">{latest_voltage:.1f} V</div>
            <div class="metric-note">Latest sample reading</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚗 VEHICLE SPEED</div>
            <div class="metric-value">{latest_speed:.1f} km/h</div>
            <div class="metric-note">Latest sample reading</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Sensor Overview")

    fig = px.line(
        data,
        x="Time",
        y=[
            "Engine Temperature (°C)",
            "Battery Voltage (V)",
            "Speed (km/h)"
        ],
        markers=True,
        template="plotly_dark",
        title="Sample Vehicle Sensor Trends"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Latest Sample Data")
    st.dataframe(data.tail(5), use_container_width=True)

# ==========================================
# SENSOR READINGS
# ==========================================

elif page == "📊 Sensor Readings":
    st.title("📊 Vehicle Sensor Readings")
    st.caption("Graph aur table mein sample readings dekhein.")

    sensor = st.selectbox(
        "Graph ke liye sensor select karein",
        [
            "Engine Temperature (°C)",
            "Battery Voltage (V)",
            "Speed (km/h)"
        ],
        key="sensor_graph_select"
    )

    fig = px.line(
        data,
        x="Time",
        y=sensor,
        markers=True,
        title=sensor + " Over Time",
        template="plotly_dark"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Sample Vehicle Data")
    st.dataframe(data, use_container_width=True)

# ==========================================
# RULE-BASED ANOMALY DETECTION
# ==========================================

elif page == "🤖 ML Anomaly Detection":
    st.title("🤖 Vehicle Anomaly Detection")
    st.warning(
        "Yeh section simple threshold rules use karta hai. "
        "Ise trained ML prediction na samjhein."
    )

    st.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    if latest_temp >= 110:
        st.error("Critical demo alert: engine temperature bahut high hai.")
    elif latest_temp >= 100:
        st.warning("Temperature warning: reading ko verify karein.")
    else:
        st.success("Temperature configured demo threshold se neeche hai.")

    st.metric("Battery Voltage", f"{latest_voltage:.1f} V")
    if latest_voltage < 11.5:
        st.error("Critical demo alert: battery voltage bahut low hai.")
    elif latest_voltage < 12.2:
        st.warning("Battery voltage warning: dobara check karein.")
    else:
        st.success("Battery voltage configured demo range mein hai.")

    st.caption(
        "Actual alerts vehicle type, operating conditions aur manufacturer "
        "specifications par depend karte hain."
    )

# ==========================================
# TRAINED ML MODEL
# ==========================================

elif page == "🧠 Trained ML Model":
    st.title("🧠 Trained ML Model Result")
    model_path = Path("vehicle_anomaly_model.pkl")

    if not model_path.exists():
        st.error(
            "vehicle_anomaly_model.pkl file nahi mili. "
            "Project folder aur model file ka naam check karein."
        )
    else:
        try:
            model = joblib.load(model_path)

            current_reading = pd.DataFrame([{
                "engine_temp": latest_temp,
                "battery_voltage": latest_voltage,
                "speed": latest_speed
            }])

            result = model.predict(current_reading)[0]

            if result == -1:
                st.warning(
                    "Model ne is sensor pattern ko unusual flag kiya hai. "
                    "Yeh confirmed vehicle fault diagnosis nahi hai."
                )
            else:
                st.success(
                    "Model ne is sensor pattern ko unusual flag nahi kiya. "
                    "Iska matlab vehicle fault-free hona confirm nahi hota."
                )

            st.subheader("Model Input")
            st.dataframe(current_reading, use_container_width=True)

        except Exception as error:
            st.error("Model prediction nahi ho paayi.")
            st.code(str(error))

# ==========================================
# ANALYZE CUSTOM READINGS
# ==========================================

elif page == "🔍 Analyze Readings":
    st.title("🔍 Analyze Vehicle Readings")
    st.write("Demo ke liye apni sensor readings enter karein.")

    with st.form("vehicle_reading_form"):
        input_temp = st.number_input(
            "Engine Temperature (°C)",
            min_value=0.0,
            max_value=200.0,
            value=90.0
        )
        input_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=0.0,
            max_value=20.0,
            value=12.4
        )
        input_speed = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=40.0
        )
        analyze = st.form_submit_button("Analyze Readings")

    if analyze:
        st.subheader("Analysis Results")

        if input_temp >= 110:
            st.error("Temperature: Critical demo alert")
        elif input_temp >= 100:
            st.warning("Temperature: Warning")
        else:
            st.success("Temperature: Demo threshold ke neeche")

        if input_voltage < 11.5:
            st.error("Battery: Critical demo alert")
        elif input_voltage < 12.2:
            st.warning("Battery: Low voltage warning")
        else:
            st.success("Battery: Demo threshold ke andar")

        st.write(f"Entered speed: {input_speed:.1f} km/h")

        custom_data = pd.DataFrame([{
            "engine_temp": input_temp,
            "battery_voltage": input_voltage,
            "speed": input_speed
        }])

        if model_path := Path("vehicle_anomaly_model.pkl"):
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    result = model.predict(custom_data)[0]
                    if result == -1:
                        st.warning("Trained model ne pattern unusual flag kiya.")
                    else:
                        st.success("Trained model ne pattern unusual flag nahi kiya.")
                except Exception as error:
                    st.error("Model prediction error.")
                    st.code(str(error))

        st.caption("Yeh educational demo hai, certified diagnosis nahi.")

# ==========================================
# LOCAL AI VEHICLE ASSISTANT
# ==========================================

elif page == "🤖 AI Vehicle Assistant":
    st.title("🤖 VehicleGPT - AI Vehicle Assistant")
    st.write(
        "Battery, engine, brakes, tyres aur maintenance ke baare mein "
        "Hindi ya English mein sawaal poochhein."
    )
    st.info(
        "Assistant local Ollama server se connect karne ki koshish karega. "
        "Ollama installed aur running hona chahiye."
    )

    question = st.text_area(
        "Apna sawaal likhein",
        placeholder="Jaise: Car ki battery low ho to kya karein?"
    )

    if st.button("Ask VehicleGPT"):
        if not question.strip():
            st.warning("Pehle apna sawaal likhein.")
        else:
            q = question.lower()

            if "battery" in q or "बैटरी" in q:
                st.markdown(
                    "**General guidance:** Battery terminals ko safely "
                    "inspect karwayein aur suitable conditions mein voltage "
                    "test karwayein. Agar battery swollen, leaking ya damaged "
                    "ho, use handle na karein aur professional help lein."
                )
            elif "temperature" in q or "overheat" in q or "गरम" in q:
                st.markdown(
                    "**General guidance:** Temperature warning ya overheating "
                    "ho to safe jagah rukkar engine band karein. Garam engine "
                    "ka radiator cap na kholein. Vehicle manual aur mechanic "
                    "ki guidance follow karein."
                )
            else:
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2:1b",
                            "prompt": (
                                "You are a cautious vehicle maintenance "
                                "assistant. Reply in simple Hindi or English "
                                "matching the user's language. Give general "
                                "educational guidance, not a definitive "
                                "diagnosis. If unsafe, advise a mechanic. "
                                "Question: " + question
                            ),
                            "stream": False
                        },
                        timeout=90
                    )
                    response.raise_for_status()
                    answer = response.json().get(
                        "response",
                        "Koi response nahi mila."
                    )
                    st.markdown(answer)

                except requests.exceptions.RequestException:
                    st.error(
                        "Ollama se connection nahi ho paaya. Check karein "
                        "ki Ollama installed hai, running hai, aur "
                        "'llama3.2:1b' model available hai."
                    )

# ==========================================
# VEHICLE HEALTH ASSESSMENT
# ==========================================

elif page == "🩺 Vehicle Health Assessment":
    st.title("🩺 Vehicle Health Assessment")
    st.write(
        "Sensor readings enter karke configured demo thresholds ke "
        "mutabik status check karein."
    )

    with st.form("health_assessment_form"):
        health_temp = st.number_input(
            "Engine Temperature (°C)",
            min_value=0.0,
            max_value=200.0,
            value=90.0
        )
        health_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=0.0,
            max_value=20.0,
            value=12.4
        )
        health_speed = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=40.0
        )
        check_health = st.form_submit_button("Check Vehicle Health")

    if check_health:
        alerts = []

        if health_temp >= 110:
            alerts.append("Critical temperature alert")
        elif health_temp >= 100:
            alerts.append("Temperature warning")

        if health_voltage < 11.5:
            alerts.append("Critical battery voltage alert")
        elif health_voltage < 12.2:
            alerts.append("Battery voltage warning")

        if alerts:
            st.warning("Please review these demo alerts:")
            for alert in alerts:
                st.write(f"- {alert}")
        else:
            st.success("No configured demo threshold alerts detected.")

        st.write(f"Engine temperature: {health_temp:.1f} °C")
        st.write(f"Battery voltage: {health_voltage:.1f} V")
        st.write(f"Vehicle speed: {health_speed:.1f} km/h")

        st.caption(
            "Thresholds educational demo ke liye hain. "
            "Ye certified vehicle diagnosis nahi hai."
        )

# ==========================================
# DOWNLOAD HEALTH REPORT
# ==========================================

elif page == "📄 Health Report":
    st.title("📄 Download Vehicle Health Report")
    st.write("Apni vehicle readings ki CSV report download karein.")

    report_temp = st.number_input(
        "Report: Engine Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=latest_temp,
        key="report_temp"
    )
    report_voltage = st.number_input(
        "Report: Battery Voltage (V)",
        min_value=0.0,
        max_value=20.0,
        value=latest_voltage,
        key="report_voltage"
    )
    report_speed = st.number_input(
        "Report: Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=300.0,
        value=latest_speed,
        key="report_speed"
    )

    if report_temp >= 110:
        temp_status = "Critical"
    elif report_temp >= 100:
        temp_status = "Warning"
    else:
        temp_status = "Within demo threshold"

    if report_voltage < 11.5:
        voltage_status = "Critical"
    elif report_voltage < 12.2:
        voltage_status = "Warning"
    else:
        voltage_status = "Within demo threshold"

    report = pd.DataFrame([{
        "Generated At": datetime.now().isoformat(timespec="seconds"),
        "Engine Temperature (°C)": report_temp,
        "Temperature Demo Status": temp_status,
        "Battery Voltage (V)": report_voltage,
        "Battery Demo Status": voltage_status,
        "Vehicle Speed (km/h)": report_speed,
        "Report Type": "Educational demo - not a certified diagnosis"
    }])

    st.dataframe(report, use_container_width=True)

    csv_data = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Vehicle Health Report (CSV)",
        data=csv_data,
        file_name="vehicle_health_report.csv",
        mime="text/csv"
    )

    st.caption(
        "Report configured thresholds par based hai. "
        "Ye certified vehicle diagnosis nahi hai."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption(
    "VehicleGPT | AI-powered vehicle intelligence prototype | "
    "Sample data only"
)
