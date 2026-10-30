
import io
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# =========================================================
# VehicleGPT - AI-powered vehicle health demo
# Simulated readings only; not a real vehicle diagnosis.
# =========================================================

st.set_page_config(
    page_title="VehicleGPT",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111c2e;
}
.metric-card {
    background: linear-gradient(145deg, #172338, #101a2b);
    border: 1px solid #30445f;
    border-radius: 16px;
    padding: 18px;
    min-height: 125px;
}
.metric-label {
    color: #a8bbd3;
    font-size: 13px;
    font-weight: 600;
}
.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}
.metric-note {
    color: #7dd3fc;
    font-size: 12px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "vehicle_anomaly_model.pkl"
DB_PATH = BASE_DIR / "vehicle_history.db"


# ----------------------- Database -------------------------

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                engine_temp REAL NOT NULL,
                battery_voltage REAL NOT NULL,
                speed REAL NOT NULL,
                rule_status TEXT NOT NULL,
                ml_result TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                engine_temp REAL NOT NULL,
                battery_voltage REAL NOT NULL,
                speed REAL NOT NULL,
                risk_level TEXT NOT NULL,
                maintenance_priority TEXT NOT NULL,
                alerts TEXT NOT NULL
            )
        """)

        conn.commit()


def save_reading(
    timestamp,
    temp,
    voltage,
    speed,
    rule_status,
    ml_result="Not run",
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sensor_history
            (
                timestamp,
                engine_temp,
                battery_voltage,
                speed,
                rule_status,
                ml_result
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(timestamp),
            float(temp),
            float(voltage),
            float(speed),
            rule_status,
            ml_result,
        ))
        conn.commit()




def load_history(limit=500):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT * FROM sensor_history
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def save_maintenance_record(
    timestamp,
    temp,
    voltage,
    speed,
    risk_level,
    maintenance_priority,
    alerts,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO maintenance_history (
                timestamp,
                engine_temp,
                battery_voltage,
                speed,
                risk_level,
                maintenance_priority,
                alerts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(timestamp),
            float(temp),
            float(voltage),
            float(speed),
            risk_level,
            maintenance_priority,
            "\n".join(alerts),
        ))
        conn.commit()

def create_maintenance_pdf(
    temp,
    voltage,
    speed,
    risk_level,
    maintenance_priority,
    alerts,
    recommendations,
):
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "VehicleGPT - Predictive Maintenance Report",
        styles["Title"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        styles["Normal"],
    ))
    story.append(Spacer(1, 16))

    data = [
        ["Parameter", "Recorded Value"],
        ["Engine Temperature", f"{temp:.1f} °C"],
        ["Battery Voltage", f"{voltage:.2f} V"],
        ["Vehicle Speed", f"{speed:.1f} km/h"],
        ["Risk Level", risk_level],
        ["Maintenance Priority", maintenance_priority],
    ]

    table = Table(data, colWidths=[190, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Maintenance Alerts", styles["Heading2"]))

    for alert in alerts:
        story.append(Paragraph(
            "• " + alert,
            styles["Normal"],
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommended Actions", styles["Heading2"]))

    for recommendation in recommendations:
        story.append(Paragraph(
            "• " + recommendation,
            styles["Normal"],
        ))

    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "Educational prototype only. Readings are simulated and "
        "this report is not a real vehicle diagnosis.",
        styles["Italic"],
    ))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# -------------------- Health Functions --------------------

def rule_status(temp, voltage):
    if temp >= 110 or voltage < 11.5:
        return "CRITICAL"

    if temp >= 100 or voltage < 12.2:
        return "WARNING"

    return "NORMAL"


def health_score(temp, voltage):
    temp_penalty = (
        50 if temp >= 110
        else 25 if temp >= 100
        else 0
    )

    voltage_penalty = (
        50 if voltage < 11.5
        else 25 if voltage < 12.2
        else 0
    )

    return max(0, 100 - temp_penalty - voltage_penalty)


def run_ml(temp, voltage, speed):
    if not MODEL_PATH.exists():
        return "Model file not found"

    try:
        model = joblib.load(MODEL_PATH)

        sample = pd.DataFrame([{
            "engine_temp": float(temp),
            "battery_voltage": float(voltage),
            "speed": float(speed),
        }])

        prediction = model.predict(sample)[0]

        if prediction == -1:
            return "Unusual pattern"

        return "No unusual pattern flagged"

    except Exception as exc:
        return f"Prediction error: {exc}"


def status_explanation(temp, voltage):
    issues = []

    if temp >= 110:
        issues.append(
            "Engine temperature is in the critical demo range. "
            "If this is a real vehicle, stop safely and seek professional help."
        )
    elif temp >= 100:
        issues.append(
            "Engine temperature is elevated according to this demo threshold. "
            "Verify the reading and vehicle manual."
        )

    if voltage < 11.5:
        issues.append(
            "Battery voltage is in the critical demo range. "
            "Have the battery and charging system checked."
        )
    elif voltage < 12.2:
        issues.append(
            "Battery voltage is low according to this demo threshold. "
            "Recheck under suitable conditions."
        )

    if not issues:
        issues.append(
            "No configured demo threshold alert was triggered. "
            "This does not certify that the vehicle is fault-free."
        )

    return issues


init_db()


def predictive_maintenance(engine_temp, battery_voltage, speed):
    """Vehicle sensor readings ke basis par maintenance advice."""

    alerts = []
    recommendations = []

    # Engine temperature analysis
    if engine_temp >= 110:
        alerts.append("CRITICAL: Engine overheating")
        recommendations.append(
            "Vehicle ko safe jagah rok kar engine thanda hone dein "
            "aur cooling system check karwayein."
        )
    elif engine_temp >= 100:
        alerts.append("WARNING: Engine temperature high")
        recommendations.append(
            "Coolant level, radiator aur cooling fan check karwayein."
        )

    # Battery voltage analysis
    if battery_voltage < 11.5:
        alerts.append("CRITICAL: Battery voltage very low")
        recommendations.append(
            "Battery aur charging system ki jaldi jaanch karwayein."
        )
    elif battery_voltage < 12.2:
        alerts.append("WARNING: Battery voltage low")
        recommendations.append(
            "Battery terminals aur charging system check karwayein."
        )

    # Speed analysis
    if speed < 0:
        alerts.append("INVALID: Speed cannot be negative")
        recommendations.append(
            "Speed sensor ki reading verify karein."
        )

    if not alerts:
        alerts.append("NORMAL: No configured maintenance warning")
        recommendations.append(
            "Readings configured limits ke andar hain. "
            "Regular vehicle servicing jaari rakhein."
        )

   
    if any(alert.startswith("CRITICAL") for alert in alerts):
        risk_level = "CRITICAL"
        maintenance_priority = "Immediate inspection recommended"

    elif any(alert.startswith("WARNING") for alert in alerts):
        risk_level = "HIGH"
        maintenance_priority = "Service inspection recommended soon"

    else:
        risk_level = "NORMAL"
        maintenance_priority = "Continue routine maintenance"

    return {
        "alerts": alerts,
        "recommendations": recommendations,
        "risk_level": risk_level,
        "maintenance_priority": maintenance_priority,
    }


# ---------------------- AI Vehicle Assistant ----------------------

def ask_vehicle_ai(pm_result, temp, voltage, speed):
    prompt = f"""
    Tum VehicleGPT ke AI Vehicle Assistant ho.
    User ko saral Hindi mein jawab do.

    Vehicle readings:
    Engine temperature: {temp} °C
    Battery voltage: {voltage} V
    Vehicle speed: {speed} km/h

    Predictive Maintenance risk:
    {pm_result["risk_level"]}

    Maintenance priority:
    {pm_result["maintenance_priority"]}

    Alerts:
    {pm_result["alerts"]}

    Recommended actions:
    {pm_result["recommendations"]}

    Risk ka matlab samjhao.
    User ko batao ki ab kya karna chahiye.
    Zarurat hone par mechanic se jaanch karwane ki salah do.
    Bina evidence ke kisi fault ki pushti mat karo.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json().get(
            "response",
            "AI se jawab nahi mila."
        )

    except requests.RequestException as e:
        return f"AI connection error: {e}"

# ---------------------- Session Data ----------------------

if "vehicle_data" not in st.session_state:
    now = datetime.now()

    st.session_state.vehicle_data = pd.DataFrame({
        "Time": pd.date_range(
            end=now,
            periods=10,
            freq="min",
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
        ],
    })

    # Save initial sample readings once per new session.
    for _, row in st.session_state.vehicle_data.iterrows():
        save_reading(
            row["Time"],
            row["Engine Temperature (°C)"],
            row["Battery Voltage (V)"],
            row["Speed (km/h)"],
            rule_status(
                row["Engine Temperature (°C)"],
                row["Battery Voltage (V)"],
            ),
        )


data = st.session_state.vehicle_data
latest = data.iloc[-1]

latest_temp = float(latest["Engine Temperature (°C)"])
latest_voltage = float(latest["Battery Voltage (V)"])
latest_speed = float(latest["Speed (km/h)"])

latest_status = rule_status(latest_temp, latest_voltage)
score = health_score(latest_temp, latest_voltage)


# ------------------------- Sidebar ------------------------

with st.sidebar:
    st.markdown("## 🚗 VehicleGPT")
    st.caption("AI-Powered Vehicle Intelligence")
    st.caption("Laptop-based educational prototype")
    st.divider()

    page = st.radio(
        "Choose section",
        [
            "🏠 Dashboard",
            "📊 Sensor Readings",
            "🤖 Rule-Based Alerts",
            "🧠 Trained ML Model",
            "🔍 Analyze Readings",
            "🤖 AI Vehicle Assistant",
            "🩺 Vehicle Health Assessment",
            "🗂️ History & Alerts",
            "📄 Health Report",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Simulated data only — no real vehicle connected.")


# ------------------------ Dashboard -----------------------

st.subheader("Predictive Maintenance")

pm_temp = st.number_input(
    "Engine Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=85.0,
    step=1.0,
    key="pm_engine_temp",
)

pm_voltage = st.number_input(
    "Battery Voltage (V)",
    min_value=0.0,
    max_value=24.0,
    value=12.6,
    step=0.1,
    key="pm_battery_voltage",
)

pm_speed = st.number_input(
    "Vehicle Speed (km/h)",
    min_value=0.0,
    max_value=300.0,
    value=40.0,
    step=1.0,
    key="pm_speed",
)



if st.button("Analyze Maintenance Status", key="pm_analyze"):
    pm_result = predictive_maintenance(
        pm_temp,
        pm_voltage,
        pm_speed,
    )

    st.session_state.pm_result = pm_result
    st.session_state.saved_pm_temp = pm_temp
    st.session_state.saved_pm_voltage = pm_voltage
    st.session_state.saved_pm_speed = pm_speed

    save_maintenance_record(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        temp=pm_temp,
        voltage=pm_voltage,
        speed=pm_speed,
        risk_level=pm_result["risk_level"],
        maintenance_priority=pm_result["maintenance_priority"],
        alerts=pm_result["alerts"],
    )


if "pm_result" in st.session_state:
    pm_result = st.session_state.pm_result

    st.markdown("### Maintenance Risk Assessment")

    if pm_result["risk_level"] == "CRITICAL":
        st.error("Risk Level: CRITICAL")
    elif pm_result["risk_level"] == "HIGH":
        st.warning("Risk Level: HIGH")
    else:
        st.success("Risk Level: NORMAL")

    st.info(
        "Maintenance Priority: "
        + pm_result["maintenance_priority"]
    )

    st.markdown("### Maintenance Alerts")
    for alert in pm_result["alerts"]:
        st.write("- " + alert)

    st.markdown("### Recommended Actions")
    for recommendation in pm_result["recommendations"]:
        st.write("- " + recommendation)

    pdf_bytes = create_maintenance_pdf(
        temp=st.session_state.saved_pm_temp,
        voltage=st.session_state.saved_pm_voltage,
        speed=st.session_state.saved_pm_speed,
        risk_level=pm_result["risk_level"],
        maintenance_priority=pm_result["maintenance_priority"],
        alerts=pm_result["alerts"],
        recommendations=pm_result["recommendations"],
    )

    st.download_button(
        label="📄 Download Maintenance PDF Report",
        data=pdf_bytes,
        file_name="VehicleGPT_Maintenance_Report.pdf",
        mime="application/pdf",
        key="download_maintenance_pdf",
    )
    st.markdown("### 🤖 AI Vehicle Assistant")

    if st.button("Explain Results with AI", key="ask_vehicle_ai"):
        with st.spinner("AI Hindi mein explanation taiyar kar raha hai..."):

            ai_answer = ask_vehicle_ai(
                pm_result=pm_result,
                temp=st.session_state.saved_pm_temp,
                voltage=st.session_state.saved_pm_voltage,
                speed=st.session_state.saved_pm_speed
            )

        st.session_state.pm_ai_answer = ai_answer

    if "pm_ai_answer" in st.session_state:
        st.markdown("#### AI Explanation")
        st.write(st.session_state.pm_ai_answer)

    st.markdown("### Maintenance Risk Assessment")

    if pm_result["risk_level"] == "CRITICAL":
        st.error("Risk Level: CRITICAL")
    elif pm_result["risk_level"] == "HIGH":
        st.warning("Risk Level: HIGH")
    else:
        st.success("Risk Level: NORMAL")

    st.info(
        "Maintenance Priority: "
        + pm_result["maintenance_priority"]
    )

    st.markdown("### Maintenance Alerts")

    for alert in pm_result["alerts"]:
        if alert.startswith("CRITICAL"):
            st.error(alert)
        elif alert.startswith("WARNING"):
            st.warning(alert)
        elif alert.startswith("INVALID"):
            st.info(alert)
        else:
            st.success(alert)

    st.markdown("### Recommended Actions")

    for recommendation in pm_result["recommendations"]:
        st.write("- " + recommendation)

if page == "🏠 Dashboard":
    st.title("🚗 VehicleGPT")
    st.subheader(
        "AI-Powered Vehicle Health & Predictive Maintenance"
    )

    st.info(
        "Demo mode: readings are simulated sample data, "
        "not live data from a real vehicle."
    )

    c1, c2, c3 = st.columns(3)

    metrics = [
        (
            c1,
            "🌡️ ENGINE TEMPERATURE",
            f"{latest_temp:.1f} °C",
            "Latest simulated reading",
        ),
        (
            c2,
            "🔋 BATTERY VOLTAGE",
            f"{latest_voltage:.2f} V",
            "Latest simulated reading",
        ),
        (
            c3,
            "🚗 VEHICLE SPEED",
            f"{latest_speed:.1f} km/h",
            "Latest simulated reading",
        ),
    ]

    for col, label, value, note in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🎯 Vehicle Health Score")
        st.metric("Demo Health Score", f"{score}/100")
        st.progress(score / 100)

        st.caption(
            "Illustrative score based only on configured "
            "temperature and voltage rules."
        )

        if score >= 80:
            st.success("Good according to configured demo thresholds.")
        elif score >= 50:
            st.warning("Readings need review.")
        else:
            st.error("Critical demo readings detected.")

    with right:
        st.subheader("🩺 Health Status")
        st.metric("Rule-based status", latest_status)

        for issue in status_explanation(
            latest_temp,
            latest_voltage,
        ):
            if latest_status == "CRITICAL":
                st.error(issue)
            elif latest_status == "WARNING":
                st.warning(issue)
            else:
                st.write("• " + issue)

    # Temperature gauge
    gauge1, gauge2 = st.columns(2)

    with gauge1:
        temp_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest_temp,
                title={"text": "Engine Temperature (°C)"},
                gauge={
                    "axis": {"range": [0, 140]},
                    "steps": [
                        {
                            "range": [0, 95],
                            "color": "#1f6b4a",
                        },
                        {
                            "range": [95, 110],
                            "color": "#9a6b17",
                        },
                        {
                            "range": [110, 140],
                            "color": "#8b2d35",
                        },
                    ],
                },
            )
        )

        temp_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )

        st.plotly_chart(
            temp_gauge,
            use_container_width=True,
        )

    # Speed gauge
    with gauge2:
        speed_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest_speed,
                title={"text": "Vehicle Speed (km/h)"},
                gauge={
                    "axis": {"range": [0, 180]},
                    "steps": [
                        {
                            "range": [0, 120],
                            "color": "#1f6b4a",
                        },
                        {
                            "range": [120, 150],
                            "color": "#9a6b17",
                        },
                        {
                            "range": [150, 180],
                            "color": "#8b2d35",
                        },
                    ],
                },
            )
        )

        speed_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )

        st.plotly_chart(
            speed_gauge,
            use_container_width=True,
        )

    # Trend chart
    st.subheader("📈 Sensor Trends")

    fig = px.line(
        data,
        x="Time",
        y=[
            "Engine Temperature (°C)",
            "Battery Voltage (V)",
            "Speed (km/h)",
        ],
        markers=True,
        template="plotly_dark",
        title="Simulated Sensor Trends",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Simulated new reading
    st.subheader("⏱️ Simulated Live Reading")
    st.write(
        "Generate a new synthetic sensor reading "
        "to demonstrate live monitoring."
    )

    if st.button("Generate Next Sensor Reading", type="primary"):
        new_temp = round(
            max(
                65,
                min(
                    125,
                    latest_temp + random.uniform(-2.5, 3.5),
                ),
            ),
            1,
        )

        new_voltage = round(
            max(
                10.5,
                min(
                    13.0,
                    latest_voltage + random.uniform(-0.18, 0.12),
                ),
            ),
            2,
        )

        new_speed = round(
            max(
                0,
                min(
                    160,
                    latest_speed + random.uniform(-12, 12),
                ),
            ),
            1,
        )

        new_time = datetime.now()

        new_row = pd.DataFrame([{
            "Time": new_time,
            "Engine Temperature (°C)": new_temp,
            "Battery Voltage (V)": new_voltage,
            "Speed (km/h)": new_speed,
        }])

        st.session_state.vehicle_data = pd.concat(
            [data, new_row],
            ignore_index=True,
        ).tail(100)

        new_status = rule_status(new_temp, new_voltage)
        ml_result = run_ml(
            new_temp,
            new_voltage,
            new_speed,
        )

        save_reading(
            new_time,
            new_temp,
            new_voltage,
            new_speed,
            new_status,
            ml_result,
        )

        st.rerun()

    st.subheader("📋 Recent Readings")
    st.dataframe(
        data.tail(10),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------- Sensor Readings -------------------

elif page == "📊 Sensor Readings":
    st.title("📊 Sensor Readings")

    sensor = st.selectbox(
        "Select sensor",
        [
            "Engine Temperature (°C)",
            "Battery Voltage (V)",
            "Speed (km/h)",
        ],
    )

    fig = px.line(
        data,
        x="Time",
        y=sensor,
        markers=True,
        template="plotly_dark",
        title=sensor + " over time",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------- Rule-Based Alerts -----------------

elif page == "🤖 Rule-Based Alerts":
    st.title("🤖 Rule-Based Anomaly Alerts")

    st.caption(
        "These are configurable demo rules, "
        "not the trained ML model."
    )

    st.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    st.metric("Battery Voltage", f"{latest_voltage:.2f} V")

    for issue in status_explanation(
        latest_temp,
        latest_voltage,
    ):
        if latest_status == "CRITICAL":
            st.error(issue)
        elif latest_status == "WARNING":
            st.warning(issue)
        else:
            st.success(issue)

    st.caption(
        "Thresholds vary by vehicle and operating conditions; "
        "consult the owner's manual."
    )


# ---------------------- Trained ML Model ------------------

elif page == "🧠 Trained ML Model":
    st.title("🧠 Trained ML Model")

    st.write(
        "This uses the existing vehicle_anomaly_model.pkl file, "
        "if available."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")

    else:
        ml_result = run_ml(
            latest_temp,
            latest_voltage,
            latest_speed,
        )

        if ml_result == "Unusual pattern":
            st.warning(
                "The model flagged this pattern as unusual. "
                "This is not a confirmed diagnosis."
            )

        elif ml_result == "No unusual pattern flagged":
            st.success(
                "The model did not flag this pattern as unusual. "
                "This does not prove the vehicle is fault-free."
            )

        else:
            st.error(ml_result)

        st.write("Input values")

        st.dataframe(
            pd.DataFrame([{
                "engine_temp": latest_temp,
                "battery_voltage": latest_voltage,
                "speed": latest_speed,
            }]),
            use_container_width=True,
        )

        st.info(
            "Model quality metrics such as precision, recall, and F1 "
            "need a labeled test dataset."
        )


# ---------------------- Analyze Readings ------------------

elif page == "🔍 Analyze Readings":
    st.title("🔍 Analyze Custom Readings")

    with st.form("analyze_custom_readings"):
        input_temp = st.number_input(
            "Engine Temperature (°C)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
        )

        input_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=0.0,
            max_value=20.0,
            value=12.4,
        )

        input_speed = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=40.0,
        )

        submitted = st.form_submit_button("Analyze")

    if submitted:
        custom_status = rule_status(
            input_temp,
            input_voltage,
        )

        st.metric("Rule-based status", custom_status)

        for issue in status_explanation(
            input_temp,
            input_voltage,
        ):
            if custom_status == "CRITICAL":
                st.error(issue)
            elif custom_status == "WARNING":
                st.warning(issue)
            else:
                st.success(issue)

        ml_result = run_ml(
            input_temp,
            input_voltage,
            input_speed,
        )

        st.write("Trained model result:", ml_result)

        save_reading(
            datetime.now(),
            input_temp,
            input_voltage,
            input_speed,
            custom_status,
            ml_result,
        )

        st.success("Reading saved to local history.")


# ---------------------- AI Vehicle Assistant -------------

elif page == "🤖 AI Vehicle Assistant":
    st.title("🤖 VehicleGPT — AI Vehicle Assistant")

    st.caption(
        "General educational guidance. "
        "This is not a certified mechanic or diagnostic tool."
    )

    question = st.text_area(
        "Ask a vehicle question",
        placeholder="Example: What should I do if the engine overheats?",
    )

    if st.button("Ask VehicleGPT"):
        if not question.strip():
            st.warning("Please enter a question first.")

        else:
            q = question.lower()

            if "battery" in q or "बैटरी" in q:
                st.markdown(
                    "**General guidance:** If this is a real vehicle, "
                    "arrange a battery/charging-system check. "
                    "Avoid handling a swollen, leaking, or damaged battery."
                )

            elif (
                "overheat" in q
                or "temperature" in q
                or "गरम" in q
            ):
                st.markdown(
                    "**General guidance:** Stop in a safe place if "
                    "overheating is suspected. Switch the engine off "
                    "when safe, and never open a hot radiator cap. "
                    "Follow the vehicle manual and seek professional help."
                )

            else:
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2:1b",
                            "prompt": (
                                "You are a cautious vehicle maintenance "
                                "assistant. Answer in simple Hindi or "
                                "English matching the user. Give general "
                                "educational guidance, not a definitive "
                                "diagnosis. For safety concerns, advise "
                                "a qualified mechanic. Question: "
                                + question
                            ),
                            "stream": False,
                        },
                        timeout=90,
                    )

                    response.raise_for_status()

                    answer = response.json().get(
                        "response",
                        "No response received.",
                    )

                    st.markdown(answer)

                except requests.exceptions.RequestException:
                    st.error(
                        "Ollama connection failed. Ensure Ollama is "
                        "running and llama3.2:1b is available."
                    )


# ---------------------- Health Assessment ----------------

elif page == "🩺 Vehicle Health Assessment":
    st.title("🩺 Vehicle Health Assessment")

    st.caption(
        "Enter sample readings to evaluate the configured demo thresholds."
    )

    with st.form("health_assessment_form"):
        health_temp = st.number_input(
            "Engine Temperature (°C)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
        )

        health_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=0.0,
            max_value=20.0,
            value=12.4,
        )

        health_speed = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=40.0,
        )

        check_health = st.form_submit_button(
            "Check Vehicle Health"
        )

    if check_health:
        current_status = rule_status(
            health_temp,
            health_voltage,
        )

        st.metric("Rule-based status", current_status)

        st.metric(
            "Illustrative health score",
            f"{health_score(health_temp, health_voltage)}/100",
        )

        for issue in status_explanation(
            health_temp,
            health_voltage,
        ):
            if current_status == "CRITICAL":
                st.error(issue)
            elif current_status == "WARNING":
                st.warning(issue)
            else:
                st.success(issue)

        st.write(f"Speed: {health_speed:.1f} km/h")

        st.caption(
            "Educational prototype only; "
            "not a certified vehicle diagnosis."
        )


# ---------------------- History & Alerts ------------------

elif page == "🗂️ History & Alerts":
    st.title("🗂️ Sensor History & Alert Log")

    history = load_history()

    st.subheader("🛠️ Predictive Maintenance History")

    with sqlite3.connect(DB_PATH) as conn:
        maintenance_history = pd.read_sql_query(
            """
            SELECT * FROM maintenance_history
            ORDER BY id DESC
            LIMIT 100
            """,
            conn,
        )

    if maintenance_history.empty:
        st.info(
            "No maintenance records yet. "
            "Run Predictive Maintenance analysis first."
        )
    else:
        st.metric(
            "Saved maintenance analyses",
            len(maintenance_history),
        )

        st.dataframe(
            maintenance_history,
            use_container_width=True,
            hide_index=True,
        )

        maintenance_csv = maintenance_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Maintenance History CSV",
            maintenance_csv,
            file_name="VehicleGPT_Maintenance_History.csv",
            mime="text/csv",
        )

    if history.empty:
        st.info(
            "No saved readings yet. Generate a reading on Dashboard."
        )

    else:
        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="coerce",
        )

        st.metric("Saved readings", len(history))

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        chart = px.line(
            history.sort_values("timestamp"),
            x="timestamp",
            y=["engine_temp", "battery_voltage", "speed"],
            markers=True,
            template="plotly_dark",
            title="Saved Sensor History",
        )

        st.plotly_chart(chart, use_container_width=True)

        csv_bytes = history.to_csv(
            index=False,
        ).encode("utf-8")

        st.download_button(
            "Download History CSV",
            csv_bytes,
            file_name="VehicleGPT_History.csv",
            mime="text/csv",
        )

        if st.button("Clear Saved History"):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM sensor_history")
                conn.commit()

            st.warning("History cleared.")
            st.rerun()


# ---------------------- Health Report ---------------------

elif page == "📄 Health Report":
    st.title("📄 Vehicle Health Report")

    st.caption(
        "Generate a report from the latest simulated sensor readings."
    )

    report_temp = st.number_input(
        "Engine Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=latest_temp,
        key="report_temp",
    )

    report_voltage = st.number_input(
        "Battery Voltage (V)",
        min_value=0.0,
        max_value=20.0,
        value=latest_voltage,
        key="report_voltage",
    )

    report_speed = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=300.0,
        value=latest_speed,
        key="report_speed",
    )

    report_status = rule_status(
        report_temp,
        report_voltage,
    )

    report_score = health_score(
        report_temp,
        report_voltage,
    )

    st.metric("Demo Status", report_status)

    st.metric(
        "Illustrative Health Score",
        f"{report_score}/100",
    )

    # CSV report
    report_csv = pd.DataFrame([{
        "Generated At": datetime.now().isoformat(timespec="seconds"),
        "Engine Temperature (°C)": report_temp,
        "Battery Voltage (V)": report_voltage,
        "Vehicle Speed (km/h)": report_speed,
        "Rule-based Status": report_status,
        "Illustrative Health Score": report_score,
        "Report Type": (
            "Simulated educational demo; not a certified diagnosis"
        ),
    }]).to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Health Report (CSV)",
        data=report_csv,
        file_name="VehicleGPT_Health_Report.csv",
        mime="text/csv",
    )

    # PDF report
    if st.button("Prepare PDF Report", type="primary"):
        pdf_buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "VehicleGPTReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#123B63"),
            alignment=TA_CENTER,
            fontSize=22,
            spaceAfter=12,
        )

        story = [
            Paragraph("VehicleGPT", title_style),
            Paragraph(
                "Vehicle Health Assessment Report",
                styles["Heading2"],
            ),
            Paragraph(
                "Generated: "
                + datetime.now().strftime("%d-%m-%Y %H:%M"),
                styles["Normal"],
            ),
            Spacer(1, 18),
            Paragraph("1. Health Summary", styles["Heading2"]),
        ]

        summary = [
            ["Parameter", "Reading / Status"],
            ["Illustrative Health Score", f"{report_score}/100"],
            ["Rule-based Status", report_status],
            ["Engine Temperature", f"{report_temp:.1f} °C"],
            ["Battery Voltage", f"{report_voltage:.2f} V"],
            ["Vehicle Speed", f"{report_speed:.1f} km/h"],
        ]

        summary_table = Table(
            summary,
            colWidths=[220, 220],
            repeatRows=1,
        )

        summary_table.setStyle(TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123B63"),
            ),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#F1F5F9"),
            ),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        story.extend([
            summary_table,
            Spacer(1, 18),
            Paragraph(
                "2. Recent Sensor Readings",
                styles["Heading2"],
            ),
        ])

        recent = data.tail(10)

        table_data = [[
            "Time",
            "Temperature (°C)",
            "Battery (V)",
            "Speed (km/h)",
        ]]

        for _, row in recent.iterrows():
            table_data.append([
                str(row["Time"]),
                f'{float(row["Engine Temperature (°C)"]):.1f}',
                f'{float(row["Battery Voltage (V)"]):.2f}',
                f'{float(row["Speed (km/h)"]):.1f}',
            ])

        readings_table = Table(
            table_data,
            colWidths=[125, 125, 90, 90],
            repeatRows=1,
        )

        readings_table.setStyle(TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123B63"),
            ),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([
            readings_table,
            Spacer(1, 18),
            Paragraph("Important Note", styles["Heading2"]),
            Paragraph(
                "This report is based on simulated/demo sensor data "
                "and configured thresholds. It is not a real vehicle "
                "inspection, safety certification, or confirmed diagnosis.",
                styles["BodyText"],
            ),
        ])

        doc.build(story)

        st.download_button(
            "⬇️ Download VehicleGPT Health Report (PDF)",
            data=pdf_buffer.getvalue(),
            file_name="VehicleGPT_Health_Report.pdf",
            mime="application/pdf",
        )


# ------------------------- Footer -------------------------
st.divider()

st.caption(
    "VehicleGPT | AI-powered vehicle intelligence prototype "
    "| Simulated data only"
)
