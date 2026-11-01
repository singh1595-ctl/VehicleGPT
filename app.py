
import io
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from predictive_maintenance import predict_maintenance
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import requests
import plotly.graph_objects as go
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
/* VehicleGPT - clean dark professional theme */
:root {
    color-scheme: dark;
    --vg-primary: #38bdf8;
    --vg-primary-dark: #0284c7;
    --vg-ink: #f1f5f9;
    --vg-muted: #94a3b8;
    --vg-border: #334155;
    --vg-surface: #172033;
    --vg-page: #0b1220;
}

.stApp {
    background: var(--vg-page) !important;
    color: #f1f5f9 !important;
}

[data-testid="stHeader"] {
    background: #0b1220 !important;
}

[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}

.stMarkdown, .stText, p, label, span {
    color: inherit;
}

h1, h2, h3, h4 {
    color: var(--vg-ink) !important;
    font-weight: 750 !important;
}

[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid var(--vg-border);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}

.metric-card {
    background: linear-gradient(145deg, #172033, #111827);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 18px;
    min-height: 125px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #38bdf8;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #5eead4;
    font-size: 12px;
    margin-top: 8px;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-color: var(--vg-border) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #172033;
    border-radius: 14px;
}

.stButton > button, .stDownloadButton > button {
    background: var(--vg-primary) !important;
    color: #ffffff !important;
    border: 1px solid var(--vg-primary) !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--vg-primary-dark) !important;
    border-color: var(--vg-primary-dark) !important;
    color: #ffffff !important;
}

.stTextInput input, .stNumberInput input, .stDateInput input,
.stTimeInput input, .stTextArea textarea,
[data-baseweb="select"] > div {
    background: #172033 !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
    border-radius: 8px !important;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #334155 !important;
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS service_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_date TEXT NOT NULL,
                odometer_km REAL NOT NULL,
                service_type TEXT NOT NULL,
                cost REAL NOT NULL,
                next_service_km REAL,
                notes TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS digital_twin_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                simulation_mode TEXT NOT NULL,
                component TEXT NOT NULL,
                engine_temp REAL NOT NULL,
                rpm REAL NOT NULL,
                battery_voltage REAL NOT NULL,
                speed REAL NOT NULL,
                tyre_pressure REAL NOT NULL,
                health_score INTEGER NOT NULL,
                rule_alert TEXT NOT NULL,
                ml_alert TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expense_date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                notes TEXT
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

from openai import OpenAI

def ask_vehicle_ai(pm_result, temp, voltage, speed):
    prompt = f"""
    Tum VehicleGPT ke AI Vehicle Assistant ho.
    User ko saral Hindi mein jawab do.

    Vehicle readings:
    Engine temperature: {temp} °C
    Battery voltage: {voltage} V
    Vehicle speed: {speed} km/h

    Predictive Maintenance risk:
    {pm_result.get("risk_level", "Not specified")}

    Maintenance priority:
    {pm_result.get("maintenance_priority", "Not specified")}

    Alerts:
    {pm_result.get("alerts", [])}

    Recommended actions:
    {pm_result.get("recommendations", [])}

    Risk ka matlab samjhao.
    User ko batao ki ab kya karna chahiye.
    Zarurat hone par mechanic se jaanch karwane ki salah do.
    Bina evidence ke kisi fault ki pushti mat karo.
    """

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tum VehicleGPT ke helpful AI Vehicle Assistant ho."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
        )

        return response.choices[0].message.content

    except Exception as e:
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
    st.title("🚗 VehicleGPT")
    st.caption("AI-Based Vehicle Management System")
    st.divider()

    page = st.radio(
        "Choose Section",
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
            "🛠️ OBD-II Fault Code Analyzer",
            "💰 Repair Cost Estimator",
            "📘 Digital Service Book",
            "⛽ Fuel Efficiency Analyzer",
            "💸 Vehicle Expense Tracker",
            "🧩 Component Life Estimator",
            "📍 Nearby Service Center Finder",
            "🚦 Driver Behavior Analysis",
            "🎙️ Voice-Based Vehicle Assistant", 
            "🔮 Predictive Maintenance ML",
            "💚 Vehicle Health Score",
            "⛽ AI Fuel Optimization",
            "🚨 Emergency SOS System",
            "👨‍💼 Admin & Fleet Dashboard",
            "📑 AI Diagnostic Report Generator",
            "📡 Real-Time Vehicle Monitoring",
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

# ---------------- Temperature and Speed Meters ----------------

gauge1, gauge2 = st.columns(2)

with gauge1:
    temp_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pm_temp,
            title={"text": "Engine Temperature (°C)"},
            gauge={
                "axis": {"range": [0, 140]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 95], "color": "#1f6b4a"},
                    {"range": [95, 110], "color": "#9a6b17"},
                    {"range": [110, 140], "color": "#8b2d35"},
                ],
            },
        )
    )

    temp_gauge.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )

    st.plotly_chart(temp_gauge, use_container_width=True)

with gauge2:
    speed_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pm_speed,
            title={"text": "Vehicle Speed (km/h)"},
            gauge={
                "axis": {"range": [0, 180]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 120], "color": "#1f6b4a"},
                    {"range": [120, 150], "color": "#9a6b17"},
                    {"range": [150, 180], "color": "#8b2d35"},
                ],
            },
        )
    )

    speed_gauge.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )

    st.plotly_chart(speed_gauge, use_container_width=True)

# ---------------- Analyze Maintenance ----------------

if st.button("Analyze Maintenance Status", key="pm_analyze"):

    # Sensor values
    pm_temp = st.session_state.get("saved_pm_temp", pm_temp)
    pm_voltage = st.session_state.get("saved_pm_voltage", 12)
    pm_speed = st.session_state.get("saved_pm_speed", pm_speed)

    # Maintenance risk calculate karna
    if pm_temp >= 110 or pm_voltage < 10:
        risk_level = "CRITICAL"
        risk_message = "Vehicle ko turant check karwana chahiye."

    elif pm_temp >= 95 or pm_voltage < 11:
        risk_level = "HIGH"
        risk_message = "Vehicle ki maintenance check karne ki zaroorat hai."

    else:
        risk_level = "NORMAL"
        risk_message = "Sensor readings normal range mein hain."

    # AI assistant ke liye result save karna
    pm_result = {
    "risk_level": risk_level,
    "maintenance_priority": risk_level,
    "temperature": pm_temp,
    "voltage": pm_voltage,
    "speed": pm_speed,
    "message": risk_message,
}

    st.session_state.pm_result = pm_result

    # Risk result dikhana
    st.markdown("### Maintenance Risk Assessment")

    if risk_level == "CRITICAL":
        st.error(f"Risk Level: {risk_level} — {risk_message}")

    elif risk_level == "HIGH":
        st.warning(f"Risk Level: {risk_level} — {risk_message}")

    else:
        st.success(f"Risk Level: {risk_level} — {risk_message}")


# ---------------- AI Vehicle Assistant ----------------

st.markdown("### 🤖 AI Vehicle Assistant")

if st.button("Explain Results with AI", key="ask_vehicle_ai"):

    if "pm_result" not in st.session_state:
        st.warning("Pehle Analyze Maintenance Status button dabayein.")

    else:
        with st.spinner("AI Hindi mein explanation taiyar kar raha hai..."):

            ai_answer = "AI Assistant फिलहाल बंद है। बाकी VehicleGPT features इस्तेमाल कर सकते हैं।"
        st.session_state.pm_ai_answer = ai_answer


# AI explanation dikhana
if "pm_ai_answer" in st.session_state:
    st.markdown("#### AI Explanation")
    st.write(st.session_state.pm_ai_answer)

    # Saved maintenance result lena
    pm_result = st.session_state.get("pm_result", {})

    # Maintenance Priority
    priority = pm_result.get("maintenance_priority", "Not specified")
    st.info(f"Maintenance Priority: {priority}")

    # Maintenance Alerts
    st.markdown("### Maintenance Alerts")

    alerts = pm_result.get("alerts", [])

    if alerts:
        for alert in alerts:
            if alert.startswith("CRITICAL"):
                st.error(alert)
            elif alert.startswith("WARNING"):
                st.warning(alert)
            elif alert.startswith("INVALID"):
                st.info(alert)
            else:
                st.success(alert)
    else:
        st.info("Koi maintenance alert available nahi hai.")

    # Recommended Actions
    st.markdown("### Recommended Actions")

    recommendations = pm_result.get("recommendations", [])

    if recommendations:
        for recommendation in recommendations:
            st.write("- " + str(recommendation))
    else:
        st.info("Abhi koi recommended action available nahi hai.")

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
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font_color="#F1F5F9",
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

    fig.update_layout(
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font_color="#F1F5F9",
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




# ---------------------- NEW FEATURE: OBD-II CODE ANALYZER ----------------------
elif page == "🛠️ OBD-II Fault Code Analyzer":
    st.title("🛠️ OBD-II Fault Code Analyzer")
    st.caption(
        "Enter a diagnostic trouble code (DTC) to view general educational information. "
        "A code alone does not confirm which part has failed."
    )

    obd_codes = {
        "P0300": {
            "title": "Random / Multiple Cylinder Misfire Detected",
            "meaning": "Engine misfire detected in one or more cylinders.",
            "causes": "Possible causes include spark plugs, ignition coils, fuel delivery, vacuum leaks, or engine mechanical issues.",
            "action": "Have the code and freeze-frame data checked by a qualified technician. Avoid hard acceleration if the engine is running roughly.",
            "cost": (1500, 12000),
        },
        "P0171": {
            "title": "System Too Lean (Bank 1)",
            "meaning": "The engine control system has detected a lean air-fuel mixture.",
            "causes": "Possible causes include an intake/vacuum leak, dirty or faulty mass-airflow sensor, fuel pressure issue, or exhaust leak.",
            "action": "Inspect for air leaks and ask a technician to check fuel trims and sensor readings before replacing parts.",
            "cost": (1000, 10000),
        },
        "P0420": {
            "title": "Catalyst System Efficiency Below Threshold",
            "meaning": "Catalyst efficiency is below the threshold monitored by the vehicle computer.",
            "causes": "Possible causes include catalytic converter deterioration, exhaust leaks, or oxygen-sensor issues.",
            "action": "Confirm with diagnostic tests before replacing the catalytic converter.",
            "cost": (1500, 35000),
        },
        "P0128": {
            "title": "Coolant Thermostat Temperature Below Regulating Temperature",
            "meaning": "The engine is not reaching the expected operating temperature within the monitored conditions.",
            "causes": "Possible causes include a thermostat stuck open, coolant temperature sensor issue, or cooling-system problem.",
            "action": "Check coolant level only when the engine is cool and arrange a cooling-system inspection.",
            "cost": (1000, 7000),
        },
        "P0562": {
            "title": "System Voltage Low",
            "meaning": "The vehicle computer has detected low system voltage.",
            "causes": "Possible causes include a weak battery, alternator/charging issue, loose terminals, or wiring problem.",
            "action": "Have the battery, alternator output, terminals, and wiring tested.",
            "cost": (800, 12000),
        },
        "P0442": {
            "title": "EVAP System Small Leak Detected",
            "meaning": "A small leak has been detected in the evaporative-emissions system.",
            "causes": "Possible causes include a loose/damaged fuel cap, hose leak, or EVAP-system component issue.",
            "action": "Check that the fuel cap is properly closed. If the code returns, request an EVAP leak test.",
            "cost": (500, 8000),
        },
    }

    code = st.selectbox("Select a sample OBD-II code", ["P0300", "P0171", "P0420", "P0128", "P0562", "P0442"])
    details = obd_codes[code]
    st.subheader(f"{code} — {details['title']}")
    st.info(details["meaning"])
    st.markdown("**Possible causes**")
    st.write(details["causes"])
    st.markdown("**Suggested next step**")
    st.write(details["action"])
    st.caption("Educational reference only. OBD-II code definitions and repair steps vary by vehicle make, model, year, and engine. Confirm using the vehicle's service information.")

    st.divider()
    st.subheader("🤖 AI-Based Fault Diagnosis")

    if st.button("Analyze Fault with AI", key="ai_obd_diagnosis"):
        prompt = f"""
        You are VehicleGPT, a vehicle maintenance assistant.
        Explain in simple Hindi.

        OBD-II Code: {code}
        Fault Title: {details['title']}
        Meaning: {details['meaning']}
        Possible Causes: {details['causes']}
        Suggested Next Step: {details['action']}

        Explain:
        1. Is fault ka simple meaning.
        2. Possible causes.
        3. Severity: Low, Medium, or High.
        4. Safe next steps.
        5. Mechanic ko kya check karna chahiye.

        Do not claim a component has definitely failed.
        Do not invent vehicle-specific facts.
        Mention that a qualified technician should confirm the fault.
        """

        try:
            with st.spinner("AI fault analysis kar raha hai..."):
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
                ai_result = response.json().get(
                    "response",
                    "AI se response nahi mila."
                )

            st.markdown("### AI Diagnosis Result")
            st.write(ai_result)

        except requests.RequestException as exc:
            st.error(f"AI connection error: {exc}")


# ---------------------- NEW FEATURE: REPAIR COST ESTIMATOR ----------------------
elif page == "💰 Repair Cost Estimator":
    st.title("💰 Repair Cost Estimator")
    st.caption("Illustrative Indian-rupee ranges only; not a quotation from a workshop.")
    repair_options = {
        "Battery / charging system check": (300, 1500),
        "Spark plug / ignition inspection": (500, 4000),
        "Cooling system inspection": (500, 2500),
        "Thermostat replacement (estimate)": (1500, 7000),
        "Brake inspection / basic service": (500, 3000),
        "General periodic service": (2000, 8000),
        "OBD-II diagnostic scan": (300, 1500),
    }
    selected_repair = st.selectbox("Choose a repair or inspection", list(repair_options.keys()))
    low_cost, high_cost = repair_options[selected_repair]
    st.markdown("### Estimated cost range")
    c1, c2 = st.columns(2)
    c1.metric("Lower estimate", f"₹{low_cost:,}")
    c2.metric("Upper estimate", f"₹{high_cost:,}")
    st.info("Actual cost depends on vehicle type, city, workshop labour, parts quality, and the confirmed fault. Get a written quote before repair.")


# ---------------------- NEW FEATURE: DIGITAL SERVICE BOOK ----------------------
elif page == "📘 Digital Service Book":
    st.title("📘 Digital Vehicle Service Book")
    st.caption("Store service and repair entries locally in vehicle_history.db.")

    with st.form("service_book_form", clear_on_submit=True):
        service_date = st.date_input("Service date")
        odometer = st.number_input("Odometer reading (km)", min_value=0.0, max_value=2000000.0, value=0.0, step=100.0)
        service_type = st.selectbox("Service type", [
            "General service", "Engine oil change", "Brake service",
            "Battery service", "Tyre replacement", "Repair", "Other"
        ])
        service_cost = st.number_input("Service cost (₹)", min_value=0.0, max_value=1000000.0, value=0.0, step=100.0)
        next_service_km = st.number_input("Next service due at (km, optional)", min_value=0.0, max_value=2000000.0, value=0.0, step=500.0)
        service_notes = st.text_area("Notes / parts changed")
        service_submitted = st.form_submit_button("Save Service Record")

    if service_submitted:
        if odometer < 0 or service_cost < 0:
            st.error("Please enter valid non-negative values.")
        else:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO service_book
                    (service_date, odometer_km, service_type, cost, next_service_km, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    service_date.isoformat(), float(odometer), service_type,
                    float(service_cost), float(next_service_km) if next_service_km > 0 else None,
                    service_notes.strip()
                ))
                conn.commit()
            st.success("Service record saved successfully.")
            st.rerun()

    with sqlite3.connect(DB_PATH) as conn:
        service_df = pd.read_sql_query(
            "SELECT * FROM service_book ORDER BY service_date DESC, id DESC", conn
        )
    st.subheader("Saved service history")
    if service_df.empty:
        st.info("No service records saved yet.")
    else:
        st.dataframe(service_df, use_container_width=True, hide_index=True)
        total_service_cost = float(service_df["cost"].sum())
        st.metric("Total recorded service cost", f"₹{total_service_cost:,.2f}")
        csv_data = service_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Service Book CSV", data=csv_data,
            file_name="VehicleGPT_Service_Book.csv", mime="text/csv"
        )


# ---------------------- NEW FEATURE: FUEL EFFICIENCY ANALYZER ----------------------
elif page == "⛽ Fuel Efficiency Analyzer":
    st.title("⛽ Fuel Efficiency Analyzer")
    st.caption("Calculate mileage from user-entered distance and fuel used.")
    with st.form("fuel_efficiency_form"):
        distance_km = st.number_input("Distance travelled (km)", min_value=0.0, max_value=100000.0, value=100.0, step=10.0)
        fuel_litres = st.number_input("Fuel consumed (litres)", min_value=0.0, max_value=10000.0, value=5.0, step=0.5)
        fuel_price = st.number_input("Fuel price per litre (₹, optional)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        fuel_submitted = st.form_submit_button("Calculate Mileage")
    if fuel_submitted:
        if fuel_litres <= 0:
            st.error("Fuel consumed must be greater than zero.")
        else:
            mileage = distance_km / fuel_litres
            st.metric("Calculated mileage", f"{mileage:.2f} km/L")
            if fuel_price > 0:
                trip_cost = fuel_litres * fuel_price
                st.metric("Fuel cost for this trip", f"₹{trip_cost:,.2f}")
                st.metric("Fuel cost per km", f"₹{trip_cost / distance_km:.2f}" if distance_km > 0 else "Not available")
            st.caption("This is a calculation from the values you entered, not a live vehicle measurement.")


# ---------------------- NEW FEATURE: VEHICLE EXPENSE TRACKER ----------------------
elif page == "💸 Vehicle Expense Tracker":
    st.title("💸 Vehicle Expense Tracker")
    st.caption("Record fuel, service, repair, and other vehicle expenses in the local database.")

    with st.form("vehicle_expense_form", clear_on_submit=True):
        expense_date = st.date_input("Expense date")
        expense_category = st.selectbox("Expense category", [
            "Fuel", "Service", "Repair", "Insurance", "Tyres", "Other"
        ])
        expense_amount = st.number_input("Amount (₹)", min_value=0.0, max_value=10000000.0, value=0.0, step=100.0)
        expense_notes = st.text_input("Description / notes")
        expense_submitted = st.form_submit_button("Save Expense")

    if expense_submitted:
        if expense_amount <= 0:
            st.error("Please enter an amount greater than zero.")
        else:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO vehicle_expenses (expense_date, category, amount, notes)
                    VALUES (?, ?, ?, ?)
                """, (expense_date.isoformat(), expense_category, float(expense_amount), expense_notes.strip()))
                conn.commit()
            st.success("Expense saved successfully.")
            st.rerun()

    with sqlite3.connect(DB_PATH) as conn:
        expense_df = pd.read_sql_query(
            "SELECT * FROM vehicle_expenses ORDER BY expense_date DESC, id DESC", conn
        )
    if expense_df.empty:
        st.info("No vehicle expenses recorded yet.")
    else:
        st.metric("Total recorded expenses", f"₹{float(expense_df['amount'].sum()):,.2f}")
        by_category = expense_df.groupby("category", as_index=False)["amount"].sum()
        expense_chart = px.pie(
            by_category, names="category", values="amount",
            title="Expenses by Category", template="plotly_dark"
        )
        st.plotly_chart(expense_chart, use_container_width=True)
        st.dataframe(expense_df, use_container_width=True, hide_index=True)
        expense_csv = expense_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Expense History CSV", data=expense_csv,
            file_name="VehicleGPT_Expense_History.csv", mime="text/csv"
        )


# ---------------------- NEW FEATURE 6: COMPONENT LIFE ESTIMATOR ----------------------
elif page == "🧩 Component Life Estimator":
    st.title("🧩 Component Life Estimator")
    st.caption(
        "Illustrative maintenance-interval calculator. It is not a sensor-based "
        "prediction and does not measure actual component condition."
    )

    component_guides = {
        "Engine oil / oil filter": {"km": 10000, "months": 12, "basis": "km and time"},
        "Air filter": {"km": 15000, "months": 12, "basis": "km and time"},
        "Brake pads": {"km": 30000, "months": 24, "basis": "km and time"},
        "Tyres": {"km": 40000, "months": 48, "basis": "km and time"},
        "Battery": {"km": None, "months": 36, "basis": "time"},
        "Spark plugs": {"km": 30000, "months": 24, "basis": "km and time"},
        "Coolant": {"km": 40000, "months": 24, "basis": "km and time"},
    }

    selected_component = st.selectbox(
        "Select vehicle component", list(component_guides.keys())
    )
    guide = component_guides[selected_component]
    st.info(
        "Reference interval shown for demonstration only. Actual intervals "
        "vary by make, model, engine, driving conditions and the owner's manual."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        km_since_change = st.number_input(
            "Distance since last replacement/service (km)",
            min_value=0, max_value=1000000, value=0, step=500,
            disabled=(guide["km"] is None),
        )
    with col_b:
        months_since_change = st.number_input(
            "Months since last replacement/service",
            min_value=0, max_value=600, value=0, step=1,
        )

    if st.button("Estimate Component Life", key="component_life_estimate"):
        usage_ratios = []
        if guide["km"] is not None:
            usage_ratios.append(km_since_change / guide["km"])
        if guide["months"] is not None:
            usage_ratios.append(months_since_change / guide["months"])

        used_ratio = min(1.0, max(usage_ratios) if usage_ratios else 0.0)
        remaining_percent = max(0.0, (1.0 - used_ratio) * 100.0)
        st.metric("Illustrative remaining interval", f"{remaining_percent:.0f}%")
        st.progress(int(round(remaining_percent)))
        if guide["km"] is not None:
            km_left = max(0, guide["km"] - km_since_change)
            st.metric("Reference distance remaining", f"{km_left:,} km")
        if guide["months"] is not None:
            months_left = max(0, guide["months"] - months_since_change)
            st.metric("Reference time remaining", f"{months_left} months")
        if used_ratio >= 1:
            st.warning("The entered usage has reached or exceeded this illustrative interval. Arrange an inspection/service.")
        else:
            st.success("The entered usage is within the illustrative interval. Follow the vehicle owner's manual and inspect the component regularly.")


# ---------------------- NEW FEATURE 7: NEARBY SERVICE CENTER FINDER ----------------------
elif page == "📍 Nearby Service Center Finder":
    st.title("📍 Nearby Vehicle Service Center Finder")
    st.caption(
        "Create a Google Maps search for workshops near a place you enter. "
        "This demo does not access your live location or verify workshop availability."
    )

    search_place = st.text_input(
        "Enter city, area or PIN code", placeholder="Example: Kanpur, Uttar Pradesh"
    )
    service_kind = st.selectbox(
        "What type of service do you need?",
        ["Car service center", "Car mechanic", "Two-wheeler service center", "Tyre shop", "Battery shop", "Car AC repair"],
    )
    if st.button("Find Nearby Service Centers", key="find_service_centers"):
        if not search_place.strip():
            st.warning("Please enter a city, area or PIN code.")
        else:
            maps_query = quote_plus(f"{service_kind} near {search_place.strip()}")
            maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"
            st.success("Your map search is ready.")
            st.markdown(f"[Open {service_kind} results in Google Maps]({maps_url})")
            st.caption("Check ratings, opening hours, distance and phone details directly in Maps. Confirm prices and availability with the workshop.")


# ---------------------- NEW FEATURE 8: DRIVER BEHAVIOR ANALYSIS ----------------------
elif page == "🚦 Driver Behavior Analysis":
    st.title("🚦 Driver Behavior Analysis")
    st.caption(
        "Enter trip event counts to calculate a simple demonstration score. "
        "No driving events are detected automatically in this software-only prototype."
    )

    with st.form("driver_behavior_form"):
        trip_distance = st.number_input(
            "Trip distance (km)", min_value=0.1, max_value=10000.0,
            value=100.0, step=10.0,
        )
        overspeed_events = st.number_input(
            "Overspeed events", min_value=0, max_value=10000, value=0, step=1,
        )
        harsh_brakes = st.number_input(
            "Harsh braking events", min_value=0, max_value=10000, value=0, step=1,
        )
        rapid_accelerations = st.number_input(
            "Rapid acceleration events", min_value=0, max_value=10000, value=0, step=1,
        )
        behavior_submitted = st.form_submit_button("Analyze Driving Pattern")

    if behavior_submitted:
        events_per_100km = 100.0 / float(trip_distance)
        penalty = events_per_100km * (
            3.0 * overspeed_events + 2.0 * harsh_brakes + 1.5 * rapid_accelerations
        )
        driving_score = max(0, min(100, round(100 - penalty)))
        st.metric("Illustrative driving score", f"{driving_score}/100")
        st.progress(driving_score)
        d1, d2, d3 = st.columns(3)
        d1.metric("Overspeed events", int(overspeed_events))
        d2.metric("Harsh braking", int(harsh_brakes))
        d3.metric("Rapid acceleration", int(rapid_accelerations))
        if driving_score >= 85:
            st.success("Entered event counts indicate a smoother trip under this demo scoring rule.")
        elif driving_score >= 60:
            st.warning("Review the entered events and aim for smoother acceleration and braking.")
        else:
            st.error("The entered event counts produce a low demo score. Review speed and driving habits.")
        st.caption("Scoring formula is illustrative, not a certified driver-risk score. Real analysis requires validated trip/sensor data and context such as road type and speed limits.")


# ---------------------- NEW FEATURE 9: VOICE-BASED VEHICLE ASSISTANT ----------------------
elif page == "🎙️ Voice-Based Vehicle Assistant":
    st.title("🎙️ Voice-Based Vehicle Assistant")
    st.caption(
        "Ask a general vehicle-maintenance question by text or microphone. "
        "Voice transcription and spoken output require internet access and optional packages."
    )

    voice_question = st.text_area(
        "Type your question (or use the microphone below)",
        placeholder="Example: Engine temperature high hone par kya karna chahiye?",
        key="voice_assistant_text_question",
    )
    recorded_audio = None
    if hasattr(st, "audio_input"):
        recorded_audio = st.audio_input("Record your question")
    else:
        st.info("Your installed Streamlit version does not provide microphone recording. You can still type your question above.")

    if st.button("Get VehicleGPT Answer", key="voice_assistant_answer"):
        final_question = voice_question.strip()
        if recorded_audio is not None:
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(recorded_audio.getvalue())) as audio_source:
                    audio_data = recognizer.record(audio_source)
                detected_text = recognizer.recognize_google(audio_data, language="hi-IN")
                st.write("Recognized voice:", detected_text)
                if not final_question:
                    final_question = detected_text
            except ImportError:
                st.error("Voice transcription package missing. Install with: python -m pip install SpeechRecognition")
            except Exception as exc:
                st.warning(f"Voice could not be transcribed: {exc}. Please type your question instead.")

        if not final_question:
            st.warning("Please type a question or record a clear voice question.")
        else:
            assistant_prompt = (
                "You are VehicleGPT, a cautious vehicle maintenance information assistant. "
                "Answer in simple Hindi. Give general educational guidance, do not claim a "
                "confirmed diagnosis, and recommend a qualified technician for safety issues. "
                "Never tell the user to open a hot radiator cap or work on a moving vehicle.\n"
                f"User question: {final_question}"
            )
            try:
                assistant_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2:1b", "prompt": assistant_prompt, "stream": False},
                    timeout=90,
                )
                assistant_response.raise_for_status()
                answer_text = assistant_response.json().get("response", "AI se jawab nahi mila.")
                st.markdown("### VehicleGPT Answer")
                st.write(answer_text)
                try:
                    from gtts import gTTS
                    speech_buffer = io.BytesIO()
                    gTTS(text=answer_text, lang="hi").write_to_fp(speech_buffer)
                    speech_buffer.seek(0)
                    st.audio(speech_buffer.getvalue(), format="audio/mp3")
                except ImportError:
                    st.info("Text answer is ready. For spoken Hindi output, install gTTS: python -m pip install gTTS")
                except Exception as speech_exc:
                    st.info(f"Text answer is ready; speech playback could not be created: {speech_exc}")
            except requests.RequestException as exc:
                st.error("Ollama connect nahi hua. Ollama start karein aur llama3.2:1b model available rakhein. Details: " + str(exc))


# ---------------------- NEW FEATURE 10: 3D VEHICLE DIGITAL TWIN ----------------------

elif page == "🧪 Advanced ML & Predictive Maintenance":
    st.title("🧪 ML-Based Anomaly Detection & Predictive Maintenance")
    st.caption(
        "Educational prototype: this page trains an Isolation Forest on "
        "synthetic normal-range sensor examples. It is not a validated "
        "real-vehicle diagnostic model."
    )

    st.subheader("1. Enter sensor readings")
    a1, a2, a3 = st.columns(3)
    with a1:
        ml_temp = st.number_input("Engine temperature (°C)", 0.0, 200.0, 92.0, 0.5, key="adv_temp")
        ml_rpm = st.number_input("Engine RPM", 0.0, 10000.0, 1800.0, 50.0, key="adv_rpm")
    with a2:
        ml_voltage = st.number_input("Battery voltage (V)", 0.0, 24.0, 12.6, 0.1, key="adv_voltage")
        ml_speed = st.number_input("Speed (km/h)", 0.0, 300.0, 40.0, 1.0, key="adv_speed")
    with a3:
        ml_tyre = st.number_input("Tyre pressure (PSI)", 0.0, 80.0, 33.0, 0.5, key="adv_tyre")
        ml_contamination = st.selectbox("Data source", ["Software simulation / manual input"])

    if st.button("Run ML Anomaly Detection", type="primary", key="run_advanced_ml"):
        # Synthetic reference data is explicitly labelled as a demonstration baseline.
        rng = np.random.default_rng(42)
        baseline = pd.DataFrame({
            "engine_temp": rng.normal(90, 4, 700).clip(70, 99),
            "rpm": rng.normal(1800, 650, 700).clip(650, 3900),
            "battery_voltage": rng.normal(13.1, 0.45, 700).clip(12.2, 14.6),
            "speed": rng.normal(45, 25, 700).clip(0, 120),
            "tyre_pressure": rng.normal(33, 1.8, 700).clip(29, 38),
        })
        model = IsolationForest(contamination=0.04, random_state=42)
        model.fit(baseline)
        sample = pd.DataFrame([{
            "engine_temp": ml_temp,
            "rpm": ml_rpm,
            "battery_voltage": ml_voltage,
            "speed": ml_speed,
            "tyre_pressure": ml_tyre,
        }])
        prediction = int(model.predict(sample)[0])
        anomaly_score = float(model.decision_function(sample)[0])
        rule_alerts = []
        if ml_temp >= 100:
            rule_alerts.append("Engine temperature high (configured threshold ≥100°C)")
        if ml_rpm > 4000:
            rule_alerts.append("RPM unusually high (configured threshold >4000)")
        if ml_voltage < 12.0:
            rule_alerts.append("Battery voltage low (configured threshold <12V)")
        if ml_tyre < 28:
            rule_alerts.append("Tyre pressure low (configured threshold <28 PSI)")
        if prediction == -1:
            st.warning("ML result: unusual sensor pattern flagged.")
        else:
            st.success("ML result: no unusual pattern flagged by this demo model.")
        st.metric("Isolation Forest anomaly score", f"{anomaly_score:.4f}")
        if rule_alerts:
            st.error("Rule-based alerts")
            for alert in rule_alerts:
                st.write("• " + alert)
        else:
            st.success("No configured rule threshold was crossed.")
        st.info(
            "ML result is a pattern flag, not a confirmed fault or failure prediction. "
            "The reference data is synthetic; accuracy must be validated using "
            "real, labelled vehicle data before real-world use."
        )
        risk_count = len(rule_alerts) + (1 if prediction == -1 else 0)
        if risk_count >= 3:
            priority = "High demo risk — arrange professional inspection"
        elif risk_count >= 1:
            priority = "Moderate demo risk — recheck readings and monitor"
        else:
            priority = "Low demo risk — continue routine checks"
        st.subheader("2. Predictive maintenance guidance")
        st.write(priority)
        actions = []
        if ml_temp >= 100:
            actions.append("Check coolant level, radiator, cooling fan and temperature sensor.")
        if ml_voltage < 12.0:
            actions.append("Check battery condition, terminals and charging system.")
        if ml_tyre < 28:
            actions.append("Check tyre for leakage and inflate to the vehicle-maker's recommended PSI.")
        if ml_rpm > 4000:
            actions.append("Check RPM reading and avoid sustained high engine speed.")
        if not actions:
            actions.append("No threshold-based action is indicated by these entered values; follow the scheduled service plan.")
        for action in actions:
            st.write("• " + action)

        status_text = "ANOMALY" if prediction == -1 else "NO ANOMALY FLAG"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO digital_twin_history
                (timestamp, simulation_mode, component, engine_temp, rpm,
                 battery_voltage, speed, tyre_pressure, health_score, rule_alert, ml_alert)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                "Manual / Advanced ML input",
                "All components",
                float(ml_temp), float(ml_rpm), float(ml_voltage),
                float(ml_speed), float(ml_tyre),
                max(0, 100 - 20 * len(rule_alerts) - (15 if prediction == -1 else 0)),
                " | ".join(rule_alerts) if rule_alerts else "No rule alert",
                status_text,
            ))
        st.success("Analysis saved in local SQLite history.")

    st.divider()
    st.subheader("3. Recent digital-twin / ML records")
    with sqlite3.connect(DB_PATH) as conn:
        recent_ml = pd.read_sql_query(
            "SELECT timestamp, engine_temp, rpm, battery_voltage, speed, tyre_pressure, rule_alert, ml_alert "
            "FROM digital_twin_history ORDER BY id DESC LIMIT 20", conn
        )
    if recent_ml.empty:
        st.info("No advanced ML records saved yet. Run an analysis above.")
    else:
        st.dataframe(recent_ml, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ML history CSV",
            recent_ml.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_ml_history.csv",
            mime="text/csv",
        )
elif page == "🔮 Predictive Maintenance ML":
    st.title("🔮 Predictive Maintenance ML")
    st.caption("Sample data पर आधारित Demo ML Model")

    st.subheader("Vehicle Sensor Inputs")

    temp_value = st.number_input(
        "Engine Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=90.0,
        key="new_pm_temp"
    )

    voltage_value = st.number_input(
        "Battery Voltage (V)",
        min_value=0.0,
        max_value=24.0,
        value=12.4,
        key="new_pm_voltage"
    )

    speed_value = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=300.0,
        value=40.0,
        key="new_pm_speed"
    )

    if st.button("Predict Maintenance", key="new_pm_predict"):
        try:
            result = predict_maintenance(
                40,
                temp_value,
                voltage_value,
                speed_value
            )

            st.metric("Model Status", result["status"])
            st.metric("Illustrative Risk", f'{result["risk"]}%')

            st.warning(
                "यह classroom demo है। इसे वास्तविक वाहन की "
                "सुरक्षा या सर्विस का प्रमाणित अनुमान न मानें।"
            )

        except Exception as exc:
            st.error(f"Prediction Error: {exc}")
elif page == "⛽ AI Fuel Optimization":
    st.title("⛽ AI Fuel Optimization")
    st.caption("Fuel saving analysis and recommendations")

    st.subheader("Enter Vehicle Details")

    distance = st.number_input(
        "Travel Distance (KM)",
        min_value=1.0,
        value=100.0,
        step=1.0,
        key="fuel_distance"
    )

    mileage = st.number_input(
        "Vehicle Mileage (KM/L)",
        min_value=1.0,
        value=15.0,
        step=0.5,
        key="fuel_mileage"
    )

    fuel_price = st.number_input(
        "Petrol Price (₹/L)",
        min_value=1.0,
        value=100.0,
        step=1.0,
        key="fuel_price"
    )

    if st.button("Analyze Fuel Usage", key="fuel_analyze"):
        fuel_used = distance / mileage
        fuel_cost = fuel_used * fuel_price

        st.subheader("Fuel Analysis")

        col1, col2 = st.columns(2)

        col1.metric("Estimated Fuel Used", f"{fuel_used:.2f} L")
        col2.metric("Estimated Fuel Cost", f"₹{fuel_cost:.2f}")

        st.subheader("AI Fuel Saving Suggestions")

        st.success("Maintain steady speed and avoid sudden acceleration.")
        st.info("Keep tyre pressure at the recommended level.")
        st.info("Avoid unnecessary idling and remove extra vehicle load.")
        st.info("Follow regular vehicle servicing.")

        st.caption(
            "Note: This is a calculation-based educational demo, "
            "not a trained AI model or guaranteed mileage prediction."
        )
elif page == "🚨 Emergency SOS System":
    st.title("🚨 Emergency SOS System")
    st.caption("Emergency assistance demo")

    st.subheader("Emergency Contact Details")

    contact = st.text_input(
        "Emergency Contact Number",
        placeholder="Enter mobile number",
        key="sos_contact"
    )

    location = st.text_input(
        "Your Current Location",
        placeholder="Enter city or location manually",
        key="sos_location"
    )

    emergency_type = st.selectbox(
        "Select Emergency Type",
        [
            "Vehicle Accident",
            "Vehicle Breakdown",
            "Medical Emergency",
            "Other Emergency"
        ],
        key="sos_type"
    )

    if st.button("🚨 ACTIVATE SOS", key="activate_sos"):
        if not contact.strip() or not location.strip():
            st.error("Please enter emergency contact and location.")
        else:
            message = (
                "EMERGENCY SOS! "
                f"Emergency: {emergency_type}. "
                f"Location: {location}. "
                "Please contact me immediately."
            )

            st.error("SOS ALERT GENERATED!")
            st.write("**Emergency Contact:**", contact)
            st.write("**Emergency Message:**")
            st.code(message)

            import urllib.parse

            whatsapp_url = (
                "https://wa.me/"
                + "".join(ch for ch in contact if ch.isdigit())
                + "?text="
                + urllib.parse.quote(message)
            )

            st.link_button(
                "Send SOS through WhatsApp",
                whatsapp_url
            )

            st.warning(
                "Demo only: SOS is not automatically sent. "
                "You must open WhatsApp and press Send. "
                "No real GPS tracking or emergency-service connection is active."
            )


elif page == "👨‍💼 Admin & Fleet Dashboard":
    st.title("👨‍💼 Admin & Fleet Management Dashboard")
    st.caption("Vehicle fleet management - Demo")

    fleet_data = pd.DataFrame([
        {
            "Vehicle Number": "UP78 AB 1234",
            "Model": "Maruti Swift",
            "Owner": "Rahul",
            "Status": "Active"
        },
        {
            "Vehicle Number": "UP78 CD 5678",
            "Model": "Hyundai Creta",
            "Owner": "Amit",
            "Status": "Maintenance"
        },
        {
            "Vehicle Number": "UP78 EF 9012",
            "Model": "Tata Nexon",
            "Owner": "Rohit",
            "Status": "Active"
        },
        {
            "Vehicle Number": "UP78 GH 3456",
            "Model": "Mahindra Thar",
            "Owner": "Vikas",
            "Status": "Inactive"
        }
    ])

    total = len(fleet_data)
    active = len(fleet_data[fleet_data["Status"] == "Active"])
    maintenance = len(fleet_data[fleet_data["Status"] == "Maintenance"])
    inactive = len(fleet_data[fleet_data["Status"] == "Inactive"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Vehicles", total)
    col2.metric("Active", active)
    col3.metric("Maintenance", maintenance)
    col4.metric("Inactive", inactive)

    st.subheader("Fleet Vehicle Records")
    st.dataframe(fleet_data, use_container_width=True)

    st.subheader("Add New Vehicle")

    with st.form("add_fleet_vehicle_2"):
        vehicle_number = st.text_input("Vehicle Number")
        model_name = st.text_input("Vehicle Model")
        owner_name = st.text_input("Owner Name")

        vehicle_status = st.selectbox(
            "Vehicle Status",
            ["Active", "Maintenance", "Inactive"]
        )

        submitted = st.form_submit_button("Add Vehicle")

        if submitted:
            if (
                vehicle_number.strip()
                and model_name.strip()
                and owner_name.strip()
            ):
                new_vehicle = pd.DataFrame([{
                    "Vehicle Number": vehicle_number,
                    "Model": model_name,
                    "Owner": owner_name,
                    "Status": vehicle_status
                }])

                fleet_data = pd.concat(
                    [fleet_data, new_vehicle],
                    ignore_index=True
                )

                st.success("Vehicle added successfully!")
                st.dataframe(
                    fleet_data,
                    use_container_width=True
                )
            else:
                st.error("Please fill all vehicle details.")

    st.info(
        "Demo data only: Records are not permanently saved in database."
    )
    # Sample fleet data
    fleet_data = pd.DataFrame([
        {
            "Vehicle Number": "UP78 AB 1234",
            "Model": "Maruti Swift",
            "Owner": "Rahul",
            "Status": "Active"
        },
        {
            "Vehicle Number": "UP78 CD 5678",
            "Model": "Hyundai Creta",
            "Owner": "Amit",
            "Status": "Maintenance"
        },
        {
            "Vehicle Number": "UP78 EF 9012",
            "Model": "Tata Nexon",
            "Owner": "Rohit",
            "Status": "Active"
        },
        {
            "Vehicle Number": "UP78 GH 3456",
            "Model": "Mahindra Thar",
            "Owner": "Vikas",
            "Status": "Inactive"
        }
    ])

    st.subheader("Fleet Summary")

    total = len(fleet_data)
    active = len(fleet_data[fleet_data["Status"] == "Active"])
    maintenance = len(fleet_data[fleet_data["Status"] == "Maintenance"])
    inactive = len(fleet_data[fleet_data["Status"] == "Inactive"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Vehicles", total)
    col2.metric("Active", active)
    col3.metric("Maintenance", maintenance)
    col4.metric("Inactive", inactive)

    st.divider()

    st.subheader("Vehicle Records")
    st.dataframe(fleet_data, use_container_width=True)

    st.subheader("Add New Vehicle")

    with st.form("add_fleet_vehicle"):
        vehicle_number = st.text_input("Vehicle Number")
        model_name = st.text_input("Vehicle Model")
        owner_name = st.text_input("Owner Name")

        vehicle_status = st.selectbox(
            "Vehicle Status",
            ["Active", "Maintenance", "Inactive"]
        )

        submitted = st.form_submit_button("Add Vehicle")

        if submitted:
            if vehicle_number.strip() and model_name.strip() and owner_name.strip():
                new_vehicle = pd.DataFrame([{
                    "Vehicle Number": vehicle_number,
                    "Model": model_name,
                    "Owner": owner_name,
                    "Status": vehicle_status
                }])

                fleet_data = pd.concat(
                    [fleet_data, new_vehicle],
                    ignore_index=True
                )

                st.success("Vehicle added successfully!")
                st.dataframe(fleet_data, use_container_width=True)
            else:
                st.error("Please fill all vehicle details.")

    st.info(
        "Demo data only: Records are not permanently saved in the database. "
        "This dashboard does not monitor real vehicles."
    )
elif page == "📑 AI Diagnostic Report Generator":
    st.title("📑 AI Diagnostic Report Generator")
    st.caption("Vehicle diagnostic report - Educational Demo")

    st.subheader("Vehicle Health Summary")

    report_score = health_score(latest_temp, latest_voltage)

    col1, col2, col3 = st.columns(3)

    col1.metric("Health Score", f"{report_score}/100")
    col2.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    col3.metric("Battery Voltage", f"{latest_voltage:.1f} V")

    st.subheader("Diagnostic Observations")

    observations = []

    if latest_temp > 100:
        observations.append("Engine temperature is above the demo threshold.")
    else:
        observations.append("Engine temperature is within the demo threshold.")

    if latest_voltage < 12:
        observations.append("Battery voltage is below the demo threshold.")
    else:
        observations.append("Battery voltage is within the demo threshold.")

    for item in observations:
        st.write("•", item)

    st.subheader("Recommendations")

    st.write("1. Check engine cooling system if temperature is high.")
    st.write("2. Check battery and charging system if voltage is low.")
    st.write("3. Follow the vehicle manufacturer's service schedule.")

    st.divider()

    st.subheader("Generate PDF Report")

    from io import BytesIO
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    if st.button("Generate Diagnostic PDF", key="generate_diagnostic_pdf"):

        pdf_buffer = BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, 800, "VehicleGPT Diagnostic Report")

        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, 770, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

        pdf.drawString(50, 735, f"Health Score: {report_score}/100")
        pdf.drawString(50, 710, f"Engine Temperature: {latest_temp:.1f} C")
        pdf.drawString(50, 685, f"Battery Voltage: {latest_voltage:.1f} V")
        pdf.drawString(50, 660, f"Vehicle Speed: {latest_speed:.1f} km/h")

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, 620, "Diagnostic Observations")

        pdf.setFont("Helvetica", 10)
        y_position = 595

        for item in observations:
            pdf.drawString(60, y_position, "- " + item)
            y_position -= 20

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y_position - 15, "Recommendations")

        pdf.setFont("Helvetica", 10)
        pdf.drawString(60, y_position - 40, "- Check cooling system when needed.")
        pdf.drawString(60, y_position - 60, "- Check battery and charging system.")
        pdf.drawString(60, y_position - 80, "- Follow scheduled maintenance.")

        pdf.setFont("Helvetica-Oblique", 9)
        pdf.drawString(50, 80, "Educational demo based on simulated sensor readings.")
        pdf.drawString(50, 65, "Not a certified vehicle inspection or safety report.")

        pdf.save()

        pdf_buffer.seek(0)

        st.download_button(
            label="Download Diagnostic PDF",
            data=pdf_buffer.getvalue(),
            file_name="VehicleGPT_Diagnostic_Report.pdf",
            mime="application/pdf",
            key="download_diagnostic_pdf"
        )    
elif page == "💚 Vehicle Health Score":
    st.title("💚 Vehicle Health Score")

    st.write("Vehicle ke sensor readings ke basis par health assessment.")

    st.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    st.metric("Battery Voltage", f"{latest_voltage:.1f} V")
    st.metric("Vehicle Speed", f"{latest_speed:.1f} km/h")

    vehicle_score = health_score(latest_temp, latest_voltage)

    st.subheader("Overall Vehicle Health")
    st.progress(max(0, min(100, int(vehicle_score))))
    st.metric("Vehicle Health Score", f"{vehicle_score}/100")

    if vehicle_score >= 80:
        st.success("Health Status: Good")
    elif vehicle_score >= 60:
        st.warning("Health Status: Warning")
    else:
        st.error("Health Status: Critical")


elif page == "📡 Real-Time Vehicle Monitoring":
    st.title("📡 Real-Time Vehicle Monitoring")

    st.caption("VehicleGPT - Simulated Sensor Monitoring")

    st.subheader("Live Vehicle Sensor Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    col2.metric("Battery Voltage", f"{latest_voltage:.1f} V")
    col3.metric("Vehicle Speed", f"{latest_speed:.1f} km/h")

    st.subheader("Vehicle Monitoring Status")

    if latest_temp > 100:
        st.error("High Engine Temperature Alert")
    else:
        st.success("Engine Temperature Normal")

    if latest_voltage < 12:
        st.warning("Battery Voltage Low")
    else:
        st.success("Battery Voltage Normal")

    st.info("Note: This dashboard uses simulated sensor data. Real vehicle hardware is not connected.")

    # Existing sensor readings
    st.metric("Engine Temperature", f"{latest_temp:.1f} °C")
    st.metric("Battery Voltage", f"{latest_voltage:.1f} V")
    st.metric("Vehicle Speed", f"{latest_speed:.1f} km/h")

    # Existing health score function
    vehicle_score = health_score(latest_temp, latest_voltage)

    st.subheader("Overall Vehicle Health")

    st.progress(max(0, min(100, int(vehicle_score))))

    st.metric("Vehicle Health Score", f"{vehicle_score}/100")

    if vehicle_score >= 80:
        st.success("Health Status: Good")
    elif vehicle_score >= 60:
        st.warning("Health Status: Warning")
    else:
        st.error("Health Status: Critical")

    st.info(
        "Yeh score simulated sensor readings par based demo assessment hai. "
        "Yeh real vehicle inspection ka replacement nahi hai."
    )
    st.title("🚘 Complete Vehicle Digital Twin")
    st.caption(
        "Software-based sensor simulation. No physical vehicle or OBD-II device is connected."
    )

    sim_mode = st.selectbox(
        "🧪 Fault Simulation Mode",
        ["Normal Simulation", "Engine Overheating", "Low Battery", "Low Tyre Pressure", "High RPM"],
        key="digital_twin_sim_mode",
    )
    selected_part = st.selectbox(
        "Select Vehicle Component",
        ["Engine", "Battery", "Front-left tyre", "Front-right tyre",
         "Rear-left tyre", "Rear-right tyre", "Cabin"],
        key="digital_twin_component",
    )

    # Stable simulated values per selected mode; the user can switch modes to demonstrate faults.
    if sim_mode == "Engine Overheating":
        twin_temp = round(random.uniform(101, 115), 1)
        twin_rpm = round(random.uniform(1200, 3200), 0)
        twin_voltage = round(random.uniform(12.3, 14.2), 2)
        twin_speed = round(random.uniform(0, 80), 1)
        twin_tyre = round(random.uniform(31, 35), 1)
    elif sim_mode == "Low Battery":
        twin_temp = round(random.uniform(85, 95), 1)
        twin_rpm = round(random.uniform(800, 2500), 0)
        twin_voltage = round(random.uniform(10.5, 11.8), 2)
        twin_speed = round(random.uniform(0, 60), 1)
        twin_tyre = round(random.uniform(31, 35), 1)
    elif sim_mode == "Low Tyre Pressure":
        twin_temp = round(random.uniform(85, 95), 1)
        twin_rpm = round(random.uniform(800, 2500), 0)
        twin_voltage = round(random.uniform(12.3, 14.2), 2)
        twin_speed = round(random.uniform(0, 60), 1)
        twin_tyre = round(random.uniform(20, 27), 1)
    elif sim_mode == "High RPM":
        twin_temp = round(random.uniform(90, 105), 1)
        twin_rpm = round(random.uniform(4200, 6000), 0)
        twin_voltage = round(random.uniform(12.3, 14.2), 2)
        twin_speed = round(random.uniform(20, 100), 1)
        twin_tyre = round(random.uniform(31, 35), 1)
    else:
        twin_temp = round(random.uniform(85, 95), 1)
        twin_rpm = round(random.uniform(800, 3500), 0)
        twin_voltage = round(random.uniform(12.2, 14.5), 2)
        twin_speed = round(random.uniform(0, 120), 1)
        twin_tyre = round(random.uniform(30, 36), 1)

    twin_alerts = []
    if twin_temp >= 100:
        twin_alerts.append("Engine temperature high")
    if twin_voltage < 12:
        twin_alerts.append("Battery voltage low")
    if twin_rpm > 4000:
        twin_alerts.append("RPM unusually high")
    if twin_tyre < 28:
        twin_alerts.append("Tyre pressure low")

    twin_health = max(
        0,
        100
        - (30 if twin_temp >= 100 else 0)
        - (20 if twin_voltage < 12 else 0)
        - (15 if twin_rpm > 4000 else 0)
        - (15 if twin_tyre < 28 else 0),
    )
    twin_status = "CRITICAL" if twin_health <= 55 else ("WARNING" if twin_alerts else "NORMAL")

    st.subheader("📡 Live Sensor Dashboard (Simulated)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Engine Temperature", f"{twin_temp} °C")
    m2.metric("RPM", f"{int(twin_rpm)}")
    m3.metric("Battery Voltage", f"{twin_voltage} V")
    m4, m5, m6 = st.columns(3)
    m4.metric("Speed", f"{twin_speed} km/h")
    m5.metric("Tyre Pressure", f"{twin_tyre} PSI")
    m6.metric("Vehicle Health", f"{twin_health}/100")

    if twin_alerts:
        st.subheader("🚨 Automatic Fault Detection")
        for alert in twin_alerts:
            st.warning(alert)
    else:
        st.success("No configured demo fault threshold crossed.")

    st.metric("Vehicle Health Status", twin_status)
    if twin_status == "CRITICAL":
        st.error("Critical simulated readings. Real vehicle decisions require verified sensor data and a qualified mechanic.")
    elif twin_status == "WARNING":
        st.warning("One or more simulated readings need attention.")
    else:
        st.success("Readings are within this demonstration's configured thresholds.")

    # Persist every run to a local SQLite table for trend/history review.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO digital_twin_history
            (timestamp, simulation_mode, component, engine_temp, rpm,
             battery_voltage, speed, tyre_pressure, health_score, rule_alert, ml_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"), sim_mode, selected_part,
            twin_temp, twin_rpm, twin_voltage, twin_speed, twin_tyre,
            int(twin_health), " | ".join(twin_alerts) if twin_alerts else "No rule alert",
            "Pending advanced ML analysis",
        ))

    st.divider()
    st.subheader("🚘 3D Vehicle Model")
    glb_path = BASE_DIR / "vehicle.glb"
    if glb_path.exists():
        import base64
        glb_b64 = base64.b64encode(glb_path.read_bytes()).decode("ascii")
        model_html = f"""
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer src="data:model/gltf-binary;base64,{glb_b64}"
            camera-controls auto-rotate shadow-intensity="1"
            style="width:100%;height:480px;background:#eef4fb;border-radius:16px;"
            alt="Vehicle 3D model"></model-viewer>
        """
        st.components.v1.html(model_html, height=500, scrolling=False)
    else:
        st.info(
            "vehicle.glb file project folder mein nahi mili. Apni 3D model file ko "
            "E:\\VehicleGPT\\vehicle.glb naam se rakhein. Neeche conceptual 3D model dikhaya ja raha hai."
        )
        selected_part_positions = {
            "Engine": (1.25, 0.0, 0.72), "Battery": (0.55, -0.48, 0.72),
            "Front-left tyre": (1.05, -0.92, 0.35), "Front-right tyre": (1.05, 0.92, 0.35),
            "Rear-left tyre": (-1.15, -0.92, 0.35), "Rear-right tyre": (-1.15, 0.92, 0.35),
            "Cabin": (-0.05, 0.0, 1.22),
        }
        def add_vehicle_box(fig, x0, x1, y0, y1, z0, z1, color, name, opacity=1.0):
            vertices = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
            ]
            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                color=color, opacity=opacity, name=name,
            ))
        fig = go.Figure()
        add_vehicle_box(fig, -1.8, 1.8, -0.78, 0.78, 0.45, 0.95, "#2563eb", "Vehicle body")
        add_vehicle_box(fig, -0.85, 0.65, -0.62, 0.62, 0.95, 1.55, "#60a5fa", "Cabin", 0.88)
        wx, wy, wz = [], [], []
        for x in (-1.15, 1.05):
            for y in (-0.88, 0.88):
                wx.append(x); wy.append(y); wz.append(0.38)
        fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="markers",
                                   marker={"size": 18, "color": "#263238"}, name="Tyres"))
        px_, py_, pz_ = selected_part_positions[selected_part]
        fig.add_trace(go.Scatter3d(x=[px_], y=[py_], z=[pz_], mode="markers+text",
                                   marker={"size": 12, "color": "#ef4444"},
                                   text=[selected_part], textposition="top center", name="Selected part"))
        fig.update_layout(height=500, margin={"l": 0, "r": 0, "t": 20, "b": 0},
                          scene={"aspectmode": "data"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📈 Sensor History & Trend")
    with sqlite3.connect(DB_PATH) as conn:
        twin_history = pd.read_sql_query(
            "SELECT timestamp, simulation_mode, engine_temp, rpm, battery_voltage, speed, "
            "tyre_pressure, health_score FROM digital_twin_history ORDER BY id DESC LIMIT 100",
            conn,
        )
    if not twin_history.empty:
        twin_history = twin_history.iloc[::-1]
        st.line_chart(
            twin_history.set_index("timestamp")[["engine_temp", "battery_voltage", "tyre_pressure", "health_score"]],
            use_container_width=True,
        )
        st.dataframe(twin_history.tail(15), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Digital Twin History CSV",
            twin_history.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_digital_twin_history.csv",
            mime="text/csv",
        )

    st.info(
        "Note: all sensor values on this page are software-generated simulations. "
        "They are not live readings from a physical vehicle."
    )


# ------------------------- Footer -------------------------
st.divider()

st.caption(
    "VehicleGPT | AI-powered vehicle intelligence prototype "
    "| Simulated data only"
)
