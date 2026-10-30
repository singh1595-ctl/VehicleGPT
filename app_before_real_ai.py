
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from openai import OpenAI
st.set_page_config(
    page_title="VehicleGPT",
    page_icon="🚗",
    layout="wide"
)

st.title("VehicleGPT")
st.subheader("AI-Powered Vehicle Intelligence System")
st.write(
    "Vehicle sensor data monitor karne aur possible "
    "problems identify karne ka smart dashboard."
)

st.info("Demo mode: neeche dikhaya gaya data sample hai, real vehicle data nahi.")

data = pd.DataFrame({
    "Time": ["10:00", "10:05", "10:10", "10:15", "10:20"],
    "Engine Temperature (°C)": [85, 88, 90, 96, 94],
    "Battery Voltage (V)": [12.6, 12.5, 12.4, 12.1, 12.3],
    "Speed (km/h)": [30, 45, 50, 40, 35]
})

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Latest Engine Temperature", "94 °C")

with col2:
    st.metric("Latest Battery Voltage", "12.3 V")

with col3:
    st.metric("Latest Speed", "35 km/h")

st.divider()

st.header("Vehicle Sensor Readings")

sensor = st.selectbox(
    "Graph ke liye sensor select karein",
    [
        "Engine Temperature (°C)",
        "Battery Voltage (V)",
        "Speed (km/h)"
    ]
)

fig = px.line(
    data,
    x="Time",
    y=sensor,
    markers=True,
    title=sensor + " Over Time"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Sample Vehicle Data")
st.dataframe(data, use_container_width=True)

st.caption("VehicleGPT | AI-powered vehicle intelligence prototype")

st.divider()
st.header("🤖 ML-Based Anomaly Detection")

st.write(
    "Ye demo sensor readings ko simple thresholds se check karta hai. "
    "Ye abhi trained ML model nahi hai."
)

latest_temp = float(data["Engine Temperature (°C)"].iloc[-1])
latest_voltage = float(data["Battery Voltage (V)"].iloc[-1])

if latest_temp > 95:
    st.warning(
        f"Temperature warning: latest reading {latest_temp} °C hai. "
        "Vehicle manual ke mutabik reading verify karein."
    )
else:
    st.success(
        f"Temperature reading {latest_temp} °C hai. "
        "Is demo ke temperature threshold se neeche hai."
    )

if latest_voltage < 12.2:
    st.warning(
        f"Battery voltage warning: {latest_voltage} V. "
        "Suitable conditions mein voltage ko dobara check karein."
    )
else:
    st.success(
        f"Battery voltage {latest_voltage} V hai. "
        "Ye demo threshold ke andar hai."
    )

st.divider()
st.header("🧠 Trained ML Model Result")

try:
    model = joblib.load("vehicle_anomaly_model.pkl")

    current_reading = pd.DataFrame([{
        "engine_temp": latest_temp,
        "battery_voltage": latest_voltage,
        "speed": float(data["Speed (km/h)"].iloc[-1])
    }])

    result = model.predict(current_reading)[0]

    if result == -1:
        st.warning(
            "ML model ne is sensor pattern ko unusual flag kiya hai. "
            "Yeh confirmed vehicle fault diagnosis nahi hai."
        )
    else:
        st.success(
            "ML model ne is sensor pattern ko unusual flag nahi kiya. "
            "Iska matlab vehicle fault-free hona confirm nahi hota."
        )

    st.write("Model ko di gayi current sample readings:")
    st.dataframe(current_reading, use_container_width=True)

except FileNotFoundError:
    st.error(
        "ML model file nahi mili. Pehle terminal mein "
        "'python train_model.py' run karein."
    )

st.divider()
st.header("🔍 Analyze Vehicle Readings")

st.write("Demo ke liye sensor readings enter karein.")

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
    user_reading = pd.DataFrame([{
        "engine_temp": input_temp,
        "battery_voltage": input_voltage,
        "speed": input_speed
    }])

    try:
        model = joblib.load("vehicle_anomaly_model.pkl")
        prediction = model.predict(user_reading)[0]

        st.subheader("Analysis Result")

        if prediction == -1:
            st.warning(
                "ML model ne readings ke combination ko unusual flag kiya. "
                "Ye confirmed mechanical fault nahi hai."
            )
        else:
            st.success(
                "ML model ne readings ko unusual flag nahi kiya. "
                "Isse vehicle ka fault-free hona prove nahi hota."
            )

        st.write("Aapki entered readings:")
        st.dataframe(user_reading, use_container_width=True)

    except FileNotFoundError:
        st.error("Model file nahi mili. Pehle python train_model.py run karein.")



# ==============================
# AI VEHICLE ASSISTANT - LOCAL
# ==============================

st.divider()
st.header("🤖 AI Vehicle Assistant")

st.write(
    "Vehicle ke engine, battery, tyre, brake, oil aur maintenance "
    "ke basic sawaal Hindi ya English mein poochhein."
)

st.info(
    "Ye local rule-based demo hai. Internet ya OpenAI API key "
    "ki zaroorat nahi hai."
)

if "vehicle_chat_history" not in st.session_state:
    st.session_state.vehicle_chat_history = []

def get_local_vehicle_answer(question):
    q = question.lower().strip()

    if q in ["hello", "hi", "namaste", "namaskar", "hey"]:
        return (
            "Namaste! Main VehicleGPT ka local vehicle assistant hoon. "
            "Aap battery, engine temperature, tyre, brake, oil ya "
            "maintenance ke baare mein poochh sakte hain."
        )

    if any(word in q for word in [
        "overheat", "overheating", "temperature",
        "engine hot", "engine garam", "heat"
    ]):
        return (
            "Engine overheating ho to safe jagah par ruk kar engine "
            "band karein aur thanda hone dein. Garam radiator cap "
            "na kholein. Problem jaari rahe to roadside assistance "
            "ya qualified mechanic se sampark karein."
        )

    if any(word in q for word in [
        "battery", "voltage", "start nahi", "starting problem"
    ]):
        return (
            "Battery ki reading battery type aur condition par depend "
            "karti hai. Engine band hone par charged 12V battery aam "
            "taur par lagbhag 12.6V dikha sakti hai. Low reading repeat "
            "ho to battery aur charging system check karayein."
        )

    if any(word in q for word in [
        "tyre", "tire", "puncture", "pressure"
    ]):
        return (
            "Tyre pressure vehicle ke door sticker ya owner's manual "
            "ke mutabik check karein, behtar hai tyres thande hon. "
            "Tyre par cut ya bulge ho to professional inspection karayein."
        )

    if any(word in q for word in [
        "brake", "break fail"
    ]):
        return (
            "Brake kamzor lag rahe hon ya brake warning light jal rahi "
            "ho to gaadi chalana band karein aur professional help lein."
        )

    if any(word in q for word in [
        "engine oil", "oil", "lubricant"
    ]):
        return (
            "Engine oil ko owner's manual ke tareeke se check karein. "
            "Oil pressure warning light jalne par safe jagah ruk kar "
            "manual aur mechanic ki salah follow karein."
        )

    if any(word in q for word in [
        "warning light", "dashboard light", "check engine", "indicator"
    ]):
        return (
            "Warning light ka meaning vehicle ke model aur symbol par "
            "depend karta hai. Owner's manual dekhein. Red warning light "
            "ya flashing engine light par safe jagah ruk kar help lein."
        )

    if any(word in q for word in [
        "maintenance", "service", "servicing"
    ]):
        return (
            "Regular maintenance mein owner's manual ke schedule ke "
            "mutabik engine oil, filters, tyres, brakes, battery aur "
            "coolant check karwana shamil hai."
        )

    return (
        "Is sawaal ka jawab mere local knowledge mein abhi nahi hai. "
        "Aap battery, engine temperature, tyre, brake, oil ya "
        "maintenance ke baare mein poochh sakte hain."
    )

for message in st.session_state.vehicle_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input(
    "Jaise: Car ki battery low ho to kya karein?"
)

if user_question:
    st.session_state.vehicle_chat_history.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    answer = get_local_vehicle_answer(user_question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.vehicle_chat_history.append({
        "role": "assistant",
        "content": answer
    })

if st.button("Clear Chat"):
    st.session_state.vehicle_chat_history = []
    st.rerun()

