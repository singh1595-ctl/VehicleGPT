
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Sample vehicle sensor data
data = pd.DataFrame({
    "engine_temp": [
        82, 84, 85, 83, 86, 87, 84, 85, 82, 86,
        83, 85, 84, 87, 86, 82, 85, 83, 84, 86
    ],
    "battery_voltage": [
        12.5, 12.6, 12.4, 12.5, 12.7,
        12.6, 12.5, 12.4, 12.6, 12.5,
        12.7, 12.5, 12.6, 12.4, 12.5,
        12.6, 12.5, 12.7, 12.4, 12.6
    ],
    "speed": [
        30, 35, 40, 32, 45, 38, 42, 36, 33, 41,
        39, 34, 37, 43, 31, 40, 35, 38, 42, 36
    ]
})

# Train anomaly detection model
model = IsolationForest(
    contamination=0.10,
    random_state=42
)

model.fit(data)

# Save model
joblib.dump(model, "vehicle_anomaly_model.pkl")

# Test an unusual sample reading
test_reading = pd.DataFrame([{
    "engine_temp": 115,
    "battery_voltage": 10.5,
    "speed": 40
}])

prediction = model.predict(test_reading)

print("VehicleGPT ML Model Training Complete!")
print("Test reading:", test_reading.iloc[0].to_dict())

if prediction[0] == -1:
    print("Result: UNUSUAL SENSOR PATTERN DETECTED")
else:
    print("Result: Pattern not flagged by the model")

print("Model saved as vehicle_anomaly_model.pkl")
