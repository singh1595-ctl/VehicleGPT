
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def predict_maintenance(pm, temp, voltage, speed):
    data = pd.DataFrame({
        "pm": [pm],
        "temp": [temp],
        "voltage": [voltage],
        "speed": [speed]
    })

    # Demo training data
    X = pd.DataFrame({
        "pm": [20, 30, 40, 60, 80, 90, 25, 70, 85, 35, 55, 95],
        "temp": [75, 80, 85, 100, 110, 115, 78, 105, 112, 82, 95, 120],
        "voltage": [13.8, 13.7, 13.5, 12.2, 11.8, 11.5, 13.9, 12.0, 11.7, 13.6, 12.5, 11.2],
        "speed": [40, 50, 60, 80, 90, 100, 30, 75, 95, 45, 70, 110]
    })

    # 0 = Normal, 1 = Maintenance Alert
    y = [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "status": "Maintenance Alert" if prediction == 1 else "Normal",
        "risk": round(probability * 100, 2)
    }
