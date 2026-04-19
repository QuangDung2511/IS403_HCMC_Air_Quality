import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from datetime import datetime, timedelta

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_PATH = os.path.join(BASE_DIR, 'outputs', 'predictions', 'tuned_tree_preds.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test_split.csv')

# Load data into memory
print("Loading model predictions and data...")
with open(PREDS_PATH, 'rb') as f:
    preds = pickle.load(f)

# Best model according to metrics
XGB_TEST_PREDS = preds['XGB_Tuned']['test']

# Load test data to get features and timestamps
test_df = pd.read_csv(TEST_DATA_PATH)
test_df['datetime_local'] = pd.to_datetime(test_df['datetime_local'])

# Current index to simulate real-time progression
# In a real app this would be driven by current time. Here we arbitrarily pick a start
# index and let the frontend request consecutive blocks.
# Let's say we start at index 100
current_sim_idx = 100

def get_aqi_category(pm25):
    if pm25 <= 12: return {"level": "Good", "color": "#00e400", "aqi_sim": int(pm25 * (50/12))}
    elif pm25 <= 35.4: return {"level": "Moderate", "color": "#ffff00", "aqi_sim": int(51 + (pm25-12.1)*(49/23.3))}
    elif pm25 <= 55.4: return {"level": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "aqi_sim": int(101 + (pm25-35.5)*(49/19.9))}
    elif pm25 <= 150.4: return {"level": "Unhealthy", "color": "#ff0000", "aqi_sim": int(151 + (pm25-55.5)*(49/94.9))}
    elif pm25 <= 250.4: return {"level": "Very Unhealthy", "color": "#8f3f97", "aqi_sim": int(201 + (pm25-150.5)*(99/99.9))}
    else: return {"level": "Hazardous", "color": "#7e0023", "aqi_sim": int(301 + (pm25-250.5)*(199/249.9))}

def calculate_risk(pred_pm25):
    if pred_pm25 <= 12: return 5, "Low risk"
    elif pred_pm25 <= 35.4: return 20, "Moderate risk"
    elif pred_pm25 <= 55.4: return 45, "High risk for sensitive people"
    elif pred_pm25 <= 150.4: return 75, "High health risk"
    else: return 95, "Extreme pollution risk"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/current')
def get_current():
    global current_sim_idx
    # Wrap around if we hit the end (minus 24 config)
    if current_sim_idx >= len(test_df) - 24:
        current_sim_idx = 0
    
    current_data = test_df.iloc[current_sim_idx]
    
    # We use actual PM2.5 for current state, since this is "now"
    current_pm25 = float(current_data['pm25'])
    aqi_info = get_aqi_category(current_pm25)
    
    # Target prediction 24h ahead
    pred_24h = float(XGB_TEST_PREDS[current_sim_idx])
    risk_pct, risk_desc = calculate_risk(pred_24h)
    
    now = datetime.now()
    
    return jsonify({
        "timestamp": now.strftime("%d/%m/%Y %H:%M"),
        "pm25": round(current_pm25, 1),
        "aqi": aqi_info["aqi_sim"],
        "level": aqi_info["level"],
        "color": aqi_info["color"],
        "weather": {
            "temperature": round(float(current_data['temperature_2m']), 1),
            "humidity": round(float(current_data['relative_humidity_2m']), 1),
            "wind_speed": round(float(current_data['wind_speed_10m']), 1),
            "pressure": round(float(current_data['surface_pressure']), 0)
        },
        "forecast_24h": round(pred_24h, 1),
        "risk": {
            "percent": risk_pct,
            "description": risk_desc
        }
    })

@app.route('/api/forecast')
def get_forecast():
    """Returns 24h forecast curve. We use the next 24 predictions."""
    global current_sim_idx
    
    if current_sim_idx >= len(test_df) - 24:
        return jsonify({"error": "End of data"}), 400
        
    start_idx = current_sim_idx
    end_idx = start_idx + 24
    
    now = datetime.now()
    times = [(now + timedelta(hours=i)).strftime("%H:00") for i in range(1, 25)]
    # In a real scenario these would be the 1h, 2h ... 24h predictions made AT current_sim_idx.
    # Since our model is purely a 24h horizon model, to simulate a curve we'll just show the upcoming
    # predictions for the next 24 hours as if they are the forecast timeline.
    preds = XGB_TEST_PREDS[start_idx:end_idx].tolist()
    
    return jsonify({
        "labels": times,
        "values": [round(v, 1) for v in preds]
    })

@app.route('/api/history')
def get_history():
    """Returns past 48h actual vs predicted for comparison chart"""
    global current_sim_idx
    if current_sim_idx < 48:
        start_idx = 0
    else:
        start_idx = current_sim_idx - 48
        
    end_idx = current_sim_idx
    
    now = datetime.now()
    num_history = end_idx - start_idx
    times = [(now - timedelta(hours=num_history - i)).strftime("%m-%d %H:00") for i in range(num_history)]
    
    actuals = test_df['target_pm25_h24'].iloc[start_idx:end_idx].tolist()
    preds = XGB_TEST_PREDS[start_idx:end_idx].tolist()
    
    return jsonify({
        "labels": times,
        "actual": [round(v, 1) for v in actuals],
        "predicted": [round(v, 1) for v in preds]
    })

@app.route('/api/model-info')
def get_model_info():
    return jsonify({
        "model_name": "XGBoost (Tuned)",
        "metrics": {
            "rmse": 17.73,
            "mae": 12.70,
            "mape": 36.35
        },
        "features": "25 selected features (Lag, Rolling Std/Mean, Weather)"
    })

@app.route('/api/advance')
def advance_time():
    """API to manually or automatically advance the simulation time"""
    global current_sim_idx
    current_sim_idx += 1
    return jsonify({"status": "success", "new_idx": current_sim_idx})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
