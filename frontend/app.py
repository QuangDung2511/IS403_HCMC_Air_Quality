"""
HCMC Air Quality Dashboard — Realtime Backend
=============================================
Sources:
  - PM2.5 current: OpenAQ v3 API (sensor 11357424, US Consulate HCMC)
  - Weather current: Open-Meteo Forecast API (free, no key needed)
  - Model: XGBoost Tuned (models/xgb_tuned_final.joblib)
  
Fallback: If any API call fails, the last-known good values are reused.
Cache: API results cached for 15 minutes to avoid hammering external services.
"""

import os
import json
import time
import math
import joblib
import requests
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # Python 3.9+
from flask import Flask, jsonify, render_template

# Vietnam timezone (UTC+7)
TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")

def now_vn():
    """Return current datetime in Vietnam timezone."""
    return datetime.now(TZ_VN)

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'xgb_tuned_final.joblib')
META_PATH   = os.path.join(BASE_DIR, 'models', 'xgb_tuned_meta.json')
SEED_PATH   = os.path.join(BASE_DIR, 'data', 'seed', 'pm25_history_72h.csv')

# ─── OpenAQ API ──────────────────────────────────────────────────────────────
OPENAQ_API_KEY  = "eb64fc98f390093d2056ab1314c1ba89deb63537745bfbfc14eca45af029966d"
OPENAQ_SENSOR   = 11357424   # PM2.5 @ US Consulate HCMC
OPENAQ_HEADERS  = {"X-API-Key": OPENAQ_API_KEY}

# ─── Open-Meteo ───────────────────────────────────────────────────────────────
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CURRENT_PARAMS = {
    "latitude": 10.7626,
    "longitude": 106.6602,
    "current": [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m"
    ],
    "timezone": "Asia/Ho_Chi_Minh"
}

WEATHER_FORECAST_PARAMS = {
    "latitude": 10.7626,
    "longitude": 106.6602,
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m"
    ],
    "timezone": "Asia/Ho_Chi_Minh",
    "forecast_days": 2
}

# ─── Load Model ───────────────────────────────────────────────────────────────
print("Loading XGBoost model...")
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
FEATURES = meta['features']
print(f"✅ Model loaded — {len(FEATURES)} features. Test MAE: {meta['test_mae']} | Test RMSE: {meta['test_rmse']}")

# ─── PM2.5 History Buffer (ring buffer, 72h) ──────────────────────────────────
pm25_buffer   = deque(maxlen=720)   # 30 days of hourly data
precip_buffer = deque(maxlen=720)


def bootstrap_history_from_api():
    """
    Fetch the last 72 hours of PM2.5 readings from OpenAQ at startup.
    This gives the model real current-era lag features immediately.
    Falls back to the CSV seed file if the API returns too few distinct values.
    """
    print("Bootstrapping 72h PM2.5 history from OpenAQ...")
    try:
        since = (datetime.utcnow() - timedelta(hours=73)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"https://api.openaq.org/v3/sensors/{OPENAQ_SENSOR}/measurements"
        params = {"limit": 100, "datetime_from": since, "order_by": "datetime", "sort_direction": "asc"}
        resp = requests.get(url, headers=OPENAQ_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get('results', [])

        valid = [float(r['value']) for r in results if 0 < float(r.get('value', 0)) < 500]

        # Require at least 10 distinct readings so the history chart isn't flat
        if len(valid) >= 10 and len(set(round(v, 1) for v in valid)) >= 5:
            for val in valid[-72:]:
                pm25_buffer.append(val)
                precip_buffer.append(0.0)
            print(f"✅ OpenAQ bootstrap: {len(pm25_buffer)} hours loaded "
                  f"(PM2.5 range: {min(pm25_buffer):.1f}–{max(pm25_buffer):.1f} µg/m³)")
            return
        else:
            print(f"⚠️  OpenAQ returned only {len(valid)} valid readings "
                  f"({len(set(round(v,1) for v in valid))} unique) — falling back to seed CSV.")
    except Exception as e:
        print(f"⚠️  OpenAQ bootstrap failed ({e}) — falling back to seed CSV.")

    # Fallback: load from pre-built CSV seed (has 72 realistic varying values)
    try:
        seed_df = pd.read_csv(SEED_PATH)
        for v in seed_df['pm25'].values:
            pm25_buffer.append(float(v))
        for v in seed_df.get('precipitation', pd.Series([0.0]*len(seed_df))).values:
            precip_buffer.append(float(v))
        print(f"✅ Seed CSV fallback: {len(pm25_buffer)} hours loaded "
              f"(PM2.5 range: {pm25_buffer[0]:.1f}–{max(pm25_buffer):.1f} µg/m³)")
    except Exception as e:
        # Last resort: fill with a reasonable default
        print(f"⚠️  Seed CSV also failed ({e}) — filling with default 30 µg/m³.")
        for _ in range(72):
            pm25_buffer.append(30.0)
            precip_buffer.append(0.0)

bootstrap_history_from_api()


# ─── In-memory cache ──────────────────────────────────────────────────────────
_cache = {}
CACHE_TTL = 15 * 60  # 15 minutes

def cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry['ts']) < CACHE_TTL:
        return entry['data']
    return None

def cache_set(key, data):
    _cache[key] = {'data': data, 'ts': time.time()}

# ─── Last known good values (fallback) ────────────────────────────────────────
_last_good = {
    'pm25': 35.0,
    'weather': {
        'temperature_2m': 28.0,
        'relative_humidity_2m': 75.0,
        'precipitation': 0.0,
        'surface_pressure': 1010.0,
        'wind_speed_10m': 5.0,
        'wind_direction_10m': 90.0,
    }
}

# ─── API Fetchers ──────────────────────────────────────────────────────────────

def fetch_current_pm25() -> float:
    """Fetch most recent PM2.5 from OpenAQ sensor."""
    cached = cache_get('pm25')
    if cached is not None:
        return cached

    try:
        now = datetime.utcnow()
        since = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"https://api.openaq.org/v3/sensors/{OPENAQ_SENSOR}/measurements"
        params = {"limit": 5, "datetime_from": since}
        resp = requests.get(url, headers=OPENAQ_HEADERS, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get('results', [])
        if results:
            val = float(results[0]['value'])
            # Sanity check
            if 0 < val < 500:
                _last_good['pm25'] = val
                cache_set('pm25', val)
                print(f"[OpenAQ] PM2.5 = {val:.1f} µg/m³")
                return val
    except Exception as e:
        print(f"[OpenAQ] Fetch failed: {e} — using fallback {_last_good['pm25']:.1f}")
    
    return _last_good['pm25']


def fetch_current_weather() -> dict:
    """Fetch current weather from Open-Meteo using the 'current' endpoint."""
    cached = cache_get('weather')
    if cached is not None:
        return cached

    try:
        resp = requests.get(WEATHER_URL, params=WEATHER_CURRENT_PARAMS, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get('current', {})

        weather = {
            'temperature_2m':        cur.get('temperature_2m', 28.0),
            'relative_humidity_2m':  cur.get('relative_humidity_2m', 75.0),
            'precipitation':         cur.get('precipitation', 0.0),
            'surface_pressure':      cur.get('surface_pressure', 1010.0),
            'wind_speed_10m':        cur.get('wind_speed_10m', 5.0),
            'wind_direction_10m':    cur.get('wind_direction_10m', 90.0),
        }
        _last_good['weather'] = weather
        cache_set('weather', weather)
        print(f"[Open-Meteo] Temp={weather['temperature_2m']}°C, Hum={weather['relative_humidity_2m']}%, Wind={weather['wind_speed_10m']}km/h")
        return weather
    except Exception as e:
        print(f"[Open-Meteo] Fetch failed: {e} — using fallback")
        return _last_good['weather']


def fetch_forecast_weather(hours=24) -> list:
    """Fetch next N hours weather forecast."""
    cached = cache_get('weather_forecast')
    if cached is not None:
        return cached[:hours]

    try:
        resp = requests.get(WEATHER_URL, params=WEATHER_FORECAST_PARAMS, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get('hourly', {})
        times  = hourly.get('time', [])

        # Find index of the NEXT hour from now
        now_str = now_vn().strftime("%Y-%m-%dT%H:00")
        start_idx = 0
        for i, t in enumerate(times):
            if t.startswith(now_str):
                start_idx = i + 1  # start from next hour
                break

        forecast = []
        for i in range(start_idx, min(start_idx + hours, len(times))):
            forecast.append({
                'time':                  times[i],
                'temperature_2m':        hourly['temperature_2m'][i],
                'relative_humidity_2m':  hourly['relative_humidity_2m'][i],
                'precipitation':         hourly['precipitation'][i],
                'surface_pressure':      hourly['surface_pressure'][i],
                'wind_speed_10m':        hourly['wind_speed_10m'][i],
                'wind_direction_10m':    hourly['wind_direction_10m'][i],
            })

        cache_set('weather_forecast', forecast)
        return forecast[:hours]
    except Exception as e:
        print(f"[Open-Meteo Forecast] Fetch failed: {e}")
        return [_last_good['weather']] * hours


# ─── Feature Engineering ──────────────────────────────────────────────────────

def make_features(pm25_now: float, weather: dict, pm25_hist: list) -> pd.DataFrame:
    """Build the 26-feature vector for model prediction."""
    now = now_vn()
    hist = list(pm25_hist)  # oldest → newest, length up to 72

    def lag(n):
        # lag_n means n hours ago; hist[-1] = most recent
        idx = -(n)
        if len(hist) >= n:
            return hist[idx]
        return pm25_now  # fallback

    def rolling_mean(n):
        window = hist[-n:] if len(hist) >= n else hist
        return float(np.mean(window)) if window else pm25_now

    def rolling_std(n):
        window = hist[-n:] if len(hist) >= n else hist
        return float(np.std(window)) if len(window) > 1 else 0.0

    # Wind components
    wind_dir_rad = math.radians(weather.get('wind_direction_10m', 90))
    wind_spd = weather.get('wind_speed_10m', 0)
    wind_u = -wind_spd * math.sin(wind_dir_rad)
    wind_v = -wind_spd * math.cos(wind_dir_rad)

    # Hours since last rain (approximate from precip buffer)
    precip_hist = list(precip_buffer)
    hours_since_rain = 0
    for prec in reversed(precip_hist):
        if prec > 0.1:
            break
        hours_since_rain += 1

    month = now.month
    hour  = now.hour
    is_dry_season = 1 if month in [12, 1, 2, 3, 4] else 0

    row = {
        'month':                now.month,
        'month_cos':            math.cos(2 * math.pi * now.month / 12),
        'pm25':                 pm25_now,
        'pm25_lag_1':           lag(1),
        'hours_since_last_rain': hours_since_rain,
        'pm25_roll_72h_mean':   rolling_mean(72),
        'pm25_roll_6h_mean':    rolling_mean(6),
        'pm25_lag_24':          lag(24),
        'pm25_lag_48':          lag(48),
        'pm25_lag_72':          lag(72),
        'pm25_roll_12h_std':    rolling_std(12),
        'wind_u':               wind_u,
        'pm25_roll_72h_std':    rolling_std(72),
        'surface_pressure':     weather.get('surface_pressure', 1010),
        'wind_v':               wind_v,
        'pm25_roll_48h_std':    rolling_std(48),
        'pm25_roll_24h_mean':   rolling_mean(24),
        'pm25_roll_48h_mean':   rolling_mean(48),
        'day_of_week':          now.weekday(),
        'hour_cos':             math.cos(2 * math.pi * hour / 24),
        'is_dry_season':        is_dry_season,
        'temperature_2m':       weather.get('temperature_2m', 28),
        'relative_humidity_2m': weather.get('relative_humidity_2m', 75),
        'pm25_roll_24h_std':    rolling_std(24),
        'pm25_lag_12':          lag(12),
        'pm25_lag_2':           lag(2),
    }

    # Ensure correct feature order
    df = pd.DataFrame([row])[FEATURES]
    return df


# ─── AQI Helpers ──────────────────────────────────────────────────────────────

def get_aqi_category(pm25):
    if pm25 <= 12:    return {"level": "Good",                       "color": "#00e400", "aqi": int(pm25 * (50/12))}
    elif pm25 <= 35.4: return {"level": "Moderate",                  "color": "#ffff00", "aqi": int(51  + (pm25-12.1)  * (49/23.3))}
    elif pm25 <= 55.4: return {"level": "Unhealthy for Sensitive",    "color": "#ff7e00", "aqi": int(101 + (pm25-35.5)  * (49/19.9))}
    elif pm25 <= 150.4:return {"level": "Unhealthy",                  "color": "#ff0000", "aqi": int(151 + (pm25-55.5)  * (49/94.9))}
    elif pm25 <= 250.4:return {"level": "Very Unhealthy",             "color": "#8f3f97", "aqi": int(201 + (pm25-150.5) * (99/99.9))}
    else:              return {"level": "Hazardous",                  "color": "#7e0023", "aqi": int(301 + (pm25-250.5) * (199/249.9))}

def calculate_risk(pred_pm25):
    if pred_pm25 <= 12:   return 5,  "Low risk"
    elif pred_pm25 <= 35.4: return 20, "Moderate risk"
    elif pred_pm25 <= 55.4: return 45, "High risk for sensitive people"
    elif pred_pm25 <= 150.4:return 75, "High health risk"
    else:                   return 95, "Extreme pollution risk"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/current')
def get_current():
    # 1. Fetch realtime data
    pm25_now = fetch_current_pm25()
    weather  = fetch_current_weather()

    # 2. Update PM2.5 ring buffer with new reading
    pm25_buffer.append(pm25_now)
    precip_buffer.append(weather.get('precipitation', 0))

    # 3. Build features & predict
    X = make_features(pm25_now, weather, list(pm25_buffer))
    pred_24h = float(model.predict(X)[0])
    pred_24h = max(0, pred_24h)  # clamp negatives

    # 4. Compose response
    aqi_info = get_aqi_category(pm25_now)
    risk_pct, risk_desc = calculate_risk(pred_24h)

    return jsonify({
        "timestamp":   now_vn().strftime("%d/%m/%Y %H:%M"),
        "pm25":        round(pm25_now, 1),
        "aqi":         aqi_info["aqi"],
        "level":       aqi_info["level"],
        "color":       aqi_info["color"],
        "weather": {
            "temperature": round(weather['temperature_2m'], 1),
            "humidity":    round(weather['relative_humidity_2m'], 1),
            "wind_speed":  round(weather['wind_speed_10m'], 1),
            "pressure":    round(weather['surface_pressure'], 0),
        },
        "forecast_24h": round(pred_24h, 1),
        "risk": {
            "percent":     risk_pct,
            "description": risk_desc
        }
    })


@app.route('/api/forecast')
def get_forecast():
    """24-hour PM2.5 forecast using XGBoost iteratively with Open-Meteo weather."""
    weather_fc = fetch_forecast_weather(hours=24)

    hist = list(pm25_buffer)
    pm25_last = hist[-1] if hist else _last_good['pm25']

    labels, values = [], []
    iterative_buffer = list(hist)  # copy to iterate without mutating ring buffer

    now = now_vn()
    for i, wfc in enumerate(weather_fc):
        hour_label = (now + timedelta(hours=i+1)).strftime("%H:00")
        X = make_features(pm25_last, wfc, iterative_buffer)
        pred = float(model.predict(X)[0])
        pred = max(0, pred)

        labels.append(hour_label)
        values.append(round(pred, 1))

        # Feed prediction back as lag for next step
        iterative_buffer.append(pred)
        if len(iterative_buffer) > 72:
            iterative_buffer.pop(0)
        pm25_last = pred

    return jsonify({"labels": labels, "values": values})


@app.route('/api/history')
def get_history():
    """Last N hours: actual from buffer vs single-step predictions."""
    hist = list(pm25_buffer)
    n = min(48, len(hist))
    if n < 2:
        return jsonify({"labels": [], "actual": [], "predicted": []})

    now = now_vn()
    labels   = [(now - timedelta(hours=n-i)).strftime("%m-%d %H:00") for i in range(n)]
    actuals  = [round(v, 1) for v in hist[-n:]]
    # Generate single-step predictions for each historic point
    preds = []
    for j in range(n):
        sub_hist = hist[:len(hist)-n+j+1]
        X = make_features(sub_hist[-1], _last_good['weather'], sub_hist)
        p = float(model.predict(X)[0])
        preds.append(round(max(0, p), 1))

    return jsonify({"labels": labels, "actual": actuals, "predicted": preds})


@app.route('/api/model-info')
def get_model_info():
    return jsonify({
        "model_name": "XGBoost (Tuned — Realtime)",
        "metrics": {
            "rmse": meta.get('test_rmse', 17.73),
            "mae":  meta.get('test_mae',  12.70),
            "mape": meta.get('test_mape', 36.35)
        },
        "features": f"{len(FEATURES)} selected features (Lag, Rolling, Weather)"
    })


@app.route('/api/advance')
def advance_time():
    """Kept for backward compatibility; now a no-op since we use realtime data."""
    return jsonify({"status": "realtime", "message": "Data is now live from OpenAQ!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
