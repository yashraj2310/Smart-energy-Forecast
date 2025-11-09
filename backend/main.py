# backend/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
import datetime as dt
import os
import asyncio
import random
from fastapi import FastAPI
from pydantic import BaseModel
# -------------------------------
# 1️⃣ FastAPI App Configuration
# -------------------------------
app = FastAPI(title="Smart Energy Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2️⃣ Model Loading
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_forecast_model.pkl")
model = joblib.load(MODEL_PATH)

# -------------------------------
# 3️⃣ Dataset Loading and Cleaning
# -------------------------------
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "household_power_consumption.txt")
df = pd.read_csv(DATA_PATH, sep=';', low_memory=False)

df['Datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

df = df[['Datetime', 'Global_active_power']].dropna()
df.set_index('Datetime', inplace=True)
df = df.resample('h').mean()

# -------------------------------
# 4️⃣ API Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "Smart Energy App Backend is Running ⚡"}


@app.get("/api/latest")
def get_latest_data():
    last_24h = df.tail(24)
    return {
        "timestamps": last_24h.index.strftime("%Y-%m-%d %H:%M").tolist(),
        "power": last_24h['Global_active_power'].tolist(),
    }


@app.get("/api/forecast")
def forecast_next_24h():
    last_24 = df.tail(24)

    # Build features
    X = pd.DataFrame({
        "hour": last_24.index.hour,
        "day": last_24.index.day,
        "weekday": last_24.index.weekday,
        "month": last_24.index.month,
    })

    # Add lag features safely
    for i in range(1, 25):
        X[f"lag_{i}"] = last_24['Global_active_power'].shift(i).bfill().values

    y_pred = model.predict(X)
    future_times = [df.index[-1] + dt.timedelta(hours=i + 1) for i in range(24)]

    return {
        "timestamps": [str(t) for t in future_times],
        "forecast": y_pred.tolist(),
    }


@app.get("/api/history")
def get_weekly_trend():
    last_week = df.tail(24 * 7)
    daily_avg = last_week.resample("D").mean()

    return {
        "dates": daily_avg.index.strftime("%Y-%m-%d").tolist(),
        "avg_power": daily_avg["Global_active_power"].round(3).tolist(),
    }


@app.get("/api/cost")
def calculate_energy_cost():
    last_24h = df.tail(24).copy()

    def get_tariff(hour):
        if 18 <= hour <= 22:
            return 8.0
        elif 8 <= hour <= 17:
            return 6.0
        else:
            return 3.5

    last_24h.loc[:, "tariff"] = last_24h.index.hour.map(get_tariff)
    last_24h.loc[:, "cost"] = last_24h["Global_active_power"] * last_24h["tariff"]

    total_cost = round(last_24h["cost"].sum(), 2)
    avg_tariff = round(last_24h["tariff"].mean(), 2)
    peak_hours = last_24h[last_24h["tariff"] == 8.0].index.strftime("%H:%M").tolist()

    return {
        "timestamps": last_24h.index.strftime("%Y-%m-%d %H:%M").tolist(),
        "power": last_24h["Global_active_power"].round(3).tolist(),
        "tariff": last_24h["tariff"].tolist(),
        "cost": last_24h["cost"].round(2).tolist(),
        "summary": {
            "total_cost": total_cost,
            "avg_tariff": avg_tariff,
            "peak_hours": peak_hours,
        },
    }


# -------------------------------
# 5️⃣ WebSocket Live Updates
# -------------------------------
@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            random_alerts = [
                "⚡ High load detected — consider reducing A/C usage.",
                "💰 Off-peak hours starting soon — shift heavy appliances after 10 PM.",
                "🌱 You’re consuming 12% less than yesterday — great job!",
                "🔥 Peak pricing now active (₹8/kWh) till 10 PM.",
            ]
            message = random.choice(random_alerts)
            await websocket.send_json({"alert": message})
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")


# -------------------------------
# 6️⃣ AI Suggestion Endpoint
# -------------------------------
@app.get("/api/suggestion")
def get_ai_suggestion():
    tips = [
        "Run the washing machine after 10 PM to save ₹18/day (Off-peak rate).",
        "Switch to LED lights to cut 5% of your energy bill monthly.",
        "Keep your fridge at 4°C for efficiency and food safety.",
        "Avoid charging EVs during 6-10 PM — peak tariff ₹8/kWh.",
        "Use fans instead of A/C during 23-06 h to cut CO₂ emissions by 0.4 kg/day.",
    ]
    return {"suggestion": random.choice(tips)}
@app.get("/api/emission")
def get_co2_emission():
    """
    Estimate CO₂ emissions (kg) based on energy usage.
    Uses India's avg grid factor ≈ 0.82 kg CO₂/kWh.
    """
    emission_factor = 0.82
    last_7_days = df.tail(24 * 7).copy()

    # daily energy (kWh) and emission
    daily_energy = last_7_days["Global_active_power"].resample("D").sum().fillna(0)
    daily_emission = daily_energy * emission_factor

    # 7-day baseline = avg emission
    baseline = daily_emission.mean()
    savings = (baseline - daily_emission).clip(lower=0)

    total_emission = round(daily_emission.sum(), 2)
    total_savings = round(savings.sum(), 2)

    return {
        "dates": daily_emission.index.strftime("%Y-%m-%d").tolist(),
        "emission": daily_emission.round(2).tolist(),
        "savings": savings.round(2).tolist(),
        "summary": {
            "total_emission": total_emission,
            "total_savings": total_savings,
            "baseline": round(baseline, 2),
        },
    }
# ===============================================
# ✅ Fixed: Demand Forecast Endpoint (Final Version)
# ===============================================
@app.get("/api/demand_forecast")
def demand_forecast():
    """Return next-hour forecast and dispatch recommendations."""

    # ✅ Resolve paths
    local_path = os.path.join(BASE_DIR, "..", "dataset", "unified_energy_full.csv")
    kaggle_path = "/kaggle/working/unified_energy_full.csv"

    if os.path.exists(local_path):
        data_path = local_path
    elif os.path.exists(kaggle_path):
        data_path = kaggle_path
    else:
        raise FileNotFoundError(
            f"Dataset not found in either {local_path} or {kaggle_path}. "
            "Please place unified_energy_full.csv in your /dataset folder."
        )

    # ✅ Suppress mixed-type warnings and enforce numeric coercion
    df = pd.read_csv(data_path, parse_dates=["Datetime"], low_memory=False)
    df = df.set_index("Datetime").sort_index()

    # Clean up possible mixed-type columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass

    # ✅ Load model safely (multi-output model)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_multi_model.pkl")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please train and save xgb_multi_model.pkl in backend/models/")
    xgb_demand_model = joblib.load(MODEL_PATH)

    # ✅ Feature engineering (must match training)
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"gen_lag_{lag}"] = df["Total_Gen_MW"].shift(lag).bfill()
        df[f"demand_lag_{lag}"] = df["Total_Demand_MW"].shift(lag).bfill()
        df[f"price_lag_{lag}"] = df["Price"].shift(lag).bfill()

    latest_row = df.iloc[-1:].copy()

    # Ensure only known model features are used
    for col in ["Name_of_State/UT", "wind"]:
        if col in latest_row.columns:
            latest_row[col] = latest_row[col].astype("category")

    # ✅ Predict demand and generation
    preds = xgb_demand_model.predict(latest_row)
    if isinstance(preds, np.ndarray) and preds.shape[1] == 2:
        predicted_gen = preds[0][0]
        predicted_price = preds[0][1]
    else:
        predicted_gen = preds[0]
        predicted_price = np.nan

    current_gen = latest_row["Total_Gen_MW"].values[0]
    storage = latest_row.get("Storage_Level", np.nan).values[0]

    # ✅ Recommendations
    if predicted_gen < current_gen:
        recommendation = {
            "status": "✅ Supply exceeds demand",
            "actions": [
                "Charge battery storage or reduce turbine load.",
                "Offer surplus energy at lower IEX rate.",
            ],
        }
    else:
        recommendation = {
            "status": "⚠️ Demand exceeds supply",
            "actions": [
                "Discharge storage or import from neighboring zone.",
                "Increase bidding price on IEX to secure capacity.",
            ],
        }

    # ✅ JSON response
    return {
        "predicted_generation_MW": round(float(predicted_gen), 2),
        "predicted_price_USD_MWh": round(float(predicted_price), 2) if not np.isnan(predicted_price) else None,
        "current_generation_MW": round(float(current_gen), 2),
        "storage_level_MWh": round(float(storage), 2) if not np.isnan(storage) else None,
        "recommendation": recommendation,
    }
@app.get("/api/price_forecast")
def price_forecast():
    """
    Simulate dynamic bidding price forecast (₹/MWh)
    based on predicted demand-supply imbalance and ±20% IEX market volatility.
    Ensures realistic visible fluctuations for visualization.
    """
    base_price = 4000  # INR/MWh
    volatility = 0.2   # ±20%
    np.random.seed(int(dt.datetime.now().timestamp()) % 10000)

    # ✅ Load unified dataset (local or Kaggle)
    local_path = os.path.join(BASE_DIR, "..", "dataset", "unified_energy_full.csv")
    kaggle_path = "/kaggle/working/unified_energy_full.csv"
    if os.path.exists(local_path):
        data_path = local_path
    elif os.path.exists(kaggle_path):
        data_path = kaggle_path
    else:
        raise FileNotFoundError("unified_energy_full.csv not found in dataset folder.")

    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    df = df.set_index("Datetime").sort_index()

    # ✅ Clean numeric columns
    df["Total_Gen_MW"] = pd.to_numeric(df["Total_Gen_MW"], errors="coerce").fillna(method="ffill")
    df["Total_Demand_MW"] = pd.to_numeric(df["Total_Demand_MW"], errors="coerce").fillna(method="ffill")

    # ✅ Take last 24 hours of data
    latest = df.iloc[-24:].copy()

    # ✅ Compute demand/supply ratio (fallback to 1 if zero division)
    demand_ratio = latest["Total_Demand_MW"] / (latest["Total_Gen_MW"].replace(0, np.nan))
    demand_ratio = demand_ratio.fillna(1.0)

    # ✅ Add random ±10% weather or market fluctuations
    random_fluct = np.random.normal(0, 0.1, size=len(latest))
    adjusted_ratio = demand_ratio + random_fluct

    # ✅ Convert ratio to price with controlled scaling
    dynamic_price = base_price * (1 + volatility * (adjusted_ratio - 1))

    # ✅ Clip values to ±20%
    dynamic_price = np.clip(dynamic_price, base_price * 0.8, base_price * 1.2)

    timestamps = latest.index.strftime("%Y-%m-%d %H:%M").tolist()

    # ✅ Return simulated data
    return {
        "timestamps": timestamps,
        "dynamic_price_inr_per_mwh": dynamic_price.round(2).tolist(),
        "base_price": base_price,
        "volatility": f"±{int(volatility * 100)}%",
        "note": "Includes ±10% random fluctuation weighted by demand ratio for realistic IEX simulation."
    }
