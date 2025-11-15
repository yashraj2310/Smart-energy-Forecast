# backend/main.py
# Full corrected, robust version of your Smart Energy Forecast backend.
# IMPORTANT: set your GROQ key in environment variable GROQ_API_KEY before running:
#   PowerShell:  setx GROQ_API_KEY "your_key_here"  (then restart terminal)
#   Linux/mac:   export GROQ_API_KEY="your_key_here"
import uuid
import json
from collections import deque
from fastapi import Response, Cookie
from fastapi.responses import StreamingResponse
import asyncio
from typing import Any, Dict, List, Optional
import os
import random
import asyncio
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Prophet can be strict in typing stubs; silence some checks if needed
from prophet import Prophet  # type: ignore

# language detection
from langdetect import detect

# Groq SDK (install with: pip install groq)
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None  # we'll handle missing SDK at runtime

# ---------------------------
# Paths & constants
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'energy_data.db')}"
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_forecast_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "household_power_consumption.txt")

# ---------------------------
# FastAPI app + middleware
# ---------------------------
app = FastAPI(title="Smart Energy Optimization API")

# Enable CORS so your frontend (different origin/port) can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Database Setup (SQLite)
# ---------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class EnergyReading(Base):
    __tablename__ = "energy_readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    power = Column(Float, nullable=False)


Base.metadata.create_all(bind=engine)

# ---------------------------
# Groq client initialization (robust)
# ---------------------------
GROQ_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_KEY and Groq is not None:
    try:
        client = Groq(api_key=GROQ_KEY)
        print("✅ Groq client initialized.")
    except Exception as e:
        client = None
        print("❌ Groq initialization failed:", e)
else:
    if Groq is None:
        print("⚠️ Groq SDK not installed. Install with `pip install groq`.")
    else:
        print("⚠️ GROQ_API_KEY not set; rag_chat disabled.")

# ---------------------------
# Load XGBoost model (if present)
# ---------------------------
xgb_model: Optional[Any] = None
try:
    if os.path.exists(MODEL_PATH):
        xgb_model = joblib.load(MODEL_PATH)
        print("✅ XGBoost model loaded.")
    else:
        print("⚠️ XGBoost model file not found; continuing without it.")
except Exception as ex:
    xgb_model = None
    print("⚠️ Failed loading XGBoost model:", str(ex))

# ---------------------------
# Load dataset (if present)
# ---------------------------
if os.path.exists(DATA_PATH):
    try:
        df_raw = pd.read_csv(DATA_PATH, sep=";", low_memory=False)
        df_raw["Datetime"] = pd.to_datetime(
            df_raw["Date"].astype(str) + " " + df_raw["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df_raw["Global_active_power"] = pd.to_numeric(df_raw["Global_active_power"], errors="coerce")
        df = df_raw[["Datetime", "Global_active_power"]].dropna()
        if not df.empty:
            df = df.set_index("Datetime")
            df = df.resample("h").mean()
        print("✅ Dataset loaded and resampled hourly.")
    except Exception as e:
        print("⚠️ Error loading dataset:", str(e))
        df = pd.DataFrame(columns=["Global_active_power"])
else:
    df = pd.DataFrame(columns=["Global_active_power"])  # fallback empty
    print("⚠️ Dataset file not found; using empty DataFrame.")

# ---------------------------
# Utility helpers
# ---------------------------
def _as_dtindex(index) -> pd.DatetimeIndex:
    """Return a DatetimeIndex cast for Pylance/type-checking friendliness."""
    return pd.DatetimeIndex(index)

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "⚡ Smart Energy Backend Running"}


@app.get("/api/latest")
def get_latest_data() -> Dict[str, List[Any]]:
    if df.empty:
        return {"timestamps": [], "power": []}
    last_24h = df.tail(24).copy()
    idx = _as_dtindex(last_24h.index)
    return {
        "timestamps": idx.strftime("%Y-%m-%d %H:%M").tolist(),
        "power": last_24h["Global_active_power"].fillna(0).tolist(),
    }


@app.get("/api/forecast")
def forecast_next_24h() -> Dict[str, List[Any]]:
    if df.empty or xgb_model is None:
        return {"timestamps": [], "forecast": []}

    last_24 = df.tail(24).copy()
    idx = _as_dtindex(last_24.index)

    # Build features
    X = pd.DataFrame({
        "hour": idx.hour,
        "day": idx.day,
        "weekday": idx.weekday,
        "month": idx.month,
    })

    # Add lag features
    for i in range(1, 25):
        X[f"lag_{i}"] = last_24["Global_active_power"].shift(i).bfill().values

    # Predict using the pre-loaded xgboost model
    try:
        y_pred = xgb_model.predict(X)
    except Exception:
        y_pred = [0.0] * 24

    future_times = [df.index[-1] + timedelta(hours=i + 1) for i in range(24)]
    return {
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in future_times],
        "forecast": np.asarray(y_pred).tolist(),
    }


@app.get("/api/history")
def get_weekly_trend() -> Dict[str, List[Any]]:
    if df.empty:
        return {"dates": [], "avg_power": []}
    last_week = df.tail(24 * 7)
    daily_avg = last_week.resample("D").mean()
    idx = _as_dtindex(daily_avg.index)
    return {
        "dates": idx.strftime("%Y-%m-%d").tolist(),
        "avg_power": daily_avg["Global_active_power"].round(3).fillna(0).tolist(),
    }


@app.get("/api/cost")
def calculate_energy_cost() -> Dict[str, Any]:
    if df.empty:
        return {"timestamps": [], "cost": []}

    last_24h = df.tail(24).copy()
    idx = _as_dtindex(last_24h.index)

    def get_tariff(hour: int) -> float:
        if 18 <= hour <= 22:
            return 8.0
        elif 8 <= hour <= 17:
            return 6.0
        else:
            return 3.5

    # apply tariff
    hours = idx.hour
    tariffs = [get_tariff(int(h)) for h in hours]
    last_24h["tariff"] = tariffs
    last_24h["cost"] = last_24h["Global_active_power"].fillna(0) * last_24h["tariff"]

    total_cost = round(last_24h["cost"].sum(), 2)
    avg_tariff = round(float(np.mean(tariffs)), 2) if tariffs else 0.0
    peak_hours = [t.strftime("%H:%M") for i, t in enumerate(idx) if tariffs[i] == 8.0]

    return {
        "timestamps": idx.strftime("%Y-%m-%d %H:%M").tolist(),
        "tariff": tariffs,
        "cost": last_24h["cost"].round(2).tolist(),
        "summary": {
            "total_cost": total_cost,
            "avg_tariff": avg_tariff,
            "peak_hours": peak_hours,
        },
    }


# ---------------------------
# WebSocket Alerts
# ---------------------------
@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            msg = random.choice([
                "⚡ High load detected — reduce heavy usage.",
                "🌱 Solar contribution +12%.",
                "💰 Off-peak pricing starts soon.",
                "🔥 Peak tariff active (₹8/kWh) till 10 PM.",
            ])
            await websocket.send_json({"alert": msg})
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        print("WebSocket disconnected.")


# ---------------------------
# Suggestions & Emissions
# ---------------------------
@app.get("/api/suggestion")
def get_ai_suggestion_dynamic():
    """Generate live smart energy tips from real data."""

    # Fetch relevant data
    latest = get_latest_data()
    cost_data = calculate_energy_cost()
    forecast = forecast_next_24h()
    emission = get_co2_emission()

    tips = []

    # --- Usage analysis ---
    if latest["power"]:
        avg_usage = sum(latest["power"]) / len(latest["power"])
        peak_usage = max(latest["power"])
        if peak_usage > avg_usage * 1.4:
            tips.append(
                f"Your peak usage is {peak_usage:.2f} kW, much higher than your daily average {avg_usage:.2f} kW — avoid running heavy appliances together."
            )
        if avg_usage > 1.5:
            tips.append(
                f"Your average usage ({avg_usage:.2f} kW) is high — consider using energy-efficient appliances and checking AC/fridge settings."
            )

    # --- Cost pattern analysis ---
    summary = cost_data.get("summary", {})
    if summary:
        if summary["avg_tariff"] >= 6:
            tips.append(
                f"Your average tariff is ₹{summary['avg_tariff']}/kWh — shift washing machine, geyser & EV charging outside {', '.join(summary['peak_hours'][:3])}."
            )
        if summary["total_cost"] > 150:
            tips.append(
                f"You spent ₹{summary['total_cost']} in the last 24 hours — reduce load during peak tariff (₹8/kWh) hours."
            )

    # --- Forecast analysis ---
    if forecast["forecast"]:
        tomorrow_avg = sum(forecast["forecast"]) / len(forecast["forecast"])
        if tomorrow_avg > avg_usage:
            tips.append(
                f"Tomorrow’s expected load is {tomorrow_avg:.2f} kW — plan heavy tasks during off-peak hours to reduce your bill."
            )

    # --- Emission pattern ---
    emi_summary = emission.get("summary", {})
    if emi_summary.get("total_emission", 0) > 120:
        tips.append(
            f"Your CO₂ emission this week is {emi_summary['total_emission']} kg — reducing evening power spikes can lower your footprint."
        )

    # --- Fallback ---
    if not tips:
        tips.append("Your energy usage is stable — no actions needed right now!")

    return {"suggestion": random.choice(tips)}



@app.get("/api/emission")
def get_co2_emission() -> Dict[str, Any]:
    if df.empty:
        return {"dates": [], "emission": []}

    emission_factor = 0.82
    last_7_days = df.tail(24 * 7).copy()
    daily_energy = last_7_days["Global_active_power"].resample("D").sum().fillna(0)
    daily_emission = daily_energy * emission_factor

    baseline = float(daily_emission.mean()) if not daily_emission.empty else 0.0
    savings = (baseline - daily_emission).clip(lower=0)

    return {
        "dates": _as_dtindex(daily_emission.index).strftime("%Y-%m-%d").tolist(),
        "emission": daily_emission.round(2).tolist(),
        "summary": {
            "total_emission": round(float(daily_emission.sum()), 2),
            "total_savings": round(float(savings.sum()), 2),
            "baseline": round(baseline, 2),
        },
    }


@app.get("/api/demand_forecast")
def demand_forecast() -> Dict[str, Any]:
    """
    Real AI-driven forecast using Prophet on stored power readings.
    If database is empty, fallback to simulated data.
    """
    db = SessionLocal()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    data = db.query(EnergyReading).filter(EnergyReading.timestamp >= start_date).all()
    db.close()

    if not data:
        # fallback simulated
        predicted_gen = round(random.uniform(45, 60), 2)
        current_gen = round(random.uniform(40, 55), 2)
        storage_level = round(random.uniform(20, 30), 2)
        return {
            "predicted_generation_MW": predicted_gen,
            "current_generation_MW": current_gen,
            "storage_level_MWh": storage_level,
            "recommendation": {
                "status": "⚠️ Demand likely to exceed supply",
                "actions": [
                    "Increase renewable dispatch or discharge battery storage.",
                    "Raise IEX bidding price to maintain capacity margin.",
                ],
            },
        }

    df_db = pd.DataFrame([(d.timestamp, d.power) for d in data], columns=["ds", "y"]).dropna()
    if df_db.empty:
        return {
            "predicted_generation_MW": 0.0,
            "current_generation_MW": 0.0,
            "storage_level_MWh": 0.0,
            "recommendation": {"status": "No data", "actions": []},
        }

    df_db = df_db.sort_values("ds")

    # Prophet model training
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)  # type: ignore
    prophet_model.fit(df_db)

    future = prophet_model.make_future_dataframe(periods=24, freq="H")
    forecast = prophet_model.predict(future)

    predicted_gen = round(float(forecast["yhat"].tail(24).mean()), 2)
    current_gen = round(float(df_db["y"].tail(24).mean()), 2) if len(df_db) >= 1 else 0.0
    storage_level = round(float(df_db["y"].tail(24).sum()) / 10.0, 2)

    if predicted_gen > current_gen * 1.05:
        recommendation = {
            "status": "⚠️ Demand likely to exceed supply",
            "actions": [
                "Increase renewable dispatch or discharge battery storage.",
                "Raise IEX bidding price to maintain capacity margin.",
            ],
        }
    else:
        recommendation = {
            "status": "✅ Stable or surplus generation",
            "actions": [
                "Charge batteries or export surplus to nearby grid.",
                "Reduce turbine load during off-peak hours.",
            ],
        }

    return {
        "predicted_generation_MW": predicted_gen,
        "current_generation_MW": current_gen,
        "storage_level_MWh": storage_level,
        "recommendation": recommendation,
    }


@app.get("/api/monthly_bill")
def monthly_bill():
    if df.empty:
        return {
            "month": None,
            "total_usage_kwh": 0,
            "total_cost": 0,
            "daily_breakdown": []
        }

    # Take last 30 days
    last_30_days = df.tail(24 * 30).copy()
    idx = _as_dtindex(last_30_days.index)

    def get_tariff(hour: int) -> float:
        if 18 <= hour <= 22:
            return 8.0
        elif 8 <= hour <= 17:
            return 6.0
        else:
            return 3.5

    # Apply tariff
    tariffs = [get_tariff(int(h)) for h in idx.hour]
    last_30_days["tariff"] = tariffs
    last_30_days["cost"] = last_30_days["Global_active_power"].fillna(0) * last_30_days["tariff"]

    # Daily aggregation
    daily = last_30_days.resample("D").sum()

    total_usage = float(daily["Global_active_power"].sum())
    total_bill = float(daily["cost"].sum())

    return {
        "month": idx[0].strftime("%Y-%m"),
        "total_usage_kwh": round(total_usage, 2),
        "total_cost": round(total_bill, 2),
        "daily_breakdown": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "usage_kwh": round(float(row["Global_active_power"]), 2),
                "cost": round(float(row["cost"]), 2)
            }
            for d, row in daily.iterrows()
        ]
    }


@app.post("/api/rag_chat")
async def rag_chat(request: Request):
    """
    Handles conversational queries such as:
    - "why cost increased?"
    - "explain today's forecast"
    - "why bill high this month"
    """

    body = await request.json()
    query = body.get("query", "").strip()
    lang = body.get("lang", "en")

    if not query:
        return JSONResponse({"reply": "Please enter a valid question."})

    # If no Groq client
    if client is None:
        return JSONResponse({
            "reply": "AI model unavailable because GROQ_API_KEY is missing."
        })

    # Build RAG context
    context_parts = []

    # Latest 24h usage
    latest = get_latest_data()
    if latest["power"]:
        avg_use = sum(latest["power"]) / len(latest["power"])
        context_parts.append(f"Average usage last 24h: {avg_use:.2f} kW.")

    # Cost summary
    cost_data = calculate_energy_cost()
    context_parts.append(
        f"Last 24h cost: ₹{cost_data['summary']['total_cost']} | "
        f"Avg tariff: ₹{cost_data['summary']['avg_tariff']}/kWh."
    )

    # Emission
    emi = get_co2_emission()
    context_parts.append(
        f"CO₂ (7 days): {emi['summary']['total_emission']} kg."
    )

    # AI Forecast
    dfc = demand_forecast()
    context_parts.append(
        f"Predicted gen: {dfc['predicted_generation_MW']} MW, "
        f"Current gen: {dfc['current_generation_MW']} MW."
    )

    context = "\n".join(context_parts)

    # ===============================
    # GROQ API CALL (FIXED)
    # ===============================
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
               {"role": "system", "content": 
    f"""
You are an energy assistant for a smart dashboard.

❗ ALWAYS answer in a short, crisp, human tone (3–5 sentences max).
❗ Do NOT do long calculations unless user specifically asks.
❗ Focus on giving clear reasons, insights, and actionable suggestions.
❗ Speak like a friendly expert, not a textbook.

Here is the relevant latest data:
{context}

Now answer the user naturally.
"""
},

                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )

        # FIXED: USE DOT NOTATION
        answer = completion.choices[0].message.content

        return JSONResponse({"reply": answer})

    except Exception as e:
        return JSONResponse({"reply": f"AI processing error: {str(e)}"})
