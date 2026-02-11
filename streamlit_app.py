import streamlit as st
import pandas as pd

# Reuse backend logic
from backend.main import (
    get_latest_data,
    forecast_next_24h,
    get_weekly_trend,
    calculate_energy_cost,
    get_co2_emission,
    demand_forecast,
)

st.set_page_config(page_title="Smart Energy Forecast", layout="wide")
st.title("Smart Energy Forecast Dashboard")

st.caption("Powered by your existing backend logic")

col1, col2, col3 = st.columns(3)

latest = get_latest_data()
forecast = forecast_next_24h()
weekly = get_weekly_trend()
cost = calculate_energy_cost()
emission = get_co2_emission()
demand = demand_forecast()

with col1:
    st.subheader("Latest 24h Usage")
    if latest["timestamps"]:
        df_latest = pd.DataFrame({"time": latest["timestamps"], "power": latest["power"]})
        st.line_chart(df_latest.set_index("time"))
    else:
        st.info("No latest data available.")

with col2:
    st.subheader("Next 24h Forecast")
    if forecast["timestamps"]:
        df_forecast = pd.DataFrame({"time": forecast["timestamps"], "forecast": forecast["forecast"]})
        st.line_chart(df_forecast.set_index("time"))
    else:
        st.info("No forecast available.")

with col3:
    st.subheader("Weekly Trend")
    if weekly["dates"]:
        df_week = pd.DataFrame({"date": weekly["dates"], "avg_power": weekly["avg_power"]})
        st.area_chart(df_week.set_index("date"))
    else:
        st.info("No weekly trend data.")

st.divider()

col4, col5 = st.columns(2)

with col4:
    st.subheader("Cost Snapshot (24h)")
    if cost["timestamps"]:
        df_cost = pd.DataFrame({"time": cost["timestamps"], "cost": cost["cost"]})
        st.line_chart(df_cost.set_index("time"))
        st.write("Total cost:", cost["summary"]["total_cost"])
        st.write("Avg tariff:", cost["summary"]["avg_tariff"])
    else:
        st.info("No cost data available.")

with col5:
    st.subheader("CO2 Emissions (7 days)")
    if emission["dates"]:
        df_em = pd.DataFrame({"date": emission["dates"], "emission": emission["emission"]})
        st.line_chart(df_em.set_index("date"))
        st.write("Total emissions:", emission["summary"]["total_emission"])
        st.write("Total savings:", emission["summary"]["total_savings"])
    else:
        st.info("No emission data available.")

st.divider()
st.subheader("Demand Forecast Summary")
st.json(demand)
