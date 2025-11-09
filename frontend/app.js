const usageCtx = document.getElementById("usageChart").getContext("2d");
const forecastCtx = document.getElementById("forecastChart").getContext("2d");
const weeklyCtx = document.getElementById("weeklyChart").getContext("2d");

let usageChart, forecastChart, weeklyChart;

async function fetchData() {
  const latest = await fetch("http://127.0.0.1:8000/api/latest").then(r => r.json());
  const forecast = await fetch("http://127.0.0.1:8000/api/forecast").then(r => r.json());
  const history = await fetch("http://127.0.0.1:8000/api/history").then(r => r.json());

  updateCharts(latest, forecast, history);
  updateInsights(latest, forecast);
}

function updateCharts(latest, forecast, history) {
  // Destroy existing charts before re-rendering
  if (usageChart) usageChart.destroy();
  if (forecastChart) forecastChart.destroy();
  if (weeklyChart) weeklyChart.destroy();

  // ✅ Actual Power Chart
  usageChart = new Chart(usageCtx, {
    type: "line",
    data: {
      labels: latest.timestamps,
      datasets: [
        {
          label: "Actual Power (kW)",
          data: latest.power,
          borderColor: "#00d2ff",
          backgroundColor: "rgba(0,210,255,0.2)",
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb", maxRotation: 60, minRotation: 45 } },
        y: { ticks: { color: "#bbb" } },
      },
    },
  });

  // ✅ Forecasted Power Chart
  forecastChart = new Chart(forecastCtx, {
    type: "line",
    data: {
      labels: forecast.timestamps,
      datasets: [
        {
          label: "Forecasted Power (kW)",
          data: forecast.forecast,
          borderColor: "#ff6b6b",
          backgroundColor: "rgba(255,107,107,0.2)",
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb", maxRotation: 60, minRotation: 45 } },
        y: { ticks: { color: "#bbb" } },
      },
    },
  });

  // ✅ Weekly Trend Chart
  weeklyChart = new Chart(weeklyCtx, {
    type: "bar",
    data: {
      labels: history.dates,
      datasets: [
        {
          label: "Daily Avg (kW)",
          data: history.avg_power,
          backgroundColor: "#6610f2",
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb" } },
        y: { ticks: { color: "#bbb" } },
      },
    },
  });
}

function updateInsights(latest, forecast) {
  const avgRecent = latest.power.reduce((a, b) => a + b, 0) / latest.power.length;
  const avgForecast = forecast.forecast.reduce((a, b) => a + b, 0) / forecast.forecast.length;
  const diff = ((avgForecast - avgRecent) / avgRecent) * 100;

  document.getElementById("alerts").innerText =
    diff > 0
      ? `⚠️ Predicted +${diff.toFixed(1)}% usage tomorrow.`
      : `✅ Forecast −${Math.abs(diff).toFixed(1)}% — efficiency improving!`;
}

// 🔁 Auto-refresh every 30 seconds
fetchData();
setInterval(fetchData, 30000);
const costCtx = document.getElementById("costChart").getContext("2d");
let costChart;

async function fetchCost() {
  const data = await fetch("http://127.0.0.1:8000/api/cost").then(r => r.json());

  const tariffs = data.tariff.map(v =>
    v === 8 ? "#ff4d4d" : v === 6 ? "#ffd43b" : "#51cf66"
  );

  if (costChart) costChart.destroy();

  costChart = new Chart(costCtx, {
    type: "bar",
    data: {
      labels: data.timestamps,
      datasets: [
        {
          label: "Energy Cost (₹)",
          data: data.cost,
          backgroundColor: tariffs,
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb", maxRotation: 60, minRotation: 45 } },
        y: { ticks: { color: "#bbb" }, title: { display: true, text: "₹ per hour" } },
      },
    },
  });

  const { total_cost, avg_tariff, peak_hours } = data.summary;
  document.getElementById("costSummary").innerHTML = `
    <p>💡 Total Cost (Last 24 h): <b>₹${total_cost}</b></p>
    <p>⚙️ Avg Tariff: <b>₹${avg_tariff}/kWh</b></p>
    <p>⏰ Peak Hours: <b>${peak_hours.slice(0, 3).join(", ")} ...</b></p>
  `;
}

// Fetch cost data alongside other charts
fetchCost();
setInterval(fetchCost, 30000);
// -------------------------
// 🔌 WebSocket Live Updates
// -------------------------
function connectWebSocket() {
  const ws = new WebSocket("ws://127.0.0.1:8000/ws/updates");

  ws.onopen = () => console.log("✅ WebSocket connected");
 ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const alertBox = document.getElementById("alerts");
  alertBox.innerText = data.alert;
  alertBox.style.background = "#282828";
  alertBox.style.color = "#ffd43b";

  // Show desktop notification
  if (Notification.permission === "granted") {
    new Notification("⚡ Smart Energy Alert", {
      body: data.alert,
      icon: "./icons/icon-192.png"
    });
  }
};


  ws.onclose = () => {
    console.warn("WebSocket disconnected, reconnecting...");
    setTimeout(connectWebSocket, 3000);
  };
}
connectWebSocket();
async function fetchTip() {
  const data = await fetch("http://127.0.0.1:8000/api/suggestion").then(r => r.json());
  document.getElementById("aiTip").innerText = data.suggestion;
}
fetchTip();
setInterval(fetchTip, 60000); // refresh tip every 1 min
// -----------------------------
// 🌿 CO₂ Emission Tracker
// -----------------------------
const emissionCtx = document.getElementById("emissionChart").getContext("2d");
let emissionChart;

async function fetchEmission() {
  const data = await fetch("http://127.0.0.1:8000/api/emission").then(r => r.json());
  const colors = data.emission.map((e, i) =>
    e > data.summary.baseline ? "#ff6b6b" : "#51cf66"
  );

  if (emissionChart) emissionChart.destroy();

  emissionChart = new Chart(emissionCtx, {
    type: "bar",
    data: {
      labels: data.dates,
      datasets: [{
        label: "Daily CO₂ Emission (kg)",
        data: data.emission,
        backgroundColor: colors,
      }],
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb" } },
        y: {
          ticks: { color: "#bbb" },
          title: { display: true, text: "kg CO₂/day" },
        },
      },
    },
  });

  const { total_emission, total_savings, baseline } = data.summary;
  document.getElementById("emissionSummary").innerHTML = `
    <p>🌎 Total Emission (last 7 days): <b>${total_emission} kg CO₂</b></p>
    <p>🌱 Total Saved Emission: <b>${total_savings} kg CO₂</b></p>
    <p>📉 Baseline (average): <b>${baseline} kg/day</b></p>
  `;
}

fetchEmission();
setInterval(fetchEmission, 60000);
// ---------------------------
// 🔔 Notification Permission
// ---------------------------
async function requestNotificationPermission() {
  if (!("Notification" in window)) return;
  let permission = Notification.permission;
  if (permission === "default") {
    permission = await Notification.requestPermission();
  }
  console.log("Notification permission:", permission);
}
requestNotificationPermission();
async function loadAIForecast() {
  try {
    const response = await fetch("http://127.0.0.1:8000/api/demand_forecast");
    const data = await response.json();
    console.log("AI Forecast data:", data);

    // Use correct field names
    const predicted = data.predicted_generation_MW ?? data.predicted_demand_MW ?? 0;
    const generation = data.current_generation_MW ?? 0;
    const storage = data.storage_level_MWh ?? 0;
    const rec = data.recommendation ?? { status: "ℹ️ No data", actions: [] };

    const isWarning = rec.status.includes("⚠️");

    document.getElementById("forecast-summary").innerHTML = `
      <div class="forecast-card ${isWarning ? "negative" : "positive"}">
        <h3>${rec.status}</h3>
        <p><strong>Predicted Generation:</strong> ${predicted.toFixed(2)} MW</p>
        <p><strong>Current Generation:</strong> ${generation.toFixed(2)} MW</p>
        <p><strong>Storage Level:</strong> ${storage.toFixed(2)} MWh</p>
        <h4>🔍 Smart Actions:</h4>
        <ul>${rec.actions.map(a => `<li>${a}</li>`).join("")}</ul>
      </div>
    `;
  } catch (err) {
    console.error("Error loading AI forecast:", err);
    document.getElementById("forecast-summary").innerHTML = `<p>⚠️ Could not fetch AI forecast.</p>`;
  }
}
window.addEventListener("load", () => {
  console.log("✅ Page fully loaded, initializing forecast...");
  loadAIForecast();
  setInterval(loadAIForecast, 60000);
});
// =========================================================
// ⚡ Demand–Supply Gauge Meter (Chart.js Doughnut)
// =========================================================

let gaugeChart;

function renderGauge(demand, generation) {
  const ctx = document.getElementById("balanceGauge").getContext("2d");

  // 🧠 Validate inputs
  if (!demand || !generation || demand === 0 || isNaN(demand) || isNaN(generation)) {
    if (gaugeChart) gaugeChart.destroy();
    gaugeChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["No Data"],
        datasets: [{
          data: [100],
          backgroundColor: ["#555"],
          borderWidth: 0,
          cutout: "75%",
        }],
      },
      options: {
        rotation: -90,
        circumference: 180,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
          title: {
            display: true,
            text: "No data available",
            color: "#fff",
            font: { size: 18 },
          },
        },
      },
    });
    return;
  }

  // ✅ Compute ratio safely
  const ratio = generation / demand;
  const percent = Math.min((ratio * 100).toFixed(1), 150);

  // 🎨 Color logic
  let color = "#28a745"; // green
  if (ratio < 0.8) color = "#dc3545"; // red (deficit)
  else if (ratio < 1.0) color = "#ffc107"; // yellow (slightly below)

  const data = {
    labels: ["Generation", "Shortfall"],
    datasets: [{
      data: [percent, 150 - percent],
      backgroundColor: [color, "#333"],
      borderWidth: 0,
      cutout: "75%",
    }],
  };

  const options = {
    rotation: -90,
    circumference: 180,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
      title: {
        display: true,
        text: `Grid Balance: ${percent}%`,
        color: "#fff",
        font: { size: 18 },
      },
    },
  };

  // 🌀 Animate updates smoothly
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, { type: "doughnut", data, options });
}

// =========================================================
// 🔁 Connect Gauge with AI Forecast data
// =========================================================

async function updateGauge() {
  try {
    const response = await fetch("http://127.0.0.1:8000/api/demand_forecast");
    const data = await response.json();
    console.log("Gauge Data:", data);

    // Fix property names based on actual API output
    const generation = parseFloat(data.predicted_generation_MW);
    const currentGen = parseFloat(data.current_generation_MW);

    renderGauge(generation, currentGen);
  } catch (error) {
    console.error("Gauge update failed:", error);
  }
}


// Auto-refresh every 30 s
updateGauge();
setInterval(updateGauge, 30000);
const priceCtx = document.getElementById("priceChart").getContext("2d");
let priceChart;

async function loadPriceForecast() {
  const res = await fetch("http://127.0.0.1:8000/api/price_forecast");
  const data = await res.json();

  if (priceChart) priceChart.destroy();
  priceChart = new Chart(priceCtx, {
    type: "line",
    data: {
      labels: data.timestamps,
      datasets: [{
        label: "Dynamic Bidding Price (₹/MWh)",
        data: data.dynamic_price_inr_per_mwh,
        borderColor: "#ffcc00",
        backgroundColor: "rgba(255, 204, 0, 0.2)",
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#bbb", maxRotation: 60 } },
        y: { ticks: { color: "#bbb" }, title: { display: true, text: "₹ per MWh", color: "#fff" } }
      }
    }
  });
}

loadPriceForecast();
document.getElementById("storage-action").innerHTML = `
  🔋 Storage Action: <strong>${data.storage_action.toUpperCase()}</strong> <br>
  Storage Level: ${data.storage_level_MWh} MWh
`;
