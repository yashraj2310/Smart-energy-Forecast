// BASE URL - change if your backend is hosted elsewhere
const BASE_URL = "http://127.0.0.1:8000";

/* -------------------------
   Safe helper: getContextIfExists
   ------------------------- */
function ctxOrNull(id) {
  const el = document.getElementById(id);
  return el ? el.getContext("2d") : null;
}

/* Chart contexts (guarded) */
const usageCtx = ctxOrNull("usageChart");
const forecastCtx = ctxOrNull("forecastChart");
const weeklyCtx = ctxOrNull("weeklyChart");
const costCtx = ctxOrNull("costChart");
const emissionCtx = ctxOrNull("emissionChart");
const monthlyBillCtx = ctxOrNull("monthlyBillChart");
const priceCtx = ctxOrNull("priceChart"); // optional

let usageChart, forecastChart, weeklyChart, costChart, emissionChart, monthlyChart, priceChart;

/* Fetch core endpoints with safe fallback */
async function fetchJson(path) {
  try {
    const res = await fetch(`${BASE_URL}${path}`);
    if (!res.ok) {
      console.warn(`Fetch ${path} returned ${res.status}`);
      return null;
    }
    return await res.json();
  } catch (e) {
    console.error("Network error fetching", path, e);
    return null;
  }
}

/* Fetch & update main data */
async function fetchData() {
  const [latest, forecast, history] = await Promise.all([
    fetchJson("/api/latest"),
    fetchJson("/api/forecast"),
    fetchJson("/api/history"),
  ]);

  if (latest && forecast && history) {
    updateCharts(latest, forecast, history);
    updateInsights(latest, forecast);
  }
}

/* Update Chart.js charts (safeguarded) */
function safeDestroy(chart) {
  try { if (chart && typeof chart.destroy === "function") chart.destroy(); } catch {}
}

function updateCharts(latest = {timestamps:[], power:[]}, forecast = {timestamps:[], forecast:[]}, history = {dates:[], avg_power:[]}) {
  // Usage
  safeDestroy(usageChart);
  if (usageCtx && Array.isArray(latest.timestamps) && latest.power) {
    usageChart = new Chart(usageCtx, {
      type: "line",
      data: { labels: latest.timestamps, datasets: [{ label: "Forecasted Power Usage (kW)", data: latest.power, borderColor: "#00d2ff", backgroundColor: "rgba(0,210,255,0.12)", fill: true, tension: 0.35 }]},
      options: { plugins:{legend:{labels:{color:"#fff"}}}, scales:{ x:{ticks:{color:"#bbb"}}, y:{ticks:{color:"#bbb"}}}}
    });
  }

  // Forecast
  safeDestroy(forecastChart);
  if (forecastCtx && Array.isArray(forecast.timestamps) && forecast.forecast) {
    forecastChart = new Chart(forecastCtx, {
      type: "line",
      data: { labels: forecast.timestamps, datasets: [{ label: "Optimised Forecast Power (kW)", data: forecast.forecast, borderColor: "#ff6b6b", backgroundColor: "rgba(255,107,107,0.12)", fill: true, tension: 0.35 }]},
      options: { plugins:{legend:{labels:{color:"#fff"}}}, scales:{ x:{ticks:{color:"#bbb"}}, y:{ticks:{color:"#bbb"}}}}
    });
  }

  // Weekly
  safeDestroy(weeklyChart);
  if (weeklyCtx && Array.isArray(history.dates)) {
    weeklyChart = new Chart(weeklyCtx, {
      type: "bar",
      data: { labels: history.dates, datasets: [{ label: "Daily Avg (kW)", data: history.avg_power, backgroundColor: "#6610f2" }]},
      options: { plugins:{legend:{labels:{color:"#fff"}}}, scales:{ x:{ticks:{color:"#bbb"}}, y:{ticks:{color:"#bbb"}}}}
    });
  }
}

/* Insights box */
function updateInsights(latest = {power:[]}, forecast = {forecast:[]}) {
  const alertsEl = document.getElementById("alerts");
  try {
    const a = Array.isArray(latest.power) && latest.power.length ? latest.power : [0];
    const f = Array.isArray(forecast.forecast) && forecast.forecast.length ? forecast.forecast : [0];
    const avgRecent = a.reduce((s, v) => s + (v||0), 0) / a.length || 0.0001;
    const avgForecast = f.reduce((s, v) => s + (v||0), 0) / f.length || 0;
    const diff = ((avgForecast - avgRecent) / Math.max(avgRecent,0.0001)) * 100;
    if (diff > 0.5) alertsEl.innerText = `⚠️ Predicted +${diff.toFixed(1)}% usage tomorrow.`;
    else if (diff < -0.5) alertsEl.innerText = `✅ Forecast −${Math.abs(diff).toFixed(1)}% — efficiency improving!`;
    else alertsEl.innerText = `ℹ️ Forecast stable (±0.5%).`;
  } catch (e) {
    alertsEl.innerText = "⚠️ Unable to compute insights.";
  }
}

/* Cost */
async function fetchCost() {
  const data = await fetchJson("/api/cost");
  if (!data) return;
  if (costCtx) {
    safeDestroy(costChart);
    costChart = new Chart(costCtx, {
      type: "bar",
      data: { labels: data.timestamps, datasets: [{ label: "Energy Cost (₹)", data: data.cost, backgroundColor: data.tariff.map(t => t===8?"#ff4d4d":t===6?"#ffd43b":"#51cf66") }]},
      options: {
    plugins: {
        legend: {
            labels: { color: '#fff' }
        }
    },
    scales: {
        x: { ticks: { color: '#bbb' } },
        y: { ticks: { color: '#bbb' } }
    }
}

    });
  }
  const summary = data.summary || {};
  const cs = document.getElementById("costSummary");
  if (cs) cs.innerHTML = `<p>💡 Total Cost (Last 24 h): <b>₹${summary.total_cost ?? 0}</b></p>
                         <p>⚙️ Avg Tariff: <b>₹${summary.avg_tariff ?? 0}/kWh</b></p>
                         <p>⏰ Peak Hours: <b>${(summary.peak_hours||[]).slice(0,3).join(", ")}</b></p>`;
}

/* Emission */
async function fetchEmission() {
  const data = await fetchJson("/api/emission");
  if (!data) return;
  if (emissionCtx) {
    safeDestroy(emissionChart);
    emissionChart = new Chart(emissionCtx, {
      type: "bar",
      data: { labels: data.dates, datasets: [{ label: "Daily CO₂ Emission (kg)", data: data.emission, backgroundColor: (data.emission||[]).map(e => e > (data.summary?.baseline||0) ? "#ff6b6b" : "#51cf66") }]},
      options: { plugins:{legend:{labels:{color:'#fff'}}}, scales:{ x:{ticks:{color:'#bbb'}}, y:{ticks:{color:'#bbb'}}}}
    });
  }
  const s = data.summary || {};
  const el = document.getElementById("emissionSummary");
  if (el) el.innerHTML = `<p>🌎 Total Emission (last 7 days): <b>${s.total_emission ?? 0} kg CO₂</b></p>
                          <p>🌱 Total Saved Emission: <b>${s.total_savings ?? 0} kg CO₂</b></p>
                          <p>📉 Baseline (average): <b>${s.baseline ?? 0} kg/day</b></p>`;
}

/* AI tip */
async function fetchTip(){
  const data = await fetchJson("/api/suggestion");
  const el = document.getElementById("aiTip");
  if (!el) return;
  el.innerText = data && data.suggestion ? data.suggestion : "No tip available right now.";
}

/* AIForecast card & gauge-like values */
async function loadAIForecast(){
  const data = await fetchJson("/api/demand_forecast");
  const container = document.getElementById("forecast-summary");
  if (!container) return;
  if (!data) { container.innerHTML = "<p>⚠️ Could not fetch AI forecast.</p>"; return; }

  const rec = data.recommendation || {status:"No data", actions:[]};
  const isWarn = rec.status && rec.status.includes("⚠️");
  container.innerHTML = `<div class="forecast-card ${isWarn? 'negative':''}">
    <h3>${rec.status}</h3>
    <p><strong>Predicted Generation:</strong> ${Number(data.predicted_generation_MW||0).toFixed(2)} MW</p>
    <p><strong>Current Generation:</strong> ${Number(data.current_generation_MW||0).toFixed(2)} MW</p>
    <p><strong>Storage Level:</strong> ${Number(data.storage_level_MWh||0).toFixed(2)} MWh</p>
    <h4>🔍 Smart Actions:</h4>
    <ul>${(rec.actions||[]).map(a => `<li>${a}</li>`).join("")}</ul>
  </div>`;
}

/* Monthly bill predictor */
async function loadMonthlyBill(){
  const loading = document.getElementById("monthly-bill-loading");
  const resultBox = document.getElementById("monthly-bill-result");
  try {
    const data = await fetchJson("/api/monthly_bill");
    if (!data) {
      if (loading) loading.innerText = "⚠️ Monthly bill not available.";
      return;
    }
    if (loading) loading.style.display = "none";
    if (resultBox) resultBox.style.display = "block";
    document.getElementById("bill-month").innerText = data.month || "—";
    document.getElementById("bill-usage").innerText = (data.total_usage_kwh||0).toFixed(2);
    document.getElementById("bill-cost").innerText = (data.total_cost||0).toFixed(2);

    // draw chart
    if (monthlyBillCtx && Array.isArray(data.daily_breakdown)) {
      safeDestroy(monthlyChart);
      monthlyChart = new Chart(monthlyBillCtx, {
        type: "bar",
        data: { labels: data.daily_breakdown.map(d => d.date), datasets: [{ label: "Daily Cost (₹)", data: data.daily_breakdown.map(d => d.cost), backgroundColor:"#00e0ff" }]},
        options: { plugins:{legend:{labels:{color:'#fff'}}}, scales:{ x:{ticks:{color:'#bbb'}}, y:{ticks:{color:'#bbb'}}}}
      });
    }
  } catch (e) {
    console.error("Monthly bill load error", e);
    if (loading) loading.innerText = "⚠️ Failed to load monthly bill.";
  }
}

/* WebSocket live alerts (keeps UI lively) */
function connectWebSocket() {
  try {
    const ws = new WebSocket(`${BASE_URL.replace(/^http/, 'ws')}/ws/updates`);
    ws.onopen = () => console.log("WS connected");
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        const alerts = document.getElementById("alerts");
        if (alerts && data.alert) {
          alerts.innerText = data.alert;
          alerts.style.background = "#111";
        }
      } catch {}
    };
    ws.onclose = () => setTimeout(connectWebSocket, 3000);
  } catch (e) { console.warn("WS connect failed", e) }
}

/* RAG Chat (frontend) */
const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("userQuery");
const sendChatBtn = document.getElementById("sendBtn");
const langSelect = document.getElementById("langSelect");
const chatError = document.getElementById("chatError");

if (sendChatBtn) sendChatBtn.addEventListener("click", sendChat);
if (chatInput) chatInput.addEventListener("keypress", (e) => { if (e.key === "Enter") sendChat(); });

async function sendChat(){
  const q = chatInput.value && chatInput.value.trim();
  if (!q) return;
  appendChat("You", q);
  chatInput.value = "";
  try {
    const res = await fetch(`${BASE_URL}/api/rag_chat`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q, lang: langSelect ? langSelect.value : "en"})
    });
    if (!res.ok) {
      const txt = await res.text();
      appendChat("AI", `Error: ${res.status} ${txt}`);
      return;
    }
    const data = await res.json();
    appendChat("AI", data.reply || "No reply");
  } catch (e) {
    console.error(e);
    appendChat("AI", "⚠️ Error connecting to AI assistant.");
    if (chatError) { chatError.style.display = "block"; chatError.innerText = "Failed to contact rag_chat endpoint." }
  }
}

function appendChat(who, text) {
  if (!chatBox) return;
  const p = document.createElement("p");
  p.innerHTML = `<b>${who}:</b> ${escapeHtml(String(text))}`;
  chatBox.appendChild(p);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }

/* Boot sequence */
async function boot(){
  connectWebSocket();
  await Promise.all([ fetchData(), fetchCost(), fetchEmission(), fetchTip(), loadAIForecast(), loadMonthlyBill() ]);
  // periodic updates
  setInterval(fetchData, 30_000);
  setInterval(fetchCost, 30_000);
  setInterval(fetchEmission, 60_000);
  setInterval(fetchTip, 60_000);
  setInterval(loadAIForecast, 60_000);
  setInterval(loadMonthlyBill, 60_000);
}

window.addEventListener("load", boot);
