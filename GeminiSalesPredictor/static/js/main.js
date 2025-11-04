// Global variables
let voiceEnabled = true;
let charts = {};

// Initialize
document.addEventListener("DOMContentLoaded", function () {
  loadStats();
  loadSalesData();
  loadFakeFuturePrediction(); // using fake data
});

// -----------------------
// Load dashboard statistics
// -----------------------
async function loadStats() {
  try {
    const response = await fetch("/api/stats");
    const data = await response.json();

    document.getElementById("totalGames").textContent = data.total_games.toLocaleString();
    document.getElementById("totalSales").textContent = (data.total_sales / 1e6).toFixed(1) + "M";
    document.getElementById("avgRating").textContent = data.avg_rating.toFixed(2);
    document.getElementById("avgMetacritic").textContent = data.avg_metacritic.toFixed(0);
  } catch (err) {
    console.warn("⚠️ Using mock stats (API unavailable)");
    document.getElementById("totalGames").textContent = "1,200";
    document.getElementById("totalSales").textContent = "32.5M";
    document.getElementById("avgRating").textContent = "4.3";
    document.getElementById("avgMetacritic").textContent = "83";
  }
}

// -----------------------
// Load Sales Data
// -----------------------
async function loadSalesData() {
  try {
    const response = await fetch("/api/sales_data");
    const data = await response.json();

    createConsoleChart(data.sales_by_console);
    createRegionalChart(data.regional_sales);
    createTopGamesChart(data.top_games);
  } catch (err) {
    console.warn("⚠️ Using fake sales data");

    createConsoleChart({ PS3: 35, PS4: 70, PS5: 95 });
    createRegionalChart({ NA: 60, EU: 45, JP: 25, Other: 20 });
    createTopGamesChart([
      { Name: "Spider-Man", "Total Sales": 12000000 },
      { Name: "God of War", "Total Sales": 10500000 },
      { Name: "The Last of Us", "Total Sales": 9800000 },
      { Name: "Horizon Zero Dawn", "Total Sales": 8700000 },
      { Name: "Ghost of Tsushima", "Total Sales": 8300000 },
    ]);
  }
}

// -----------------------
// Create Console Chart
// -----------------------
function createConsoleChart(data) {
  const ctx = document.getElementById("consoleChart").getContext("2d");
  if (charts.console) charts.console.destroy();

  charts.console = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: Object.keys(data),
      datasets: [
        {
          data: Object.values(data),
          backgroundColor: ["#667eea", "#764ba2", "#f093fb"],
          borderWidth: 2,
          borderColor: "#fff",
        },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
    },
  });
}

// -----------------------
// Regional Chart
// -----------------------
function createRegionalChart(data) {
  const ctx = document.getElementById("regionalChart").getContext("2d");
  if (charts.regional) charts.regional.destroy();

  charts.regional = new Chart(ctx, {
    type: "pie",
    data: {
      labels: Object.keys(data),
      datasets: [
        {
          data: Object.values(data),
          backgroundColor: ["#667eea", "#764ba2", "#f093fb", "#4facfe"],
          borderWidth: 2,
          borderColor: "#fff",
        },
      ],
    },
    options: { plugins: { legend: { position: "bottom" } } },
  });
}

// -----------------------
// Top Games Chart
// -----------------------
function createTopGamesChart(games) {
  const ctx = document.getElementById("topGamesChart").getContext("2d");
  if (charts.topGames) charts.topGames.destroy();

  charts.topGames = new Chart(ctx, {
    type: "bar",
    data: {
      labels: games.map((g) => g.Name),
      datasets: [
        {
          label: "Sales (Millions)",
          data: games.map((g) => g["Total Sales"] / 1e6),
          backgroundColor: "#667eea",
          borderColor: "#5568d3",
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

// -----------------------
// Fake Future Prediction
// -----------------------
function loadFakeFuturePrediction() {
  const ctx = document.getElementById("futureChart").getContext("2d");
  if (charts.future) charts.future.destroy();

  const months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"];
  const actualSales = [120, 135, 150, 160, 185, 210, 240];
  const predictedSales = [240, 255, 270, 285, 310, 330, 350];

  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, "rgba(102,126,234,0.5)");
  gradient.addColorStop(1, "rgba(118,75,162,0.05)");

  charts.future = new Chart(ctx, {
    type: "line",
    data: {
      labels: months,
      datasets: [
        {
          label: "Historical Sales",
          data: actualSales,
          borderColor: "#667eea",
          backgroundColor: gradient,
          tension: 0.4,
          fill: true,
        },
        {
          label: "Predicted Sales",
          data: predictedSales,
          borderColor: "#764ba2",
          backgroundColor: "rgba(118,75,162,0.1)",
          borderDash: [5, 5],
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Sales (in ₹K)" } },
        x: { title: { display: true, text: "Month" } },
      },
    },
  });
}

// -----------------------
// Chatbot functionality (same as before)
// -----------------------
async function sendMessage() {
  const input = document.getElementById("chatInput");
  const msg = input.value.trim();
  if (!msg) return;
  addMessage("user", msg);
  input.value = "";

  const loading = addMessage("bot", '<span class="loading"></span>');
  animateMouth(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();
    loading.remove();
    addMessage("bot", data.response || "I'm not sure, try rephrasing?");
    if (voiceEnabled) speak(data.response);
  } catch {
    loading.remove();
    addMessage("bot", "⚠️ Connection issue. Try again later.");
  }
  animateMouth(false);
}

function addMessage(sender, text) {
  const box = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = `message ${sender}-message`;
  div.innerHTML = `<strong>${sender === "user" ? "You" : "Bot"}:</strong> ${text}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

function handleKeyPress(e) {
  if (e.key === "Enter") sendMessage();
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.onstart = () => animateMouth(true);
  utterance.onend = () => animateMouth(false);
  window.speechSynthesis.speak(utterance);
}

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById("voiceBtn");
  btn.textContent = voiceEnabled ? "🔊 Voice On" : "🔇 Voice Off";
  if (!voiceEnabled) window.speechSynthesis.cancel();
}

function animateMouth(speaking) {
  const mouth = document.getElementById("botMouth");
  if (speaking) {
    mouth.classList.add("speaking");
    if (!mouth.animationInterval) {
      mouth.animationInterval = setInterval(() => mouth.classList.toggle("speaking"), 200);
    }
  } else {
    mouth.classList.remove("speaking");
    clearInterval(mouth.animationInterval);
    mouth.animationInterval = null;
  }
}
