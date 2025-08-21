# 🏦 Credit Risk Xplain v2 – Live Data + Event Integration + Trends/Alerts

This is an upgraded end-to-end prototype aligned with the hackathon spec:

- **2× Structured**: Yahoo Finance (returns, volatility, momentum) + World Bank (GDP growth, inflation)
- **1× Unstructured**: Real-time news (NewsAPI or Reuters RSS) with
  - **Sentiment** (VADER)
  - **Event extraction & classification** (rule-based patterns → risk tags)
- **Scoring Engine**: Interpretable Logistic Regression with feature contributions
- **Explainability**: Contributions + plain-language reasoning + **latest events embedded**
- **Trends & Alerts**: Simple persistence (SQLite) to compute short vs. long trend and spike alerts
- **Frontend**: React (Vite) dashboard showing score, explanation, events, trends, alerts
- **Docker**: Backend Dockerfile for quick hosting

> No LLM prompting is used for explanations.

---

## 🚀 Quickstart

### Backend
```bash
cd backend
pip install -r requirements.txt
# optional for better news coverage:
# export NEWSAPI_KEY=your_newsapi_key
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```

Open http://localhost:5173

---

## 🧠 How event integration works
- Fetch recent headlines via **NewsAPI** (if key) or **Reuters RSS** (fallback).
- Classify events using **rule-based patterns** (e.g., “debt restructuring”, “CEO resigns”, “rating downgrade”).
- Aggregate an **event_risk** score ∈ ~[-1, +1] which feeds the model, and return the matched event list.
- Explanation appends the strongest detected event headline.

> Swap in spaCy/HuggingFace later for entity-level classification if you want more depth.

---

## 📈 Trends & Alerts
- Persistence in **SQLite** (backend/state.db) stores each scoring call per symbol.
- Compute **short-term (last 5)** vs **long-term (all)** averages and **delta** from last score.
- Alerts:
  - `score_spike` if |Δ| ≥ 40
  - `downtrend` if short-term ≪ long-term

---

## ❗ Notes
- This is a hackathon-friendly prototype focused on clarity + stability.
- Add CRON/worker for scheduled refresh and auto-retraining as a next step.
