from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, math, time, datetime as dt, json, sqlite3, re
import numpy as np
import requests

from sklearn.linear_model import LogisticRegression

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except Exception:
    VADER_OK = False

DB_PATH = os.path.join(os.path.dirname(__file__), "state.db")

app = FastAPI(title="Credit Risk Xplain API (v2)", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Model & Features =================
FEATURES = [
    "ret_30d",
    "vol_30d",
    "momentum_7d",
    "gdp_growth",
    "inflation",
    "news_sentiment",
    "event_risk",           # NEW: aggregated risk from detected events
]

rng = np.random.default_rng(123)
X = rng.normal(0, 1, size=(1200, len(FEATURES)))
coef = np.array([0.9, -1.0, 0.6, 0.7, -0.8, 0.7, -0.9])  # event_risk negative for risk↑
y = (X @ coef + rng.normal(0, 0.9, 1200) > 0).astype(int)
_model = LogisticRegression(max_iter=800)
_model.fit(X, y)
_beta = _model.coef_.ravel()
_intercept = float(_model.intercept_)

def _sigmoid(z: float) -> float:
    return 1.0/(1.0+math.exp(-z))

def _ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        ts INTEGER,
        score INTEGER,
        probability REAL,
        features TEXT,
        events TEXT
    )""")
    con.commit()
    con.close()

_ensure_db()

# ================= DTOs =================
class ManualPayload(BaseModel):
    features: dict
    symbol: str | None = None

class LivePayload(BaseModel):
    symbol: str          # e.g., "AAPL" or "RELIANCE.NS"
    country: str         # e.g., "US" or "IN"
    query: str           # e.g., "Reliance Industries"

# ================= Data Sources =================
def fetch_market_metrics(symbol: str):
    if yf is None:
        raise HTTPException(500, detail="yfinance not installed on server")
    t = yf.Ticker(symbol)
    hist = t.history(period="90d", interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise HTTPException(404, detail=f"No market data for {symbol}")
    close = hist["Close"].astype(float)
    ret_30d = float((close[-1]/close[-30]) - 1.0) if len(close)>=30 else float((close[-1]/close[0])-1.0)
    rets = close.pct_change().dropna()
    vol_30d = float(rets[-30:].std()) if len(rets)>=30 else float(rets.std())
    momentum_7d = float((close[-1]/close[-7]) - 1.0) if len(close)>=7 else ret_30d/4.0
    return {"ret_30d": ret_30d, "vol_30d": vol_30d, "momentum_7d": momentum_7d}

def fetch_worldbank_indicator(country: str, indicator: str):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=5"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data)<2:
        return None
    series = data[1] or []
    for entry in series:
        val = entry.get("value")
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return None

def fetch_worldbank_metrics(country: str):
    gdp_growth = fetch_worldbank_indicator(country, "NY.GDP.MKTP.KD.ZG")
    inflation = fetch_worldbank_indicator(country, "FP.CPI.TOTL.ZG")
    return {
        "gdp_growth": float(gdp_growth) if gdp_growth is not None else 0.0,
        "inflation": float(inflation) if inflation is not None else 0.0,
    }

# ================= Unstructured Event Integration =================
EVENT_RULES = [
    # pattern, impact_score (-1 risk↑, +1 risk↓), risk_tag
    (r"\b(debt|liability).*(restructur|refinanc|downgrad)\b", -0.8, "leverage"),
    (r"\b(default|insolvenc|bankrupt|chapter 11)\b", -1.0, "default"),
    (r"\b(liquidit|cash crunch|cash burn)\b", -0.7, "liquidity"),
    (r"\b(demand|sales|orders).*(declin|weak|slow)\b", -0.6, "demand"),
    (r"\b(cost|expense).*(spike|surge|rise)\b", -0.4, "margin"),
    (r"\b(capex|expansion|investment)\b", +0.2, "growth"),
    (r"\b(upgrad|rating upgrade)\b", +0.6, "rating"),
    (r"\b(downgrad|rating cut)\b", -0.8, "rating"),
    (r"\b(exec|CEO|CFO).*(resign|steps down)\b", -0.5, "governance"),
    (r"\b(secur|breach|cyber)\b", -0.4, "operational"),
]

def classify_event(text: str):
    t = text.lower()
    hits = []
    total = 0.0
    for pat, impact, tag in EVENT_RULES:
        if re.search(pat, t):
            hits.append({"pattern": pat, "impact": impact, "risk_tag": tag})
            total += impact
    return total, hits

def fetch_news_texts(query: str):
    key = os.getenv("NEWSAPI_KEY", "").strip()
    texts = []
    if key:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 20}
        headers = {"X-Api-Key": key}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            arts = r.json().get("articles", [])
            for a in arts:
                title = a.get("title","")
                desc = a.get("description","")
                texts.append({"headline": title, "summary": desc})
        except Exception:
            pass
    # Fallback to Reuters RSS titles
    if not texts:
        rss = "https://feeds.reuters.com/reuters/businessNews"
        try:
            rr = requests.get(rss, timeout=15)
            if rr.ok:
                lines = [line.strip() for line in rr.text.splitlines() if "<title>" in line]
                for ln in lines[:20]:
                    clean = re.sub(r"<.*?>", "", ln)
                    texts.append({"headline": clean, "summary": ""})
        except Exception:
            pass
    return texts

def compute_news_sentiment(texts):
    if not texts:
        return 0.0
    if VADER_OK:
        sid = SentimentIntensityAnalyzer()
        scores = [sid.polarity_scores((t.get("headline","")+" "+t.get("summary","")).strip())["compound"] for t in texts]
        return float(sum(scores)/len(scores))
    return 0.0

def extract_events_and_score(query: str):
    texts = fetch_news_texts(query)
    sentiment = compute_news_sentiment(texts)
    events = []
    agg = 0.0
    for t in texts[:15]:
        headline = (t.get("headline") or "").strip()
        if not headline:
            continue
        impact, hits = classify_event(headline)
        if abs(impact) > 0:
            events.append({"headline": headline, "impact": impact, "risk_matches": hits})
            agg += impact
    # Normalize event risk to approx [-1, +1]
    if events:
        agg = max(-2.0, min(2.0, agg)) / 2.0
    return sentiment, events, agg

# ================= Scoring, Persistence, Trends, Alerts =================
def score_from_features(feats: dict):
    x = np.array([float(feats.get(k, 0.0)) for k in FEATURES], dtype=float)
    z = _intercept + x @ _beta
    p = _sigmoid(z)
    score = int(300 + p*600)
    contrib = (x * _beta)
    ranked = sorted(
        [{"feature": f, "contribution": float(c)} for f,c in zip(FEATURES, contrib)],
        key=lambda d: abs(d["contribution"]), reverse=True
    )[:5]
    pos = [r["feature"] for r in ranked if r["contribution"]>0]
    neg = [r["feature"] for r in ranked if r["contribution"]<0]
    nl = ""
    if pos: nl += "Boosted by " + ", ".join(pos)
    if neg: nl += ("; " if nl else "") + "dragged by " + ", ".join(neg)
    if not nl: nl = "Balanced drivers with no dominant factor."
    return score, float(p), ranked, nl

def store_result(symbol: str, score: int, p: float, feats: dict, events: list):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO results(symbol, ts, score, probability, features, events) VALUES (?,?,?,?,?,?)",
                (symbol, int(time.time()), score, p, json.dumps(feats), json.dumps(events)))
    con.commit()
    con.close()

def load_history(symbol: str, limit: int = 100):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT ts, score FROM results WHERE symbol=? ORDER BY ts DESC LIMIT ?", (symbol, limit))
    rows = cur.fetchall()
    con.close()
    rows.reverse()
    return rows

def compute_trends(history):
    # history: list of (ts, score) oldest->newest
    if not history:
        return {"short_term": None, "long_term": None, "delta": None}
    scores = [s for _, s in history]
    short = sum(scores[-5:])/min(5, len(scores))
    long = sum(scores)/len(scores)
    delta = scores[-1] - (scores[-2] if len(scores)>1 else scores[-1])
    return {"short_term": short, "long_term": long, "delta": delta}

def generate_alerts(trends):
    alerts = []
    if trends["delta"] is not None and abs(trends["delta"]) >= 40:
        alerts.append({"type":"score_spike", "message": f"Sudden score change: {trends['delta']:+.0f} points"})
    if trends["short_term"] and trends["long_term"]:
        if trends["short_term"] < trends["long_term"] - 30:
            alerts.append({"type":"downtrend", "message":"Short-term trend below long-term average"})
    return alerts

# ================= API =================
@app.get("/health")
def health():
    return {"ok": True, "time": int(time.time())}

class ManualResp(BaseModel):
    score: int
    probability_good: float
    top_contributions: list
    explanation: str
    features_used: dict
    trends: dict | None = None
    alerts: list | None = None
    events: list | None = None
    sources: dict | None = None

@app.post("/score-manual", response_model=ManualResp)
def score_manual(payload: ManualPayload):
    feats = payload.features.copy()
    # Ensure event_risk present
    feats.setdefault("event_risk", 0.0)
    score, p, ranked, nl = score_from_features(feats)
    symbol = payload.symbol or "MANUAL"
    store_result(symbol, score, p, feats, [])
    hist = load_history(symbol)
    trends = compute_trends(hist)
    alerts = generate_alerts(trends)
    return ManualResp(
        score=score, probability_good=round(p,4),
        top_contributions=ranked, explanation=nl,
        features_used=feats, trends=trends, alerts=alerts, events=[],
        sources={}
    )

class LiveResp(ManualResp):
    pass

class LivePayload(BaseModel):
    symbol: str
    country: str
    query: str

@app.post("/score-live", response_model=LiveResp)
def score_live(payload: LivePayload):
    market = fetch_market_metrics(payload.symbol)
    macro = fetch_worldbank_metrics(payload.country)
    sentiment, events, event_risk = extract_events_and_score(payload.query)

    feats = {**market, **macro, "news_sentiment": float(sentiment), "event_risk": float(event_risk)}
    score, p, ranked, nl = score_from_features(feats)

    # Enrich explanation with event summary if any
    if events:
        # Take top magnitude event
        top = sorted(events, key=lambda e: abs(e["impact"]), reverse=True)[0]
        extra = " Recent event: " + top["headline"]
        nl = (nl + ";" if nl else "") + extra

    store_result(payload.symbol, score, p, feats, events)
    hist = load_history(payload.symbol)
    trends = compute_trends(hist)
    alerts = generate_alerts(trends)

    return LiveResp(
        score=score, probability_good=round(p,4),
        top_contributions=ranked, explanation=nl,
        features_used=feats, trends=trends, alerts=alerts,
        events=events,
        sources={"market": market, "macro": macro, "news_sentiment": sentiment}
    )
