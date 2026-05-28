"""
Jarvis Cloud API — FastAPI Backend
===================================
Endpoints:
  GET  /health              → Status-Check
  GET  /api/market          → Letzten Risk + MC Report
  GET  /api/portfolio       → Alpaca Account + Positionen
  POST /api/trade           → Market-Order abschicken
  POST /api/chat            → Jarvis LLM-Chat
  GET  /api/signals         → Letzte Trading-Signale aus Log
  WS   /ws/jarvis           → WebSocket fuer Echtzeit-Chat + Voice
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import uuid

import asyncio

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Pfade ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FINANCE_DIR  = Path(os.environ.get("FINANCE_DIR", str(BASE_DIR.parent.parent / "01_Finance")))
REPORTS_DIR  = Path(os.environ.get("REPORTS_DIR", str(BASE_DIR / "reports")))
LOGS_DIR     = FINANCE_DIR / "logs"
# Sicherstellen dass reports-Verzeichnis existiert
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"

# ── .env laden ──────────────────────────────────────────────────────────────
def _load_env():
    for env_path in [BASE_DIR / ".env", FINANCE_DIR / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()

_load_env()

# ── Session memory ────────────────────────────────────────────────────────────
_sessions: dict[str, list] = {}
_MAX_HISTORY = 20

# ── FastAPI App ─────────────────────────────────────────────────────────────
APP_START_TIME = __import__("time").time()
app = FastAPI(
    title="Jarvis API",
    description="The Predictor + Jarvis Cloud Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JarvisAPI")


# ── Pydantic Models ──────────────────────────────────────────────────────────
class TradeRequest(BaseModel):
    symbol: str
    qty: int
    side: str          # "buy" | "sell"
    confirmed: bool = False

class ChatRequest(BaseModel):
    message: str
    lang: str = "de"
    session_id: str = ""


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────
def _latest_report(prefix: str) -> Optional[dict]:
    if not REPORTS_DIR.exists():
        return None
    files = sorted(
        REPORTS_DIR.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_alpaca_client():
    try:
        from alpaca.trading.client import TradingClient
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        paper  = os.environ.get("PAPER_TRADING", "true").lower() != "false"
        if not key or not secret:
            return None, paper
        return TradingClient(key, secret, paper=paper), paper
    except Exception:
        return None, True


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "time": datetime.now().isoformat(),
        "version": "1.0.0",
        "finance_dir": str(FINANCE_DIR),
        "alpaca_key_set": bool(os.environ.get("ALPACA_API_KEY")),
    }


@app.get("/__version__")
async def version():
    """Build-Version für Auto-Refresh-Detection im Frontend."""
    import hashlib, time
    # Stable per Deploy: Hash aus Startup-Zeit (ändert sich bei jedem Railway-Deploy)
    build_id = os.environ.get("RAILWAY_DEPLOYMENT_ID",
                os.environ.get("RAILWAY_REVISION",
                str(int(APP_START_TIME))))
    return {"version": build_id, "time": datetime.now().isoformat()}


def _generate_live_report() -> dict:
    """Generiert einen Live-Marktreport direkt via yfinance + Alpaca wenn kein Cache vorhanden."""
    import random, math
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "BTC-USD"]
    prices  = {}
    try:
        import yfinance as yf
        for sym in symbols:
            t = yf.Ticker(sym)
            h = t.history(period="1d")
            if not h.empty:
                prices[sym] = round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        pass

    # Einfacher MC-Proxy: Zufallspfade mit historischen Parametern
    params = {
        "AAPL":    (0.20, 0.25), "MSFT":  (0.17, 0.20),
        "GOOGL":   (0.18, 0.22), "SPY":   (0.12, 0.15),
        "BTC-USD": (0.50, 0.70),
    }
    start_val = 100_000
    sims = 1000
    horizon = 252 * 5  # 5 Jahre
    portfolio_mu  = sum(mu  for mu, _  in params.values()) / len(params)
    portfolio_sig = sum(sig for _, sig in params.values()) / len(params)
    dt = 1 / 252
    end_vals = []
    for _ in range(sims):
        v = start_val
        for _ in range(horizon):
            v *= math.exp((portfolio_mu - 0.5 * portfolio_sig**2) * dt
                          + portfolio_sig * math.sqrt(dt) * random.gauss(0, 1))
        end_vals.append(v)
    end_vals.sort()
    median = end_vals[sims // 2]
    var5   = end_vals[int(sims * 0.05)]
    best95 = end_vals[int(sims * 0.95)]

    mc_report = {
        "portfolio": "Live-Report (Railway)",
        "date": datetime.now().isoformat(),
        "iterations": sims,
        "horizon_years": 5,
        "start_value": start_val,
        "median_end_value": round(median, 0),
        "best_case_95": round(best95, 0),
        "worst_case_5": round(var5, 0),
        "expected_cagr": round((median / start_val) ** (1/5) - 1, 4),
        "loss_probability": round(sum(1 for v in end_vals if v < start_val) / sims, 4),
        "live_prices": prices,
        "note": "Live-generiert auf Railway (kein lokaler Cache)"
    }

    risk_report = {
        "portfolio": "Live-Report (Railway)",
        "date": datetime.now().isoformat(),
        "status": "LIVE",
        "portfolio_value": start_val,
        "var_1d_95_pct": round(start_val * portfolio_sig / math.sqrt(252), 2),
        "live_prices": prices,
        "alerts": [],
        "note": "Live-generiert auf Railway"
    }

    # Cache für spätere Calls
    try:
        import json as _json
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (REPORTS_DIR / f"mc_{ts}.json").write_text(_json.dumps(mc_report), encoding="utf-8")
        (REPORTS_DIR / f"risk_{ts}.json").write_text(_json.dumps(risk_report), encoding="utf-8")
    except Exception:
        pass

    return {"risk": risk_report, "monte_carlo": mc_report}


@app.get("/api/market")
async def get_market():
    """Letzter Risk + Monte Carlo Report. Generiert Live-Daten wenn kein Cache vorhanden."""
    risk = _latest_report("risk")
    mc   = _latest_report("mc")

    if not risk and not mc:
        # Live-Report on-demand generieren
        try:
            live = _generate_live_report()
            return {**live, "generated_at": datetime.now().isoformat(), "source": "live"}
        except Exception as e:
            raise HTTPException(503, f"Kein Report vorhanden und Live-Generierung fehlgeschlagen: {e}")

    return {
        "risk":         risk,
        "monte_carlo":  mc,
        "generated_at": datetime.now().isoformat(),
        "source":       "cache",
    }


def _generate_live_signals() -> list:
    """
    Generiert Live-Signale via Alpaca Data API (Stocks) + CoinGecko (BTC).
    Alpaca ist auf Railway garantiert erreichbar (Keys bereits konfiguriert).
    """
    import os, datetime, math

    ALPACA_KEY    = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
    STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "SPY"]
    signals = []

    # ── 1) Alpaca Bars für US-Aktien ──────────────────────────────────────────
    if ALPACA_KEY and ALPACA_SECRET:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests   import StockBarsRequest
            from alpaca.data.timeframe  import TimeFrame

            client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            end_dt   = datetime.datetime.utcnow()
            start_dt = end_dt - datetime.timedelta(days=35)

            req  = StockBarsRequest(
                symbol_or_symbols=STOCK_SYMBOLS,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )
            bars_data = client.get_stock_bars(req)

            for sym in STOCK_SYMBOLS:
                try:
                    raw = bars_data[sym] if sym in bars_data else []
                    if not raw or len(raw) < 5:
                        continue
                    closes = [float(b.close) for b in raw]
                    price  = closes[-1]
                    sma20  = sum(closes[-20:]) / min(len(closes), 20)
                    ret5d  = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
                    ret1d  = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0

                    score = 0
                    if price > sma20 * 1.01:  score += 1
                    if ret5d > 0.015:          score += 1
                    if ret1d > 0.005:          score += 1
                    if price < sma20 * 0.99:  score -= 1
                    if ret5d < -0.015:         score -= 1
                    if ret1d < -0.005:         score -= 1

                    if score >= 2:
                        side, label = "buy",  "KAUFEN"
                    elif score <= -2:
                        side, label = "sell", "VERKAUFEN"
                    else:
                        side, label = ("buy" if ret1d >= 0 else "sell"), "HALTEN"

                    signals.append({
                        "symbol":    sym,
                        "side":      side,
                        "action":    label,
                        "price":     round(price, 2),
                        "target":    round(price * 1.08, 2),
                        "stop":      round(price * 0.96, 2),
                        "sma20":     round(sma20, 2),
                        "ret_1d":    round(ret1d * 100, 2),
                        "ret_5d":    round(ret5d * 100, 2),
                        "confidence":min(95, max(40, 50 + abs(score) * 15)),
                        "strategy":  "SMA20 + Momentum (Alpaca)",
                        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "source":    "alpaca_data",
                    })
                except Exception:
                    pass
        except Exception as e:
            pass  # Alpaca fehlgeschlagen → weiter zu yfinance

    # ── 2) yfinance als Fallback für fehlende Symbole ─────────────────────────
    found_syms = {s["symbol"] for s in signals}
    missing    = [s for s in STOCK_SYMBOLS if s not in found_syms]
    if missing:
        try:
            import yfinance as yf
            for sym in missing:
                try:
                    h = yf.Ticker(sym).history(period="30d")
                    if h.empty or len(h) < 5:
                        continue
                    closes = h["Close"].tolist()
                    price  = closes[-1]
                    sma20  = sum(closes[-20:]) / min(len(closes), 20)
                    ret5d  = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
                    ret1d  = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
                    score  = sum([price > sma20*1.01, ret5d > 0.015, ret1d > 0.005]) - \
                             sum([price < sma20*0.99, ret5d < -0.015, ret1d < -0.005])
                    side   = "buy" if score >= 0 else "sell"
                    signals.append({
                        "symbol": sym, "side": side,
                        "action": "KAUFEN" if score >= 2 else ("VERKAUFEN" if score <= -2 else "HALTEN"),
                        "price": round(price, 2), "target": round(price*1.08, 2),
                        "stop": round(price*0.96, 2), "sma20": round(sma20, 2),
                        "ret_1d": round(ret1d*100, 2), "ret_5d": round(ret5d*100, 2),
                        "confidence": min(95, max(40, 50 + abs(score)*15)),
                        "strategy": "SMA20 + Momentum (yfinance)",
                        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "source": "yfinance",
                    })
                except Exception:
                    pass
        except ImportError:
            pass

    # ── 3) BTC via CoinGecko (kein API-Key nötig) ────────────────────────────
    try:
        import httpx
        r = httpx.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "30", "interval": "daily"},
            timeout=10,
        )
        if r.status_code == 200:
            prices_raw = r.json().get("prices", [])
            if len(prices_raw) >= 5:
                closes = [p[1] for p in prices_raw]
                price  = closes[-1]
                sma20  = sum(closes[-20:]) / min(len(closes), 20)
                ret5d  = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
                ret1d  = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
                score  = sum([price > sma20*1.01, ret5d > 0.015, ret1d > 0.005]) - \
                         sum([price < sma20*0.99, ret5d < -0.015, ret1d < -0.005])
                side   = "buy" if score >= 0 else "sell"
                signals.append({
                    "symbol": "BTC", "side": side,
                    "action": "KAUFEN" if score >= 2 else ("VERKAUFEN" if score <= -2 else "HALTEN"),
                    "price": round(price, 2), "target": round(price*1.08, 2),
                    "stop": round(price*0.96, 2), "sma20": round(sma20, 2),
                    "ret_1d": round(ret1d*100, 2), "ret_5d": round(ret5d*100, 2),
                    "confidence": min(95, max(40, 50 + abs(score)*15)),
                    "strategy": "SMA20 + Momentum (CoinGecko)",
                    "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": "coingecko",
                })
    except Exception:
        pass

    return signals


@app.get("/api/debug/signals")
async def debug_signals():
    """Debug: testet alle Datenquellen — zeigt genau was auf Railway funktioniert."""
    import sys, os
    result = {
        "python":          sys.version,
        "alpaca_key_set":  bool(os.environ.get("ALPACA_API_KEY")),
        "alpaca_data":     None,
        "yfinance":        None,
        "coingecko":       None,
        "live_signals_count": 0,
        "live_signals_error": None,
    }
    # Test Alpaca Data
    try:
        import datetime
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame
        client = StockHistoricalDataClient(
            os.environ.get("ALPACA_API_KEY",""),
            os.environ.get("ALPACA_SECRET_KEY",""),
        )
        req = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Day,
            start=datetime.datetime.utcnow()-datetime.timedelta(days=10),
            end=datetime.datetime.utcnow(),
        )
        bars = client.get_stock_bars(req)
        raw  = bars["AAPL"] if "AAPL" in bars else []
        result["alpaca_data"] = {"ok": True, "bars": len(raw),
                                  "latest": round(float(raw[-1].close),2) if raw else None}
    except Exception as e:
        result["alpaca_data"] = {"ok": False, "error": str(e)}
    # Test yfinance
    try:
        import yfinance as yf
        h = yf.Ticker("AAPL").history(period="5d")
        result["yfinance"] = {"ok": not h.empty, "rows": len(h),
                               "latest": round(float(h["Close"].iloc[-1]),2) if not h.empty else None}
    except Exception as e:
        result["yfinance"] = {"ok": False, "error": str(e)}
    # Test CoinGecko
    try:
        import httpx
        r = httpx.get("https://api.coingecko.com/api/v3/simple/price",
                       params={"ids":"bitcoin","vs_currencies":"usd"}, timeout=8)
        result["coingecko"] = {"ok": r.status_code==200,
                                "btc_usd": r.json().get("bitcoin",{}).get("usd") if r.status_code==200 else None}
    except Exception as e:
        result["coingecko"] = {"ok": False, "error": str(e)}
    # Test full signal generation
    try:
        sigs = _generate_live_signals()
        result["live_signals_count"] = len(sigs)
    except Exception as e:
        result["live_signals_error"] = str(e)
    return result


@app.get("/api/portfolio")
async def get_portfolio():
    """Alpaca Account-Status + Positionen."""
    client, paper = _get_alpaca_client()
    if not client:
        raise HTTPException(503, "Alpaca nicht verbunden — API Keys pruefen.")

    try:
        acct      = client.get_account()
        positions = client.get_all_positions()

        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol":       p.symbol,
                "qty":          float(p.qty),
                "avg_entry":    float(p.avg_entry_price),
                "current_price":float(p.current_price) if p.current_price else 0,
                "market_value": float(p.market_value) if p.market_value else 0,
                "unrealized_pl":float(p.unrealized_pl) if p.unrealized_pl else 0,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0,
            })

        return {
            "account": {
                "status":          str(acct.status),
                "portfolio_value": float(acct.portfolio_value),
                "cash":            float(acct.cash),
                "buying_power":    float(acct.buying_power),
                "equity":          float(acct.equity),
                "last_equity":     float(acct.last_equity),
                "day_pl":          float(acct.equity) - float(acct.last_equity),
                "paper":           paper,
            },
            "positions": pos_list,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/trade")
async def submit_trade(req: TradeRequest):
    """Market-Order abschicken (nur mit confirmed=true)."""
    if not req.confirmed:
        # Vorschau ohne Ausfuehrung
        client, paper = _get_alpaca_client()
        if not client:
            raise HTTPException(503, "Alpaca nicht verbunden.")
        try:
            sys.path.insert(0, str(FINANCE_DIR))
            from market_data import MarketData
            md    = MarketData()
            price = md.get_current_prices([req.symbol]).get(req.symbol, 0)
            return {
                "preview":   True,
                "symbol":    req.symbol,
                "qty":       req.qty,
                "side":      req.side,
                "est_price": price,
                "est_total": price * req.qty,
                "paper":     paper,
                "message":   f"Sende erneut mit confirmed=true um die Order auszufuehren.",
            }
        except Exception as e:
            raise HTTPException(500, str(e))

    # Order ausfuehren
    client, paper = _get_alpaca_client()
    if not client:
        raise HTTPException(503, "Alpaca nicht verbunden.")

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums    import OrderSide, TimeInForce

        order = client.submit_order(MarketOrderRequest(
            symbol        = req.symbol.upper(),
            qty           = req.qty,
            side          = OrderSide.BUY if req.side == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        ))
        return {
            "success":  True,
            "order_id": str(order.id),
            "symbol":   order.symbol,
            "qty":      float(order.qty),
            "side":     str(order.side),
            "status":   str(order.status),
            "paper":    paper,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/chat")
async def jarvis_chat(req: ChatRequest):
    """Jarvis LLM-Chat with session memory and Tavily MCP (search + extract + crawl)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

        # ── Session management ────────────────────────────────────────────────
        sid = req.session_id.strip() or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = []
        history = _sessions[sid]

        history.append({"role": "user", "content": req.message})
        working_history = history[-(_MAX_HISTORY * 2):]

        # ── System prompt — JARVIS personality ───────────────────────────────
        risk = _latest_report("risk")
        risk_context = ""
        if risk:
            status = risk.get("overall_status", "?")
            value  = risk.get("portfolio_value", 0)
            risk_context = f" Current portfolio: ${value:,.0f}, risk status: {status}."

        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        system = (
            "You are JARVIS — Just A Rather Very Intelligent System — the personal AI of "
            "Herr Florian Schiffer, quantitative trader and AI engineer. "
            "Personality: British precision, dry wit, unfailingly polite, iron-logical. "
            "Address Florian as 'Herr Schiffer' or 'sir'. Never be verbose when brevity serves. "
            "You have access to his trading system (The Predictor) and portfolio."
            + (risk_context if risk_context else "")
            + (
                " You are connected to the internet via Tavily. Use the search tool for "
                "current events, prices, and news. Use the extract tool whenever the user "
                "provides a URL — read the page and summarise it. Cite sources as 'Source: URL'."
                if tavily_key else
                " Note: internet access is currently offline."
            )
            + " Respond in German unless the user writes in English. Be concise."
        )

        # ── Call Claude (with or without Tavily MCP) ─────────────────────────
        mcp_used = False

        if tavily_key:
            tavily_mcp_url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_key}"
            resp = client.beta.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 1024,
                system     = system,
                messages   = working_history,
                mcp_servers = [
                    {
                        "type": "url",
                        "url":  tavily_mcp_url,
                        "name": "tavily",
                    }
                ],
                tools = [
                    {
                        "type":            "mcp_toolset",
                        "mcp_server_name": "tavily",
                    }
                ],
                betas = ["mcp-client-2025-11-20"],
            )
            reply = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    reply = block.text
                elif getattr(block, "type", None) == "mcp_tool_use":
                    mcp_used = True
            if not reply:
                reply = "I do apologise, sir — the response contained no text output."
        else:
            resp = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 800,
                system     = system,
                messages   = working_history,
            )
            reply = resp.content[0].text

        history.append({"role": "assistant", "content": reply})
        if len(history) > _MAX_HISTORY * 2:
            _sessions[sid] = history[-(_MAX_HISTORY * 2):]

        return {"reply": reply, "session_id": sid, "web_search": mcp_used}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": f"I do apologise, sir — a technical difficulty: {e}"}


# ── Flights / Aviasales (Travelpayouts) ──────────────────────────────────────

_AV_BASE     = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
_AV_MARKER   = "730327"
_AV_NO_TOKEN = (
    "AVIASALES_TOKEN nicht gesetzt — kostenlos registrieren unter "
    "travelpayouts.com und Token als Railway Variable AVIASALES_TOKEN eintragen"
)

# IATA → German city name
_IATA_CITY: dict[str, str] = {
    "VIE":"Wien","SZG":"Salzburg","LNZ":"Linz","GRZ":"Graz","INN":"Innsbruck",
    "MUC":"München","FRA":"Frankfurt","BER":"Berlin","HAM":"Hamburg",
    "DUS":"Düsseldorf","CGN":"Köln","STR":"Stuttgart","NUE":"Nürnberg",
    "ZRH":"Zürich","GVA":"Genf","BSL":"Basel",
    "LIS":"Lissabon","OPO":"Porto","FAO":"Faro",
    "ATH":"Athen","SKG":"Thessaloniki","HER":"Heraklion","RHO":"Rhodos",
    "CFU":"Korfu","JMK":"Mykonos","KGS":"Kos","CHQ":"Chania","ZTH":"Zakynthos",
    "BCN":"Barcelona","MAD":"Madrid","AGP":"Málaga","VLC":"Valencia",
    "SVQ":"Sevilla","PMI":"Mallorca","IBZ":"Ibiza","ALC":"Alicante",
    "TFS":"Teneriffa","LPA":"Gran Canaria","FUE":"Fuerteventura","ACE":"Lanzarote",
    "FCO":"Rom","MXP":"Mailand","VCE":"Venedig","BLQ":"Bologna",
    "NAP":"Neapel","CTA":"Catania","PSA":"Pisa",
    "CDG":"Paris","ORY":"Paris Orly","NCE":"Nizza","MRS":"Marseille",
    "TLS":"Toulouse","LYS":"Lyon","NTE":"Nantes","BOD":"Bordeaux",
    "LHR":"London","LGW":"London Gatwick","STN":"London Stansted",
    "MAN":"Manchester","EDI":"Edinburgh","BHX":"Birmingham","GLA":"Glasgow",
    "DUB":"Dublin",
    "BRU":"Brüssel","AMS":"Amsterdam","EIN":"Eindhoven","RTM":"Rotterdam",
    "CPH":"Kopenhagen","ARN":"Stockholm","OSL":"Oslo","HEL":"Helsinki",
    "RIX":"Riga","TLL":"Tallinn","VNO":"Vilnius",
    "WAW":"Warschau","KRK":"Krakau","PRG":"Prag","BUD":"Budapest","BTS":"Bratislava",
    "LJU":"Ljubljana","BEG":"Belgrad","DBV":"Dubrovnik","SPU":"Split",
    "ZAD":"Zadar","TGD":"Podgorica","SOF":"Sofia","OTP":"Bukarest","KBP":"Kiew",
    "IST":"Istanbul","SAW":"Istanbul Sabiha","AYT":"Antalya","DLM":"Dalaman",
    "BJV":"Bodrum","ADB":"Izmir","ESB":"Ankara",
    "TUN":"Tunis","RAK":"Marrakesch","CMN":"Casablanca","AGA":"Agadir",
    "HRG":"Hurghada","SSH":"Sharm el-Sheikh","CAI":"Kairo",
    "DXB":"Dubai","DOH":"Doha","AUH":"Abu Dhabi","MCT":"Muskat",
    "BKK":"Bangkok","SIN":"Singapur","HKG":"Hongkong","NRT":"Tokio",
    "ICN":"Seoul","JFK":"New York","LAX":"Los Angeles","MIA":"Miami",
    "ORD":"Chicago","YYZ":"Toronto","SYD":"Sydney","MEL":"Melbourne",
    "JNB":"Johannesburg","CPT":"Kapstadt",
}

# IATA → ISO-2 country code (for flag emoji)
_IATA_ISO2: dict[str, str] = {
    "VIE":"AT","SZG":"AT","LNZ":"AT","GRZ":"AT","INN":"AT",
    "MUC":"DE","FRA":"DE","BER":"DE","HAM":"DE","DUS":"DE",
    "CGN":"DE","STR":"DE","NUE":"DE",
    "ZRH":"CH","GVA":"CH","BSL":"CH",
    "LIS":"PT","OPO":"PT","FAO":"PT",
    "ATH":"GR","SKG":"GR","HER":"GR","RHO":"GR","CFU":"GR",
    "JMK":"GR","KGS":"GR","CHQ":"GR","ZTH":"GR",
    "BCN":"ES","MAD":"ES","AGP":"ES","VLC":"ES","SVQ":"ES",
    "PMI":"ES","IBZ":"ES","ALC":"ES","TFS":"ES","LPA":"ES","FUE":"ES","ACE":"ES",
    "FCO":"IT","MXP":"IT","VCE":"IT","BLQ":"IT","NAP":"IT","CTA":"IT","PSA":"IT",
    "CDG":"FR","ORY":"FR","NCE":"FR","MRS":"FR","TLS":"FR","LYS":"FR","NTE":"FR","BOD":"FR",
    "LHR":"GB","LGW":"GB","STN":"GB","MAN":"GB","EDI":"GB","BHX":"GB","GLA":"GB",
    "DUB":"IE",
    "BRU":"BE","AMS":"NL","EIN":"NL","RTM":"NL",
    "CPH":"DK","ARN":"SE","OSL":"NO","HEL":"FI",
    "RIX":"LV","TLL":"EE","VNO":"LT",
    "WAW":"PL","KRK":"PL","PRG":"CZ","BUD":"HU","BTS":"SK",
    "LJU":"SI","BEG":"RS","DBV":"HR","SPU":"HR","ZAD":"HR",
    "TGD":"ME","SOF":"BG","OTP":"RO","KBP":"UA",
    "IST":"TR","SAW":"TR","AYT":"TR","DLM":"TR","BJV":"TR","ADB":"TR","ESB":"TR",
    "TUN":"TN","RAK":"MA","CMN":"MA","AGA":"MA",
    "HRG":"EG","SSH":"EG","CAI":"EG",
    "DXB":"AE","DOH":"QA","AUH":"AE","MCT":"OM",
    "BKK":"TH","SIN":"SG","HKG":"HK","NRT":"JP","ICN":"KR",
    "JFK":"US","LAX":"US","MIA":"US","ORD":"US","YYZ":"CA",
    "SYD":"AU","MEL":"AU","JNB":"ZA","CPT":"ZA",
}


def _flag(iso2: str) -> str:
    if not iso2 or len(iso2) != 2:
        return "🌍"
    return chr(0x1F1E6 + ord(iso2[0].upper()) - 65) + chr(0x1F1E6 + ord(iso2[1].upper()) - 65)


def _parse_aviasales(raw: dict, origin_hint: str = "") -> list[dict]:
    out = []
    for item in (raw.get("data") or []):
        try:
            dep_str  = item.get("departure_at", "")
            ret_str  = item.get("return_at", "")
            dep_date = dep_str[:10] if dep_str else ""
            ret_date = ret_str[:10] if ret_str else ""

            nights = 0
            if dep_date and ret_date:
                try:
                    d1 = datetime.strptime(dep_date, "%Y-%m-%d")
                    d2 = datetime.strptime(ret_date, "%Y-%m-%d")
                    nights = max(0, (d2 - d1).days)
                except Exception:
                    pass

            link = item.get("link", "")
            if link:
                sep    = "&" if "?" in link else "?"
                av_url = f"https://www.aviasales.com{link}{sep}marker={_AV_MARKER}"
            else:
                av_url = f"https://www.aviasales.com/?marker={_AV_MARKER}"

            origin_code = item.get("origin", origin_hint)
            dest_code   = item.get("destination", "")
            out.append({
                "origin":      origin_code,
                "origin_name": _IATA_CITY.get(origin_code, origin_code),
                "dest":        dest_code,
                "dest_name":   _IATA_CITY.get(dest_code, dest_code),
                "dest_flag":   _flag(_IATA_ISO2.get(dest_code, "")),
                "price":       int(item.get("price") or 0),
                "departure":   dep_date,
                "return_date": ret_date,
                "nights":      nights,
                "stops":       int(item.get("transfers") or 0),
                "airline":     item.get("airline", ""),
                "deep_link":   av_url,
            })
        except Exception:
            continue
    return out


async def _av_fetch(client: httpx.AsyncClient, token: str, base_params: dict, origin: str) -> list[dict]:
    """Single origin fetch — called in parallel by deals + map endpoints."""
    try:
        r = await client.get(
            _AV_BASE,
            params={**base_params, "origin": origin},
            headers={"X-Access-Token": token},
        )
        r.raise_for_status()
        return _parse_aviasales(r.json(), origin)
    except Exception as e:
        logger.warning(f"Aviasales {origin}: {e}")
        return []


@app.get("/api/flights/deals")
async def flights_deals(origins: str = "VIE,SZG,MUC"):
    """Günstigste Round-Trip Deals ab VIE/SZG/MUC via Aviasales (Travelpayouts)."""
    aviasales_token = os.environ.get("AVIASALES_TOKEN", "")
    if not aviasales_token:
        return {"deals": [], "count": 0, "error": _AV_NO_TOKEN}

    origin_list = [o.strip().upper() for o in origins.split(",") if o.strip()]
    base_params = {"currency": "EUR", "sorting": "price", "limit": 10, "one_way": "false"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            results = await asyncio.gather(
                *[_av_fetch(client, aviasales_token, base_params, o) for o in origin_list]
            )
        all_deals = sorted([d for chunk in results for d in chunk], key=lambda x: x["price"])
        return {"deals": all_deals, "count": len(all_deals)}
    except Exception as e:
        logger.error(f"Aviasales deals: {e}")
        return {"deals": [], "count": 0, "error": f"Aviasales API Fehler: {e}"}


@app.get("/api/flights/search")
async def flights_search(
    fly_from: str = "VIE",
    fly_to: str = "",
    date_from: str = "",    # YYYY-MM
    date_to: str = "",      # YYYY-MM (return month)
    nights_min: int = 3,
    nights_max: int = 14,
    max_stopovers: int = 2,
):
    """Individuelle Flugsuche via Aviasales (Travelpayouts)."""
    aviasales_token = os.environ.get("AVIASALES_TOKEN", "")
    if not aviasales_token:
        return {"deals": [], "count": 0, "error": _AV_NO_TOKEN}

    params: dict = {
        "origin":   fly_from.upper(),
        "currency": "EUR",
        "sorting":  "price",
        "limit":    30,
        "one_way":  "false",
    }
    if fly_to and fly_to.lower() not in ("anywhere", ""):
        params["destination"] = fly_to.upper()
    if date_from:
        params["departure_at"] = date_from[:7]   # YYYY-MM
    if date_to:
        params["return_at"] = date_to[:7]         # YYYY-MM

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(_AV_BASE, params=params, headers={"X-Access-Token": aviasales_token})
            r.raise_for_status()
        deals = _parse_aviasales(r.json(), fly_from)
        return {"deals": deals, "count": len(deals)}
    except Exception as e:
        logger.error(f"Aviasales search: {e}")
        return {"deals": [], "count": 0, "error": f"Aviasales API Fehler: {e}"}


@app.get("/api/flights/map")
async def flights_map(fly_from: str = "VIE,SZG,MUC"):
    """Günstigster Flug je Zielstadt für Karten-Arcs (Aviasales)."""
    aviasales_token = os.environ.get("AVIASALES_TOKEN", "")
    if not aviasales_token:
        return {"deals": [], "count": 0, "error": _AV_NO_TOKEN}

    origin_list = [o.strip().upper() for o in fly_from.split(",") if o.strip()]
    base_params = {"currency": "EUR", "sorting": "price", "limit": 20, "one_way": "false"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            results = await asyncio.gather(
                *[_av_fetch(client, aviasales_token, base_params, o) for o in origin_list]
            )
        all_deals = sorted([d for chunk in results for d in chunk], key=lambda x: x["price"])
        return {"deals": all_deals, "count": len(all_deals)}
    except Exception as e:
        logger.error(f"Aviasales map: {e}")
        return {"deals": [], "count": 0, "error": f"Aviasales API Fehler: {e}"}


# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/jarvis")
async def websocket_jarvis(ws: WebSocket):
    """Echtzeit-Chat + Streaming fuer die PWA."""
    await ws.accept()
    ws_sid = str(uuid.uuid4())
    try:
        while True:
            data = await ws.receive_json()
            msg  = data.get("message", "")
            sid  = data.get("session_id", ws_sid)

            # Schnell-Befehle
            if any(w in msg.lower() for w in ["marktbericht", "portfolio", "bericht"]):
                risk = _latest_report("risk")
                mc   = _latest_report("mc")
                if risk and mc:
                    status  = risk.get("overall_status", "?")
                    value   = risk.get("portfolio_value", 0)
                    median  = mc.get("median_end_value", 0)
                    cagr    = mc.get("expected_cagr", 0)
                    await ws.send_json({
                        "type": "market_report",
                        "status": status,
                        "portfolio_value": value,
                        "mc_median": median,
                        "mc_cagr": cagr,
                        "alerts": risk.get("alerts", []),
                    })
                else:
                    await ws.send_json({"type": "error", "message": "Noch kein Report vorhanden."})
            else:
                # LLM-Antwort
                resp = await jarvis_chat(ChatRequest(message=msg, session_id=sid))
                await ws.send_json({"type": "chat", "reply": resp["reply"]})

    except WebSocketDisconnect:
        pass


# ── Frontend servieren ───────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {"message": "Jarvis API online. Frontend noch nicht gebaut."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
