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

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Pfade ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FINANCE_DIR  = BASE_DIR.parent.parent / "01_Finance"
REPORTS_DIR  = FINANCE_DIR / "reports"
LOGS_DIR     = FINANCE_DIR / "logs"
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


@app.get("/api/market")
async def get_market():
    """Letzter Risk + Monte Carlo Report."""
    risk = _latest_report("risk")
    mc   = _latest_report("mc")

    if not risk and not mc:
        raise HTTPException(404, "Noch kein Report vorhanden. Bitte predictor.py --dry-run ausfuehren.")

    return {
        "risk":         risk,
        "monte_carlo":  mc,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/signals")
async def get_signals():
    """Letzte Trading-Signale aus dem Trade-Log."""
    log_path = LOGS_DIR / "trades.json"
    if not log_path.exists():
        return {"signals": [], "message": "Noch keine Trades aufgezeichnet."}
    try:
        trades = json.loads(log_path.read_text(encoding="utf-8"))
        # Letzte 20
        recent = trades[-20:] if len(trades) > 20 else trades
        return {"signals": list(reversed(recent)), "total": len(trades)}
    except Exception as e:
        raise HTTPException(500, str(e))


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


# ── Flights / Kiwi Tequila ───────────────────────────────────────────────────

_KIWI_MOCK: list[dict] = [
    {"origin":"VIE","origin_name":"Wien","dest":"LIS","dest_name":"Lissabon","dest_country":"Portugal","dest_flag":"🇵🇹","price":89,"departure":"2026-06-12","return_date":"2026-06-19","nights":7,"stops":1,"airline":"Ryanair","deep_link":"https://www.kiwi.com/de/"},
    {"origin":"VIE","origin_name":"Wien","dest":"ATH","dest_name":"Athen","dest_country":"Griechenland","dest_flag":"🇬🇷","price":125,"departure":"2026-07-03","return_date":"2026-07-10","nights":7,"stops":0,"airline":"Aegean Airlines","deep_link":"https://www.kiwi.com/de/"},
    {"origin":"MUC","origin_name":"München","dest":"BCN","dest_name":"Barcelona","dest_country":"Spanien","dest_flag":"🇪🇸","price":178,"departure":"2026-06-20","return_date":"2026-06-27","nights":7,"stops":1,"airline":"Vueling","deep_link":"https://www.kiwi.com/de/"},
    {"origin":"SZG","origin_name":"Salzburg","dest":"PMI","dest_name":"Mallorca","dest_country":"Spanien","dest_flag":"🇪🇸","price":210,"departure":"2026-07-15","return_date":"2026-07-22","nights":7,"stops":0,"airline":"Wizz Air","deep_link":"https://www.kiwi.com/de/"},
    {"origin":"VIE","origin_name":"Wien","dest":"DUB","dest_name":"Dublin","dest_country":"Irland","dest_flag":"🇮🇪","price":247,"departure":"2026-08-01","return_date":"2026-08-08","nights":7,"stops":1,"airline":"Ryanair","deep_link":"https://www.kiwi.com/de/"},
]

_KIWI_BASE = "https://tequila.kiwi.com/v2/search"


def _country_flag(iso2: str) -> str:
    if not iso2 or len(iso2) != 2:
        return "🌍"
    return chr(0x1F1E6 + ord(iso2[0].upper()) - 65) + chr(0x1F1E6 + ord(iso2[1].upper()) - 65)


def _unix_to_date(ts: int) -> str:
    try:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _parse_kiwi(raw: dict) -> list[dict]:
    out = []
    for item in raw.get("data") or []:
        try:
            nights = item.get("nightsInDest") or 7
            dep_ts = item.get("dTime") or 0
            ret_ts = item.get("dTimeReturn") or (dep_ts + nights * 86400)
            country = item.get("countryTo") or {}
            iso2 = country.get("code", "")
            airlines = item.get("airlines") or []
            route = item.get("route") or []
            stops = max(0, len([r for r in route if r.get("flyFrom") != item.get("flyFrom")]) - 1)
            out.append({
                "origin":       item.get("flyFrom", ""),
                "origin_name":  item.get("cityFrom", ""),
                "dest":         item.get("flyTo", ""),
                "dest_name":    item.get("cityTo", ""),
                "dest_country": country.get("name", ""),
                "dest_flag":    _country_flag(iso2),
                "price":        int(item.get("price") or 0),
                "departure":    _unix_to_date(dep_ts),
                "return_date":  _unix_to_date(ret_ts),
                "nights":       nights,
                "stops":        stops,
                "airline":      ", ".join(dict.fromkeys(airlines)) if airlines else "",
                "deep_link":    item.get("deep_link", "https://www.kiwi.com/de/"),
            })
        except Exception:
            continue
    return out


@app.get("/api/flights/deals")
async def flights_deals(origins: str = "VIE,SZG,MUC"):
    """Günstigste Round-Trip Deals ab VIE/SZG/MUC via Kiwi Tequila."""
    key = os.environ.get("KIWI_API_KEY", "")
    if not key:
        return {"deals": _KIWI_MOCK, "mock": True, "count": len(_KIWI_MOCK)}

    today = datetime.now()
    params = {
        "fly_from": origins, "fly_to": "anywhere",
        "date_from": today.strftime("%d/%m/%Y"),
        "date_to": (today + timedelta(days=180)).strftime("%d/%m/%Y"),
        "flight_type": "round",
        "nights_in_dst_from": 3, "nights_in_dst_to": 21,
        "max_stopovers": 2, "curr": "EUR",
        "sort": "price", "limit": 30, "one_for_city": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(_KIWI_BASE, params=params, headers={"apikey": key})
            r.raise_for_status()
        deals = _parse_kiwi(r.json())
        return {"deals": deals, "mock": False, "count": len(deals)}
    except Exception as e:
        logger.error(f"Kiwi deals: {e}")
        return {"deals": _KIWI_MOCK, "mock": True, "count": len(_KIWI_MOCK), "error": str(e)}


@app.get("/api/flights/search")
async def flights_search(
    fly_from: str = "VIE",
    fly_to: str = "anywhere",
    date_from: str = "",
    date_to: str = "",
    nights_min: int = 3,
    nights_max: int = 14,
    max_stopovers: int = 2,
):
    """Individuelle Flugsuche via Kiwi Tequila."""
    key = os.environ.get("KIWI_API_KEY", "")
    if not key:
        return {"deals": _KIWI_MOCK, "mock": True, "count": len(_KIWI_MOCK)}

    today = datetime.now()
    params = {
        "fly_from": fly_from, "fly_to": fly_to,
        "date_from": date_from or today.strftime("%d/%m/%Y"),
        "date_to": date_to or (today + timedelta(days=90)).strftime("%d/%m/%Y"),
        "flight_type": "round",
        "nights_in_dst_from": nights_min, "nights_in_dst_to": nights_max,
        "max_stopovers": max_stopovers, "curr": "EUR",
        "sort": "price", "limit": 30, "one_for_city": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(_KIWI_BASE, params=params, headers={"apikey": key})
            r.raise_for_status()
        deals = _parse_kiwi(r.json())
        return {"deals": deals, "mock": False, "count": len(deals)}
    except Exception as e:
        logger.error(f"Kiwi search: {e}")
        raise HTTPException(502, f"Kiwi API Fehler: {e}")


@app.get("/api/flights/map")
async def flights_map(fly_from: str = "VIE,SZG,MUC"):
    """Günstigster Flug je Zielstadt für Karten-Arcs."""
    key = os.environ.get("KIWI_API_KEY", "")
    if not key:
        return {"deals": _KIWI_MOCK, "mock": True, "count": len(_KIWI_MOCK)}

    today = datetime.now()
    params = {
        "fly_from": fly_from, "fly_to": "anywhere",
        "date_from": today.strftime("%d/%m/%Y"),
        "date_to": (today + timedelta(days=180)).strftime("%d/%m/%Y"),
        "flight_type": "round",
        "nights_in_dst_from": 3, "nights_in_dst_to": 21,
        "max_stopovers": 2, "curr": "EUR",
        "sort": "price", "limit": 50, "one_for_city": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(_KIWI_BASE, params=params, headers={"apikey": key})
            r.raise_for_status()
        deals = _parse_kiwi(r.json())
        return {"deals": deals, "mock": False, "count": len(deals)}
    except Exception as e:
        logger.error(f"Kiwi map: {e}")
        return {"deals": _KIWI_MOCK, "mock": True, "count": len(_KIWI_MOCK), "error": str(e)}


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
