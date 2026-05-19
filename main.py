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
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    """Einfacher Jarvis LLM-Chat (sync)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

        # Kontext aus letztem Report
        risk = _latest_report("risk")
        risk_summary = ""
        if risk:
            status = risk.get("overall_status", "?")
            value  = risk.get("portfolio_value", 0)
            risk_summary = f"\nAktuelles Portfolio: ${value:,.0f}, Risiko-Status: {status}."

        system = (
            "Du bist Jarvis, ein intelligenter persoenlicher Assistent fuer Florian. "
            "Du hast Zugriff auf sein Trading-System (The Predictor) und sein Portfolio. "
            "Antworte praezise, freundlich und auf Deutsch. Keine langen Erklaerungen."
            + risk_summary
        )

        resp = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 500,
            system     = system,
            messages   = [{"role": "user", "content": req.message}],
        )
        return {"reply": resp.content[0].text}

    except Exception as e:
        return {"reply": f"Fehler: {e}"}


# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/jarvis")
async def websocket_jarvis(ws: WebSocket):
    """Echtzeit-Chat + Streaming fuer die PWA."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            msg  = data.get("message", "")

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
                resp = await jarvis_chat(ChatRequest(message=msg))
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
