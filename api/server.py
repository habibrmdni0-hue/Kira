"""
api/server.py — Kira FastAPI server.

Endpoints:
  POST /chat            Reactive flow (orchestrator)
  GET  /proactive/{id}  Scheduled proactive check
  POST /scan-receipt    Receipt OCR (multipart image upload)
  GET  /business/{id}   Business summary + today's P&L
  GET  /health          Status check

Static files served from api/static/; root / serves index.html.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import AgentRequest, BookkeeperAgent, EyesAgent
from config.settings import settings
from data.firestore_client import firestore_status, get_business_state
from orchestrator import KiraOrchestrator, KiraRequest, run_proactive_check

# ── Singletons (built once at startup, reused per request) ──────
_orchestrator = KiraOrchestrator()
_eyes         = EyesAgent()
_bookkeeper   = BookkeeperAgent()

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(title="Kira API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_STATIC / "index.html")


# ── POST /chat ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    language: str = "id"


@app.post("/chat")
def chat(body: ChatRequest) -> Dict[str, Any]:
    try:
        result = _orchestrator.run(KiraRequest(
            payload=body.message,
            user_id=body.user_id,
            language=body.language,
        ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    dev_note = (
        result.get("agent_results", {})
        .get("voice_agent", {})
        .get("result", {})
        .get("_dev_note", "")
    )
    return {
        "response":       result["final_response"],
        "agents_invoked": result["agents_invoked"],
        "dev_note":       dev_note,
    }


# ── GET /proactive/{user_id} ─────────────────────────────────────

@app.get("/proactive/{user_id}")
def proactive(user_id: str) -> Dict[str, Any]:
    try:
        alerts = run_proactive_check(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if alerts:
        business_name = alerts[0].get("business", user_id)
    else:
        state = get_business_state(user_id)
        business_name = state["business_name"] if state else user_id

    return {"business_name": business_name, "alerts": alerts}


# ── POST /scan-receipt ───────────────────────────────────────────

@app.post("/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await file.read()
    b64 = base64.b64encode(raw).decode()

    req = AgentRequest(
        payload=b64,
        user_id="guest",
        language="id",
        input_type="image",
    )
    try:
        response = _eyes.handle(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    r = response.result
    return {
        "tier_used":  r.get("source_tier"),
        "confidence": r.get("confidence"),
        "items":      r.get("items", []),
        "total":      r.get("total", 0),
        "raw_text":   r.get("raw_text", ""),
    }


# ── GET /business/{user_id} ──────────────────────────────────────

@app.get("/business/{user_id}")
def business(user_id: str) -> Dict[str, Any]:
    state = get_business_state(user_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No data for {user_id!r}")

    # P&L from bookkeeper
    bk_req = AgentRequest(
        payload="",
        user_id=user_id,
        language=state.get("language", "id"),
    )
    fin = _bookkeeper.handle(bk_req).result.get("financials", {})

    # Items with ≤ 3 days of stock (high + critical severity)
    inventory = state.get("inventory", [])
    critical_stock = sum(
        1 for item in inventory
        if item["daily_usage"] > 0
        and (item["stock"] / item["daily_usage"]) <= 3
    )

    # Products with negative gross margin
    losing = sum(
        1 for sale in state.get("recent_sales_7d", [])
        if sale["revenue"] > 0
        and (sale["revenue"] - sale["cost"]) / sale["revenue"] < 0
    )

    return {
        "business_name":      state["business_name"],
        "cash_balance":       state["cash_balance"],
        "avg_daily_sales":    state["avg_daily_sales"],
        "inventory_count":    len(inventory),
        "critical_stock_count": critical_stock,
        "losing_product_count": losing,
        "net_profit":         fin.get("net_profit", 0),
        "gross_margin_pct":   fin.get("gross_margin_pct", 0.0),
    }


# ── GET /health ──────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status":    "ok",
        "mode":      "mock" if settings.mock_mode else "live",
        "firestore": firestore_status(),
    }


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
