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

import sys
import traceback

try:
    import base64
    import os
    from pathlib import Path
    from typing import Any, Dict

    import uvicorn
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from agents import AgentRequest, BookkeeperAgent, EyesAgent
    from config.settings import settings
    from data.firestore_client import (
        firestore_status,
        get_business_state,
        get_daily_sales_timeseries,
    )
    from data.forecast import forecast_business
    from orchestrator import KiraOrchestrator, KiraRequest, run_proactive_check

except Exception as _import_err:
    print(f"STARTUP ERROR: {_import_err}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── Singletons (built once at startup, reused per request) ──────
try:
    _orchestrator = KiraOrchestrator()
    _eyes         = EyesAgent()
    _bookkeeper   = BookkeeperAgent()
except Exception as _init_err:
    print(f"STARTUP ERROR (singleton init): {_init_err}", flush=True)
    traceback.print_exc()
    sys.exit(1)

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

    bk_req = AgentRequest(
        payload="",
        user_id=user_id,
        language=state.get("language", "id"),
    )
    fin = _bookkeeper.handle(bk_req).result.get("financials", {})

    inventory = state.get("inventory", [])
    sales_7d  = state.get("recent_sales_7d", [])

    critical_stock = sum(
        1 for item in inventory
        if item["daily_usage"] > 0
        and (item["stock"] / item["daily_usage"]) <= 3
    )

    losing = sum(
        1 for sale in sales_7d
        if sale["revenue"] > 0
        and (sale["revenue"] - sale["cost"]) / sale["revenue"] < 0
    )

    # Full per-item inventory with days_remaining (for inventory bar chart)
    inventory_items = [
        {
            "item":           item["item"],
            "unit":           item["unit"],
            "days_remaining": round(item["stock"] / item["daily_usage"], 1)
                              if item["daily_usage"] > 0 else None,
        }
        for item in inventory
    ]

    # Per-product gross margin breakdown (for margin bar chart)
    product_margins = [
        {
            "name":       sale["item"],
            "revenue_7d": sale["revenue"],
            "cost_7d":    sale["cost"],
            "margin_pct": round((sale["revenue"] - sale["cost"]) / sale["revenue"] * 100, 1)
                         if sale["revenue"] > 0 else 0.0,
        }
        for sale in sales_7d
    ]

    return {
        "business_name":        state["business_name"],
        "language":             state.get("language", "id"),
        "cash_balance":         state["cash_balance"],
        "avg_daily_sales":      state["avg_daily_sales"],
        "inventory_count":      len(inventory),
        "critical_stock_count": critical_stock,
        "losing_product_count": losing,
        "net_profit":           fin.get("net_profit", 0),
        "gross_margin_pct":     fin.get("gross_margin_pct", 0.0),
        "inventory_items":      inventory_items,
        "product_margins":      product_margins,
    }


# ── GET /forecast/{user_id} ─────────────────────────────────────

@app.get("/forecast/{user_id}")
def forecast(user_id: str) -> Dict[str, Any]:
    result = forecast_business(user_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No data for {user_id!r}")
    return result


# ── GET /sales-history/{user_id} ─────────────────────────────────

@app.get("/sales-history/{user_id}")
def sales_history(user_id: str) -> Dict[str, Any]:
    state   = get_business_state(user_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No data for {user_id!r}")
    history = get_daily_sales_timeseries(user_id)
    return {
        "user_id":       user_id,
        "business_name": state["business_name"],
        "days":          len(history),
        "history":       history,
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
