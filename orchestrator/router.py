"""
LLM-based intent router — classifies a user request into one or more
of Kira's six specialist agents.

In mock mode it uses keyword heuristics so the full system is testable
without an API key. In live mode it calls the router LLM.
"""
import json
import re
from typing import List, Tuple

from config.settings import settings


# ──────────────────────────────────────────────────────────────
# Routing table used by both the mock heuristic AND the LLM
# prompt so the two stay in sync.
# ──────────────────────────────────────────────────────────────
AGENT_DESCRIPTIONS = {
    "eyes_agent":       "Reading / OCR of receipts, invoices, handwritten notes, images",
    "bookkeeper_agent": "Profit & loss, revenue, expenses, cashflow, financial summaries",
    "inventory_agent":  "Stock levels, reorder, stockout prediction, supply tracking",
    "strategy_agent":   "Pricing advice, product performance, promotions, business strategy",
    "voice_agent":      "General conversation, greetings, simple questions, small talk",
    "reasoning_agent":  "Complex multi-step analysis, forecasting, cross-domain reasoning",
    "data_entry_agent": (
        "Recording or updating business data the owner is reporting as new "
        "fact — e.g. 'update stok gula jadi 5kg', 'stok minyak sekarang 2 liter'. "
        "Use ONLY when the owner is telling Kira new information to SAVE, "
        "never when they're just asking to view/check existing data."
    ),
}

_ROUTER_SYSTEM = """\
You are Kira's intent classifier for a warung (small shop) business assistant.
Given a user request, output ONLY a valid JSON object with two fields:
  "intent": a concise 3-7 word description of what the user wants
  "agents": a list of agent names from the allowed set that should handle it

Allowed agents and their responsibilities:
{agent_list}

Rules:
- Always include "voice_agent" when a conversational reply is needed.
- Use "reasoning_agent" only for complex cross-domain analysis.
- Multiple agents are fine when the request spans domains.
- "data_entry_agent" generates its own natural-language reply — use it ALONE
  (do not combine with "voice_agent") when the owner wants to record/update data.
- Output raw JSON only — no markdown, no explanation.
""".format(
    agent_list="\n".join(f'  {k}: {v}' for k, v in AGENT_DESCRIPTIONS.items())
)


def route(payload: str, language: str) -> Tuple[str, List[str]]:
    """
    Returns (intent_description, list_of_agent_names).
    Falls back to mock routing when MOCK_MODE=true or LLM call fails.
    """
    if settings.mock_mode:
        return _mock_route(payload, language)
    return _llm_route(payload, language)


# ──────────────────────────────────────────────────────────────
# Live LLM routing
# ──────────────────────────────────────────────────────────────

def _llm_route(payload: str, language: str) -> Tuple[str, List[str]]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        base_url=settings.router_base_url,
        api_key=settings.router_api_key,
        model=settings.router_model,
        temperature=0,
    )
    messages = [
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=f"Language: {language}\nRequest: {payload}"),
    ]
    raw = llm.invoke(messages).content

    # Strip markdown fences if the model added them
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    try:
        parsed = json.loads(raw)
        intent: str       = parsed.get("intent", "unknown intent")
        agents: List[str] = parsed.get("agents", ["voice_agent"])
        # Validate agent names
        valid = set(AGENT_DESCRIPTIONS.keys())
        agents = [a for a in agents if a in valid] or ["voice_agent"]
        return intent, agents
    except json.JSONDecodeError:
        # Graceful fallback — never crash the orchestrator on a bad router response
        return "unknown intent", ["voice_agent"]


# ──────────────────────────────────────────────────────────────
# Mock routing (keyword heuristics — no API key needed)
# ──────────────────────────────────────────────────────────────

_KEYWORD_MAP = [
    # Data-entry (write) — check BEFORE the read-only "stok" group below so
    # explicit write phrasing doesn't get routed to inventory_agent instead.
    (["update stok", "ubah stok", "ganti stok", "catat stok", "stok jadi",
      "stok sekarang", "set stok", "stok baru", "update stock", "set stock"],
     "record stock update", ["data_entry_agent"]),

    # Strategy/pricing — check BEFORE generic P&L so "produk rugi" + "saran" routes correctly
    (["strategi", "saran", "produk laris", "produk rugi", "harga jual", "promosi",
      "strategy", "pricing", "promotion", "best seller", "losing product", "advice"],
     "business strategy and pricing", ["strategy_agent", "reasoning_agent", "voice_agent"]),

    (["nota", "struk", "kuitansi", "foto", "gambar", "receipt", "invoice", "image", "scan"],
     "read document or receipt", ["eyes_agent", "bookkeeper_agent"]),

    (["laba", "untung", "pendapatan", "pengeluaran", "keuangan",
      "profit", "revenue", "expense", "cashflow", "financial"],
     "financial summary and P&L", ["bookkeeper_agent", "voice_agent"]),

    (["rugi", "loss"],
     "financial loss analysis", ["bookkeeper_agent", "strategy_agent", "voice_agent"]),

    (["stok", "persediaan", "barang habis", "restock", "pesan barang",
      "stock", "inventory", "reorder", "stockout", "supply"],
     "inventory and stock management", ["inventory_agent", "voice_agent"]),

    (["analisis", "prediksi", "forecast", "kenapa", "mengapa", "bandingkan",
      "analyze", "predict", "why", "compare", "deep dive"],
     "deep business analysis", ["reasoning_agent", "voice_agent"]),

    (["halo", "hai", "selamat", "apa kabar", "hello", "hi", "help", "bantuan"],
     "greeting and general conversation", ["voice_agent"]),
]


def _mock_route(payload: str, language: str) -> Tuple[str, List[str]]:
    payload_lower = payload.lower()
    for keywords, intent, agents in _KEYWORD_MAP:
        if any(kw in payload_lower for kw in keywords):
            return intent, agents
    return "general conversation", ["voice_agent"]
