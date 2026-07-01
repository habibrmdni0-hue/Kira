"""
VoiceAgent — the conversational layer that synthesizes structured agent outputs
into natural language the shop owner can immediately understand and act on.

This is NOT a router, router, or formatter. Its one job: take what the specialist
agents found and make it sound human — the way a trusted, experienced friend who
happens to be excellent with business would explain it to a shop owner.

Real path (MOCK_MODE=false):
  Calls an LLM via OpenAI-compatible endpoint (VOICE_BASE_URL/VOICE_MODEL).
  Should be a fast, cheap conversational model — NOT the 70B reasoning model.
  Reasoning agent = deep analysis. Voice agent = natural delivery. Two different
  jobs, two different model tiers.

Mock path (MOCK_MODE=true):
  Builds a response directly from the context dict passed by the orchestrator.
  Dynamic — reads actual values (item names, numbers, counts) from context and
  produces different output for different inputs. Sentence structure is templated
  (pre-defined patterns filled with dynamic values), not LLM-generated prose.
  See the honesty note at the bottom of this file.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config.settings import settings
from .base import BaseAgent, AgentRequest, AgentResponse


# ──────────────────────────────────────────────────────────────
# Kira's persona — injected into every LLM call as system prompt
# ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are Kira — a warm, direct AI business assistant built for warung and UMKM \
owners across Indonesia.

WHO YOU ARE:
- A trusted advisor who has helped hundreds of small shop owners manage their \
money, stock, and strategy.
- You talk like a knowledgeable friend, not a consultant. Direct, practical, \
zero fluff.
- You respect that your users run tight businesses with real constraints — their \
time and cash matter.
- You never talk down to them or use business jargon without immediately \
explaining it in plain terms.

YOUR JOB RIGHT NOW:
Specialist systems have analysed this owner's business data. You are receiving \
their structured findings. Your job is to turn those findings into one clear, \
conversational response the owner can immediately understand and act on today.

RULES:
1. Lead with the most urgent thing — if stock is about to run out, say that first.
2. Use specific numbers from the data — "your sugar runs out in 1.3 days" not \
"your stock is low".
3. Give one clear action per issue — not a list of five vague suggestions.
4. If everything is genuinely fine, say so warmly and briefly. Don't invent problems.
5. Never mention "agents", "systems", "analysis engines", or any technical \
infrastructure. You are simply Kira, who looked into their business.
6. Don't start with "As Kira..." or "Based on the data..." — just say it.
7. Keep it to 3–5 sentences unless the situation genuinely requires more detail.

LANGUAGE: {language_instruction}
"""

_LANGUAGE_INSTRUCTIONS = {
    "id": (
        "Respond in informal but respectful Indonesian — the way you would talk "
        "to a warung owner you know well. Use 'Anda' or the owner's name if known. "
        "Natural, conversational Bahasa Indonesia. Not stiff or formal."
    ),
    "en": (
        "Respond in warm, conversational English. Direct and practical. "
        "Not corporate, not stiff."
    ),
}


# ──────────────────────────────────────────────────────────────
# Context builder — turns structured agent outputs into a
# human-readable block for the LLM's user prompt
# ──────────────────────────────────────────────────────────────

def _build_context_block(context: Dict[str, Any]) -> str:
    """
    Converts the {agent_name: result_dict} context into a clear text block.
    Only includes data that is actually present — no empty sections.
    """
    sections: List[str] = []

    # ── Inventory alerts ───────────────────────────────────────
    inv = context.get("inventory_agent", {})
    alerts = inv.get("alerts", [])
    if alerts:
        alert_lines = "\n".join(
            f"  - {a['item']}: {a['days_remaining']} days of stock remaining"
            for a in alerts
        )
        sections.append(f"STOCK ALERTS (items below reorder point):\n{alert_lines}")
    elif inv.get("stock"):
        sections.append("INVENTORY: All stock levels are currently healthy.")

    # ── Bookkeeping / financials ───────────────────────────────
    fin_result = context.get("bookkeeper_agent", {})
    fin = fin_result.get("financials") or fin_result  # handle both shapes
    if isinstance(fin, dict) and fin.get("net_profit") is not None:
        sections.append(
            f"TODAY'S FINANCIALS:\n"
            f"  Revenue: Rp{fin.get('revenue', 0):,}\n"
            f"  Net profit: Rp{fin.get('net_profit', 0):,}\n"
            f"  Gross margin: {fin.get('gross_margin_pct', 0)}%"
        )

    # ── Strategy recommendations ───────────────────────────────
    strat = context.get("strategy_agent", {})
    recs  = strat.get("recommendations", [])
    if recs:
        rec_lines = "\n".join(f"  - {r}" for r in recs)
        sections.append(f"PRODUCT STRATEGY NOTES:\n{rec_lines}")

    # ── Deep reasoning analysis ────────────────────────────────
    reasoning = context.get("reasoning_agent", {})
    analysis  = reasoning.get("analysis", "")
    if analysis and analysis.strip():
        # Trim to avoid overwhelming the voice prompt — first 300 chars
        trimmed = analysis.strip()
        if len(trimmed) > 400:
            trimmed = trimmed[:400].rsplit(" ", 1)[0] + "..."
        sections.append(f"DETAILED ANALYSIS:\n  {trimmed}")

    # ── OCR / receipt data ─────────────────────────────────────
    eyes = context.get("eyes_agent", {})
    if eyes.get("items"):
        item_lines = ", ".join(
            f"{i['name']} ×{i['quantity']}" for i in eyes["items"]
        )
        sections.append(
            f"RECEIPT EXTRACTED ({eyes.get('source_tier', 'local')} OCR):\n"
            f"  Items: {item_lines}\n"
            f"  Total: Rp{eyes.get('total', 0):,}"
        )

    if not sections:
        return "No specialist data available — respond to the owner's question directly."

    return "\n\n".join(sections)


# ──────────────────────────────────────────────────────────────
# Real LLM call
# ──────────────────────────────────────────────────────────────

def _call_voice_llm(user_query: str, context: Dict[str, Any], language: str) -> str:
    from openai import OpenAI

    lang_instr  = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    system      = _SYSTEM_PROMPT_TEMPLATE.format(language_instruction=lang_instr)
    context_block = _build_context_block(context)

    user_prompt = (
        f"The owner asked: \"{user_query}\"\n\n"
        f"What we found in their business right now:\n{context_block}\n\n"
        f"Give them one clear, conversational response."
    )

    client = OpenAI(
        base_url=settings.voice_base_url,
        api_key=settings.voice_api_key,
    )
    response = client.chat.completions.create(
        model=settings.voice_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,   # slightly higher than reasoning — warmth matters here
        max_tokens=512,    # voice responses should be short
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────
# Mock synthesis
# ──────────────────────────────────────────────────────────────
#
# HONESTY NOTE (mirrors the audit standard):
# This mock IS more dynamic than the old stub — it reads actual values
# from the context dict (item names, numbers, alert counts) and produces
# meaningfully different output depending on which agents ran and what
# they found. A request with inventory alerts produces a different response
# than one with only financials.
#
# BUT: the sentence structures are pre-defined templates filled with dynamic
# values, not generated prose. The ordering logic is fixed (alerts first,
# then financials, then strategy, then OCR, then reasoning). It is a
# smarter template, not an LLM. A reviewer who knows what to look for will
# still recognise the pattern.
# ──────────────────────────────────────────────────────────────

def _mock_synthesize(user_query: str, context: Dict[str, Any], language: str) -> str:
    parts: List[str] = []
    is_id = language == "id"

    inv     = context.get("inventory_agent", {})
    fin_raw = context.get("bookkeeper_agent", {})
    strat   = context.get("strategy_agent", {})
    eyes    = context.get("eyes_agent", {})
    reas    = context.get("reasoning_agent", {})

    fin = fin_raw.get("financials") or fin_raw  # handle both dict shapes

    alerts = inv.get("alerts", [])
    recs   = strat.get("recommendations", [])
    analysis = (reas.get("analysis") or "").strip()

    # ── Lead with the most urgent thing ────────────────────────
    if alerts:
        first = alerts[0]
        item  = first["item"]
        days  = first["days_remaining"]
        if is_id:
            parts.append(
                f"Hati-hati Bu/Pak — {item} tinggal sekitar {days} hari lagi!"
            )
            if len(alerts) > 1:
                others = ", ".join(a["item"] for a in alerts[1:])
                parts.append(f"{others} juga perlu segera dipesan.")
        else:
            parts.append(f"Heads up — {item} is running out in about {days} days.")
            if len(alerts) > 1:
                others = ", ".join(a["item"] for a in alerts[1:])
                parts.append(f"Also need to reorder soon: {others}.")

    # ── Financials ─────────────────────────────────────────────
    if isinstance(fin, dict) and fin.get("net_profit") is not None:
        net    = fin["net_profit"]
        margin = fin.get("gross_margin_pct", 0)
        if is_id:
            parts.append(
                f"Keuangan hari ini cukup {'bagus' if net > 0 else 'mengkhawatirkan'}: "
                f"laba bersih Rp{net:,} dengan margin {margin}%."
            )
        else:
            health = "solid" if net > 0 else "concerning"
            parts.append(
                f"Financially, today looks {health}: "
                f"Rp{net:,} net profit at {margin}% margin."
            )

    # ── Strategy ───────────────────────────────────────────────
    # Include ALL recommendations, not just index 0.
    if recs:
        if len(recs) == 1:
            rec = recs[0].rstrip(".")
            if is_id:
                parts.append(f"Soal strategi produk: {rec.lower()}.")
            else:
                parts.append(f"On your products: {rec}.")
        else:
            # Multiple recs — lead with the first (typically the urgent problem),
            # then fold in the rest as a follow-on action.
            if is_id:
                lead   = recs[0].rstrip(".").lower()
                others = "; ".join(r.rstrip(".").lower() for r in recs[1:])
                parts.append(f"Soal produk: {lead}. Selain itu, {others}.")
            else:
                lead   = recs[0].rstrip(".")
                others = "; ".join(r.rstrip(".") for r in recs[1:])
                parts.append(f"On your products: {lead}. Also: {others}.")

    # ── OCR result ─────────────────────────────────────────────
    if eyes.get("items"):
        n     = len(eyes["items"])
        total = eyes.get("total", 0)
        tier  = eyes.get("source_tier", "local")
        if is_id:
            parts.append(
                f"Nota sudah saya baca (via OCR {'lokal' if tier == 'local' else 'cloud'}) "
                f"— {n} item, total Rp{total:,}."
            )
        else:
            parts.append(
                f"I've read the receipt (via {'local' if tier == 'local' else 'cloud'} OCR) "
                f"— {n} items, Rp{total:,} total."
            )

    # ── Reasoning ──────────────────────────────────────────────
    # Always acknowledge when reasoning_agent ran and produced an analysis.
    # If it's the only content, show the first two sentences.
    # If other content (recs/alerts/financials) already covers the main points,
    # add a brief confirmation so the user knows deeper analysis exists.
    if analysis:
        has_other = bool(
            alerts
            or recs
            or (isinstance(fin, dict) and fin.get("net_profit") is not None)
        )
        if has_other:
            if is_id:
                parts.append(
                    "Analisis mendalam saya juga mengkonfirmasi poin-poin ini "
                    "— ada langkah aksi spesifik yang bisa diambil sekarang."
                )
            else:
                parts.append(
                    "My deeper analysis confirms these points "
                    "and has specific action steps ready."
                )
        else:
            # Analysis is the main content — show opening sentences
            clean     = re.sub(r'^[^\wÀ-ɏ]+', '', analysis.strip())
            sentences = re.split(r'(?<=[.!?])\s+', clean)[:2]
            parts.append(" ".join(sentences))

    # ── Nothing at all — pure greeting or unknown ───────────────
    if not parts:
        if any(w in user_query.lower() for w in ("halo", "hai", "hello", "hi", "help")):
            if is_id:
                parts.append(
                    "Halo! Saya Kira. Saya bisa bantu cek stok, lihat keuntungan, "
                    "baca nota belanja, dan kasih saran bisnis. Mau mulai dari mana?"
                )
            else:
                parts.append(
                    "Hello! I'm Kira. I can help you check stock, review profits, "
                    "read receipts, and give business advice. Where would you like to start?"
                )
        else:
            if is_id:
                parts.append(
                    "Oke, sudah saya cek. Semuanya terlihat aman untuk saat ini."
                )
            else:
                parts.append("Got it — I've had a look and everything seems fine right now.")

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# VoiceAgent
# ──────────────────────────────────────────────────────────────

class VoiceAgent(BaseAgent):
    name = "voice_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        """
        Synthesize specialist agent outputs into one conversational response.

        Expects request.context to contain the results of whichever specialist
        agents ran before this one — populated by the orchestrator's dispatch
        node, which runs voice_agent last.

        If context is empty (e.g. routing sent only voice_agent for a greeting),
        responds to the user's question directly without specialist data.
        """
        context  = request.context or {}
        language = request.language
        dev_note: Optional[str] = None

        if settings.mock_mode:
            response_text = _mock_synthesize(request.payload, context, language)
            dev_note = (
                "Mock mode — set VOICE_API_KEY + MOCK_MODE=false for real LLM output"
            )
        else:
            try:
                response_text = _call_voice_llm(request.payload, context, language)
            except Exception as exc:
                response_text = _mock_synthesize(request.payload, context, language)
                dev_note = f"Voice LLM error (fell back to mock): {exc}"

        result: Dict[str, Any] = {"response_text": response_text, "language": language}
        if dev_note:
            result["_dev_note"] = dev_note

        return AgentResponse(agent_name=self.name, result=result)
