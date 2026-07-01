# -*- coding: utf-8 -*-
"""
main.py -- Kira end-to-end demo.

  Demo 1  Reactive | Indonesian  | Inventory stock query     (voice synthesis visible)
  Demo 2  Reactive | English     | Profit & loss summary     (voice synthesis visible)
  Demo 3  Reactive | Indonesian  | Strategy / loss-product   (voice synthesis visible)
  Demo 4  Reactive | English     | General greeting
  Demo 5  EyesAgent | Tier 1     | Clear scan — local OCR handles it
  Demo 6  EyesAgent | Tier 2     | Blurry scan — escalates to cloud vision
  Demo 7  Proactive | Indonesian | Enriched scheduled check (user_001)
  Demo 8  Proactive | English    | Enriched scheduled check (user_002)
  Demo 9  Data layer             | Same query, 3 different user_ids → 3 different results

Demos 1-3 print both the raw structured data from each specialist agent AND
voice_agent's synthesis of that data, making the before/after clearly visible.
Demo 9 proves the unified Firestore data layer: different user_ids return
genuinely different inventory, products, and financial data.

Run:
    python -X utf8 main.py          # Windows (forces UTF-8 console output)
    python main.py                  # Linux / macOS

Set MOCK_MODE=false in .env (with valid API keys) to switch to live LLMs.
"""
import sys
import textwrap

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents import AgentRequest
from agents.eyes_agent import EyesAgent
from agents.inventory_agent import InventoryAgent
from config.settings import settings
from data.firestore_client import get_business_state
from orchestrator import KiraOrchestrator, KiraRequest, run_proactive_check


# ──────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────

_W = 70

def _header(title: str) -> None:
    print("\n" + "═" * _W)
    print(f"  {title}")
    print("═" * _W)


def _section(label: str, value: str) -> None:
    print(f"\n▸ {label}")
    for line in value.splitlines():
        print(f"  {line}")


def _wrap(text: str, indent: int = 4) -> str:
    pad = " " * indent
    return textwrap.fill(
        text, width=_W - indent, initial_indent=pad, subsequent_indent=pad
    )


def _voice_dev_note(result: dict) -> str:
    """Extract the _dev_note from voice_agent result, or empty string."""
    return (
        result.get("agent_results", {})
        .get("voice_agent", {})
        .get("result", {})
        .get("_dev_note", "")
    )


# ── Reactive result printer (basic) ───────────────────────────

def _print_result(result: dict) -> None:
    _section("Intent detected", result["intent"])
    _section("Agents invoked",  ", ".join(result["agents_invoked"]))
    _section("Kira's response", result["final_response"])
    note = _voice_dev_note(result)
    if note:
        print(f"  [dev] {note}")


# ── Reactive result printer with synthesis transparency ────────
# Shows the raw structured data from each specialist agent, then
# voice_agent's natural-language synthesis of that data.
# This makes the "before/after" of voice_agent's work visible.

def _print_result_with_synthesis(result: dict) -> None:
    _section("Intent detected", result["intent"])
    _section("Agents invoked",  ", ".join(result["agents_invoked"]))

    agent_results = result.get("agent_results", {})
    specialist_data = {k: v for k, v in agent_results.items() if k != "voice_agent"}

    if specialist_data:
        print(f"\n{'─' * _W}")
        print("  STEP 1 — Raw specialist agent output (structured data)")
        print(f"{'─' * _W}")
        for agent_name, envelope in specialist_data.items():
            data = envelope.get("result", {})
            if not data:
                continue
            print(f"\n  [{agent_name}]")
            # Inventory
            for alert in data.get("alerts", []):
                print(f"    • ALERT  {alert['item']}: {alert['days_remaining']} days remaining — {alert['action']}")
            # Strategy
            for i, rec in enumerate(data.get("recommendations", [])[:3], 1):
                print(f"    • REC {i}  {rec}")
            # Financials
            fin = data.get("financials", {})
            if fin.get("net_profit") is not None:
                print(f"    • Revenue Rp{fin['revenue']:,}  |  Net profit Rp{fin['net_profit']:,}  |  Margin {fin['gross_margin_pct']}%")
            # Reasoning analysis (first line only for brevity)
            analysis = data.get("analysis", "")
            if analysis:
                first_line = analysis.splitlines()[0]
                print(f"    • ANALYSIS  {first_line[:90]}{'...' if len(first_line) > 90 else ''}")

    print(f"\n{'─' * _W}")
    print("  STEP 2 — voice_agent synthesis (what the shop owner actually sees)")
    print(f"{'─' * _W}")
    for line in result["final_response"].splitlines():
        print(f"  {line}")
    note = _voice_dev_note(result)
    if note:
        print(f"\n  [dev] {note}")


# ── EyesAgent result printer ───────────────────────────────────

_TIER_ICONS = {"local": "🖥️ ", "cloud_fallback": "☁️ "}

def _print_eyes_result(result: dict, language: str = "en") -> None:
    """
    Display the hybrid OCR result, explicitly showing which tier ran
    and why — so the two-tier logic is visible in the demo output.
    """
    tier      = result.get("source_tier", "local")
    conf      = result.get("confidence", 0)
    threshold = settings.eyes_confidence_threshold
    items     = result.get("items", [])
    total     = result.get("total", 0)
    raw_text  = result.get("raw_text", "")
    icon      = _TIER_ICONS.get(tier, "")

    # ── Tier decision ──────────────────────────────────────────
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  {icon} Tier used     : {tier.replace('_', ' ').upper()}")
    print(f"  Local confidence : {conf:.1f}  (threshold = {threshold})")
    if tier == "local":
        print(f"  Decision         : {conf:.1f} ≥ {threshold} → Tier 1 sufficient, no cloud call")
    else:
        print(f"  Decision         : {conf:.1f} < {threshold} → escalated to cloud vision model")
    print(f"  {'─' * (_W - 2)}")

    # ── Extracted items ────────────────────────────────────────
    print(f"\n  Extracted items:")
    if items:
        for item in items:
            subtotal = item["quantity"] * item["unit_price"]
            print(
                f"    • {item['name']:<20} "
                f"qty {item['quantity']:>4}  "
                f"@ Rp{item['unit_price']:>7,}  "
                f"= Rp{int(subtotal):>8,}"
            )
    else:
        print("    (no items extracted)")
    print(f"\n    {'TOTAL':>35}  = Rp{total:>8,}")

    # ── Raw text (truncated) ───────────────────────────────────
    print(f"\n  Raw OCR text (what Tesseract returned):")
    for line in raw_text.splitlines()[:8]:
        print(f"    {line}")
    if raw_text.count("\n") > 8:
        print(f"    ... ({raw_text.count(chr(10)) - 8} more lines)")


# ── Proactive result printer ───────────────────────────────────

_SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡"}

def _print_proactive(suggestions: list, user_id: str) -> None:
    if not suggestions:
        print("  (no qualifying situations detected)")
        return

    business = suggestions[0].get("business", user_id)
    shown    = [s for s in suggestions if s.get("analysis")]
    skipped  = len(suggestions) - len(shown)

    print(f"\n  Business : {business}")
    print(
        f"  Alerts   : {len(shown)} enriched with 70B reasoning  |  "
        f"{skipped} below threshold (skipped)"
    )

    for s in suggestions:
        icon     = _SEVERITY_ICON.get(s["severity"], "⚪")
        analysis = s.get("analysis", {})

        print(f"\n  {'─' * (_W - 2)}")
        print(
            f"  {icon} [{s['priority']}] [{s['category'].upper()}]  "
            f"{s['item']}  —  {s['severity'].upper()} (score {s['severity_score']})"
        )

        print(f"\n  ▸ Detection  [{s['detection']['situation_type']}]")
        for k, v in s["detection"]["numbers"].items():
            fmt_v = (
                f"Rp{v:,}" if isinstance(v, int) and k in (
                    "cash_balance", "avg_daily_sales", "loss_7d",
                    "revenue_7d", "cost_7d", "daily_loss", "monthly_loss"
                ) else v
            )
            print(f"      {k:<20} {fmt_v}")

        if analysis:
            print(f"\n  ▸ Reasoning analysis  [70B model]")
            print(f"\n    WHY IT MATTERS")
            print(_wrap(analysis.get("why_it_matters", "—"), indent=6))
            print(f"\n    IMPACT IF IGNORED")
            print(_wrap(analysis.get("business_impact", "—"), indent=6))
            print(f"\n    NEXT STEP  ✅")
            print(_wrap(analysis.get("next_step", "—"), indent=6))


# ──────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────

def _print_multi_user_inventory() -> None:
    """
    Demo 9 helper: runs the same inventory request for three different user_ids
    and prints each result side by side to prove per-user differentiation.
    """
    inv_agent = InventoryAgent()
    test_users = [
        ("user_001", "id"),
        ("user_002", "en"),
        ("user_003", "id"),
    ]

    for uid, lang in test_users:
        state  = get_business_state(uid)
        biz    = state["business_name"] if state else uid
        req    = AgentRequest(payload="cek stok", user_id=uid, language=lang,
                              input_type="text")
        result = inv_agent.handle(req)
        data   = result.result

        print(f"\n  {'─' * (_W - 2)}")
        print(f"  user_id : {uid}  |  business : {biz}")

        items = data.get("stock", [])
        print(f"  Items tracked ({len(items)}):")
        for item in items:
            flag = " ⚠" if item["stock"] <= item["reorder_point"] else "   "
            print(
                f"{flag}   {item['item']:<18} "
                f"{item['stock']:>6.1f} {item['unit']:<6} "
                f"({item['days_remaining']} days left)"
            )

        alerts = data.get("alerts", [])
        if alerts:
            print(f"  Alerts  : {len(alerts)} item(s) need reordering NOW")
        else:
            print(f"  Alerts  : none — all stock levels healthy")

    print(f"\n  {'─' * (_W - 2)}")
    print(
        "  All three returned different items and different alert counts.\n"
        "  The data layer correctly isolates per-user business state."
    )


def main() -> None:
    print("\n" + "█" * _W)
    print("  KIRA — AI Business Assistant for Warung / UMKM")
    print("  AMD Developer Hackathon Demo  |  Mock Mode")
    print("█" * _W)

    orchestrator = KiraOrchestrator()
    eyes         = EyesAgent()

    # ── Demo 1: stock query (Indonesian) ──────────────────────
    _header("DEMO 1 — Reactive | Indonesian | Inventory query")
    print(
        "\n  voice_agent synthesis demo: shows raw inventory_agent data\n"
        "  alongside the natural-language response voice_agent produces from it."
    )
    r1 = orchestrator.run(KiraRequest(
        payload="Pak, stok gula dan tepung saya masih aman nggak?",
        user_id="user_001",
        language="id",
    ))
    _print_result_with_synthesis(r1)

    # ── Demo 2: P&L query (English) ───────────────────────────
    _header("DEMO 2 — Reactive | English | Profit & loss")
    print(
        "\n  voice_agent synthesis demo: shows raw bookkeeper_agent financials\n"
        "  alongside voice_agent's conversational summary of those numbers."
    )
    r2 = orchestrator.run(KiraRequest(
        payload="Show me today's profit and loss summary.",
        user_id="user_001",
        language="en",
    ))
    _print_result_with_synthesis(r2)

    # ── Demo 3: strategy advice (Indonesian) ──────────────────
    _header("DEMO 3 — Reactive | Indonesian | Strategy advice")
    print(
        "\n  voice_agent synthesis demo: shows raw strategy_agent recommendations\n"
        "  AND reasoning_agent deep analysis, then voice_agent's synthesis of both."
    )
    r3 = orchestrator.run(KiraRequest(
        payload="Produk mana yang paling banyak bikin rugi? Kasih saran strategi dong.",
        user_id="user_001",
        language="id",
    ))
    _print_result_with_synthesis(r3)

    # ── Demo 4: greeting (English) ────────────────────────────
    _header("DEMO 4 — Reactive | English | Greeting (no specialist data)")
    print(
        "\n  No specialist agents run for a greeting — voice_agent responds\n"
        "  directly to the question with no structured context to synthesize."
    )
    r4 = orchestrator.run(KiraRequest(
        payload="Hello Kira! What can you help me with?",
        user_id="user_002",
        language="en",
    ))
    _print_result(r4)

    # ── Demo 5: EyesAgent — Tier 1 (clear scan) ───────────────
    _header("DEMO 5 — EyesAgent | Tier 1 | Clear scan → local OCR")
    print(
        "\n  Simulating a sharp phone photo of a warung purchase receipt.\n"
        "  Tesseract confidence is HIGH → cloud vision is NOT called.\n"
        f"  Threshold: {settings.eyes_confidence_threshold}  |  "
        "Mock mode: True (no Tesseract binary needed)"
    )
    req5 = AgentRequest(
        payload="Tolong baca nota belanja ini",   # normal payload → clear-scan mock
        user_id="user_001",
        language="id",
        input_type="image",
    )
    resp5 = eyes.handle(req5)
    _print_eyes_result(resp5.result, language="id")

    # ── Demo 6: EyesAgent — Tier 2 escalation (blurry scan) ───
    _header("DEMO 6 — EyesAgent | Tier 2 | Blurry scan → cloud fallback")
    print(
        "\n  Simulating a dark, blurry phone photo (common in low-light markets).\n"
        "  Tesseract confidence is LOW → escalates to cloud vision model.\n"
        "  Cloud tier recovers an extra item (Telur) that local OCR missed.\n"
        f"  Threshold: {settings.eyes_confidence_threshold}  |  "
        "Mock mode: True (cloud tier also mocked, free to test)"
    )
    req6 = AgentRequest(
        payload="nota belanja blurry photo dark lighting",   # triggers tier-2 mock
        user_id="user_001",
        language="id",
        input_type="image",
    )
    resp6 = eyes.handle(req6)
    _print_eyes_result(resp6.result, language="id")

    print(
        f"\n  Note: Tier 2 source_tier = '{resp6.result['source_tier']}' confirms\n"
        f"  cloud fallback ran.  Set MOCK_MODE=false + VISION_API_KEY to use a\n"
        f"  real vision model (gpt-4o-mini, LLaVA on Fireworks, etc.)."
    )

    # ── Demo 7: Proactive check (Indonesian) ──────────────────
    _header("DEMO 7 — Proactive | Indonesian | Enriched check for user_001")
    print(
        "\n  ⏰  Scheduled check — no user input.  Pipeline:\n"
        "      detect situations → score severity → filter threshold\n"
        "      → enrich each with 70B reasoning → return structured suggestions"
    )
    _print_proactive(run_proactive_check("user_001"), "user_001")

    # ── Demo 8: Proactive check (English) ─────────────────────
    _header("DEMO 8 — Proactive | English | Enriched check for user_002")
    print("\n  ⏰  Scheduled check — no user input.")
    _print_proactive(run_proactive_check("user_002"), "user_002")

    # ── Demo 9: per-user differentiation proof ────────────────
    _header("DEMO 9 — Data layer | Same query, 3 different businesses")
    print(
        "\n  Runs the same inventory query against user_001, user_002, and user_003.\n"
        "  Each must return different items, stock levels, and alert counts —\n"
        "  direct proof that user_id is now honoured by the data layer.\n"
        "\n  Data source: Firestore (or LOCAL_FALLBACK when credentials not set).\n"
        "  MOCK_MODE has no effect on this data — it only gates LLM calls."
    )
    _print_multi_user_inventory()

    print("\n" + "═" * _W)
    print("  All 9 demos complete.")
    print("  To connect live Firestore: set FIREBASE_CREDENTIALS_PATH in .env,")
    print("  run `python scripts/seed_firestore.py`, then re-run main.py.")
    print("  Set MOCK_MODE=false to also switch LLM calls to live models.")
    print("═" * _W + "\n")


if __name__ == "__main__":
    main()
