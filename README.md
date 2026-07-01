# Kira — AI Business Assistant for Warung / UMKM

> AMD Developer Hackathon 2024 · AI Agents theme

Kira is a **proactive** multi-agent AI system that helps small business owners
(warung/UMKM) in emerging markets manage stock, understand profit/loss, and
get actionable business advice — in Indonesian or English.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env — set MOCK_MODE=true to run without API keys (default)

# 3. Run the demo
python -X utf8 main.py
```

The `-X utf8` flag is only needed on Windows to enable UTF-8 output.  
On Linux/macOS: `python main.py`

---

## Project Structure

```
kira/
├── agents/
│   ├── base.py              # BaseAgent interface (AgentRequest / AgentResponse)
│   ├── eyes_agent.py        # OCR: receipts, invoices, handwriting  [stub]
│   ├── bookkeeper_agent.py  # P&L, cashflow, financial summaries    [stub]
│   ├── inventory_agent.py   # Stock tracking, stockout prediction    [stub + mock data]
│   ├── strategy_agent.py    # Pricing advice, product analysis       [stub + mock data]
│   ├── voice_agent.py       # Bilingual conversational responses     [stub]
│   └── reasoning_agent.py   # 70B LLM via OpenAI-compatible API     [LIVE in non-mock]
├── config/
│   └── settings.py          # Loads all config from .env
├── orchestrator/
│   ├── orchestrator.py      # KiraOrchestrator — LangGraph StateGraph
│   ├── router.py            # LLM intent classifier (+ mock heuristic)
│   └── proactive.py         # run_proactive_check() — scheduled push suggestions
├── main.py                  # End-to-end demo (7 scenarios)
├── requirements.txt
└── .env.example
```

---

## Architecture

```
User request
     │
     ▼
┌─────────┐   ┌────────┐   ┌──────────────────┐   ┌───────────┐
│  intake │──▶│ route  │──▶│ dispatch_agents   │──▶│ synthesize│──▶ Response
│         │   │ (LLM   │   │ (selected agents  │   │           │
│ validate│   │ class.)│   │  run in sequence) │   │ merge all │
└─────────┘   └────────┘   └──────────────────┘   └───────────┘

Proactive layer (independent of user input):
┌──────────────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ run_proactive_check  │──▶│ reasoning_agent │──▶│ push suggestions │
│ (cron / scheduler)   │   │ (70B model)     │   │ to user          │
└──────────────────────┘   └─────────────────┘   └──────────────────┘
```

The orchestrator is a **LangGraph StateGraph** with four nodes:
`intake → route → dispatch_agents → synthesize`

The proactive check bypasses the graph and calls `ReasoningAgent.call()` directly,
so it can run from a cron job, a background worker, or a push notification service.

---

## Pointing at AMD Developer Cloud

All reasoning calls go through a single `openai.OpenAI` client configured from `.env`.
To switch from mock to live AMD inference, change three lines in `.env` — no code changes:

```env
# AMD Developer Cloud (when available)
REASONING_BASE_URL=https://api.amd.com/v1
REASONING_API_KEY=amd-your-key-here
REASONING_MODEL=meta-llama/Llama-3.1-70B-Instruct

# OR: Fireworks AI (MI300X-backed, available now)
REASONING_BASE_URL=https://api.fireworks.ai/inference/v1
REASONING_API_KEY=fw_your-key-here
REASONING_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct

MOCK_MODE=false
```

The router LLM (for intent classification) uses a separate, cheaper endpoint — keep it
on GPT-4o-mini or a fast small model to minimise latency.

---

## Extending Kira

### Adding a real agent implementation

1. Open the stub in `agents/<name>_agent.py`
2. Replace the `_MOCK_*` block in `handle()` with a real LLM call
3. Call `self._get_client()` from `ReasoningAgent` or inject your own LangChain chain
4. The orchestrator picks up the change automatically — no graph edits needed

### Adding a new agent type

1. Create `agents/new_agent.py` extending `BaseAgent`
2. Register it in `_AGENT_REGISTRY` in `orchestrator/orchestrator.py`
3. Add it to `AGENT_DESCRIPTIONS` in `orchestrator/router.py`
4. The LLM router will include it in its decision-making automatically

### Running proactive checks on a schedule

```python
# In a background worker or cron job:
from orchestrator import run_proactive_check

suggestions = run_proactive_check("user_001")
for s in suggestions:
    push_to_user(s["user_id"], s["message"])   # your push/WhatsApp layer
```

---

## Bilingual Support

Set `language="id"` for Indonesian, `language="en"` for English on any `KiraRequest`.
All agents call `self._lang(id_text, en_text, language)` to pick the right string.
The reasoning agent prepends the language instruction to the system prompt.

---

## Six Agents at a Glance

| Agent | Role | Status |
|-------|------|--------|
| `eyes_agent` | OCR receipts & handwriting → structured data | Stub |
| `bookkeeper_agent` | P&L, cashflow, expense summaries | Stub |
| `inventory_agent` | Stock tracking, stockout alerts, reorder | Stub + mock data |
| `strategy_agent` | Pricing advice, product performance | Stub + mock data |
| `voice_agent` | Bilingual conversational responses | Stub |
| `reasoning_agent` | Heavy 70B reasoning (Llama / Qwen) | **Live** via `.env` |
