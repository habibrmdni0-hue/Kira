"""
DataEntryAgent — lets the shop owner record/update business data through
natural-language chat instead of the manual "Catat Data" form.

Current scope (slice 1): stock updates only. Sales and expense recording
are separate future slices — deliberately not built yet.

Multi-turn flow:
  Kira uses LLM tool-calling (confirmed working on Fireworks' gpt-oss-120b —
  see scratchpad spike_tool_calling.py). If the owner's message doesn't
  contain enough information to call update_stock, the LLM responds with a
  clarifying question in plain text instead — that becomes the reply, and
  the conversation so far is held in memory so the next message continues
  the same slot-filling turn instead of starting over.

Session state is in-process memory only (self._sessions), keyed by user_id.
Fine for a single Railway instance / demo; resets on restart or redeploy.

No confirmation step before writing — stock updates are reversible/low-risk.
Money-moving writes (sales/expenses), when built, should confirm first.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from config.settings import settings
from data.firestore_client import get_inventory, update_inventory_stock
from .base import BaseAgent, AgentRequest, AgentResponse


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": (
                "Update a business's stock quantity for one inventory item to a "
                "new absolute value (not a delta — the full new total, not an "
                "amount to add or subtract). Only call this once you know BOTH "
                "the exact item name and the new stock quantity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the inventory item, as the owner referred to it.",
                    },
                    "new_stock_quantity": {
                        "type": "number",
                        "description": "The new total stock quantity for this item.",
                    },
                },
                "required": ["item_name", "new_stock_quantity"],
            },
        },
    }
]

_SYSTEM_TEMPLATE = """\
You are Kira, helping a warung (small shop) owner update their stock records \
through chat instead of filling out a form.

Your ONLY job right now: figure out which inventory item the owner wants to \
update and the new stock quantity, then call update_stock.

RULES:
1. If the owner's message is missing the item name OR the new quantity, do \
NOT call the function — ask a short, natural follow-up question for exactly \
the missing piece(s). Do not ask about anything else.
2. Never guess a quantity or item name that wasn't stated or clearly implied.
3. The owner's actual inventory items are: {item_list}. If they mention an \
item that doesn't clearly match one of these, do NOT call the function — \
tell them that item isn't in their records and they should add it via the \
"Catat Data" page first.
4. Once you have both a valid item name (matching the list above) and a \
clear numeric quantity, call update_stock immediately — don't ask for \
confirmation, just call it.
5. Keep any text reply short — one or two sentences.

LANGUAGE: {language_instruction}
"""

_LANGUAGE_INSTRUCTIONS = {
    "id": "Respond in informal but respectful Indonesian.",
    "en": "Respond in warm, conversational English.",
}


class DataEntryAgent(BaseAgent):
    name = "data_entry_agent"

    def __init__(self):
        self._client = None
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=settings.reasoning_base_url,
                api_key=settings.reasoning_api_key,
            )
        return self._client

    def handle(self, request: AgentRequest) -> AgentResponse:
        user_id  = request.user_id
        language = request.language

        items = get_inventory(user_id)
        item_names = [i["item"] for i in items]
        if not item_names:
            text = self._lang(
                "Saya tidak menemukan data stok untuk bisnis ini.",
                "I couldn't find any stock data for this business.",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        system = _SYSTEM_TEMPLATE.format(
            item_list=", ".join(item_names),
            language_instruction=_LANGUAGE_INSTRUCTIONS.get(
                language, _LANGUAGE_INSTRUCTIONS["en"]
            ),
        )

        history = self._sessions.get(user_id, [])
        messages = (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": request.payload}]
        )

        try:
            resp = self._get_client().chat.completions.create(
                model=settings.reasoning_model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            self._sessions.pop(user_id, None)
            text = self._lang(
                "Maaf, ada gangguan teknis saat mencoba mencatat data. Coba lagi sebentar lagi.",
                "Sorry, a technical error occurred while trying to record this. Please try again shortly.",
                language,
            )
            return AgentResponse(
                agent_name=self.name, result={"response_text": text},
                success=False, error=str(exc),
            )

        choice = resp.choices[0].message

        if not choice.tool_calls:
            reply = (choice.content or "").strip() or self._lang(
                "Bisa tolong sebutkan item dan jumlah stok barunya?",
                "Could you tell me the item and the new stock quantity?",
                language,
            )
            history.append({"role": "user", "content": request.payload})
            history.append({"role": "assistant", "content": reply})
            self._sessions[user_id] = history
            return AgentResponse(agent_name=self.name, result={"response_text": reply})

        call = choice.tool_calls[0]
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            self._sessions.pop(user_id, None)
            text = self._lang(
                "Maaf, saya kesulitan memahami itu. Bisa diulang dengan lebih jelas?",
                "Sorry, I had trouble understanding that. Could you rephrase?",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        item_name = str(args.get("item_name", "")).strip()
        matched = next((n for n in item_names if n.lower() == item_name.lower()), None)
        try:
            quantity = float(args.get("new_stock_quantity"))
        except (TypeError, ValueError):
            quantity = None

        if matched is None or quantity is None or quantity < 0:
            self._sessions.pop(user_id, None)
            text = self._lang(
                f"Maaf, saya tidak yakin soal item '{item_name}' atau jumlahnya. "
                f"Item yang tercatat: {', '.join(item_names)}. Coba lagi?",
                f"Sorry, I'm not confident about item '{item_name}' or the quantity. "
                f"Recorded items: {', '.join(item_names)}. Could you try again?",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        try:
            result = update_inventory_stock(user_id, matched, quantity)
        except ValueError as exc:
            self._sessions.pop(user_id, None)
            return AgentResponse(
                agent_name=self.name,
                result={"response_text": str(exc)},
                success=False, error=str(exc),
            )

        self._sessions.pop(user_id, None)  # done — clear slot state
        text = self._lang(
            f"Oke, stok {result['updated_item']} sudah saya update jadi {result['new_stock']:g}.",
            f"Done — {result['updated_item']} stock updated to {result['new_stock']:g}.",
            language,
        )
        return AgentResponse(agent_name=self.name, result={"response_text": text})
