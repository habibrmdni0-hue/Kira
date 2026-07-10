"""
DataEntryAgent — lets the shop owner record/update business data through
natural-language chat instead of the manual "Catat Data" form.

Scope: stock updates, sales entry, and ad-hoc expense entry — the three
write actions the "Catat Data" page plus a new expense_history field
support. Restocking inventory FOR RESALE (buying more of something to
sell later) is deliberately NOT one of these — see rule 4 below.

Multi-turn flow:
  Kira uses LLM tool-calling (confirmed working on Fireworks' gpt-oss-120b —
  see scratchpad spike_tool_calling.py). If the owner's message doesn't
  contain enough information to call a tool, the LLM responds with a
  clarifying question in plain text instead — that becomes the reply, and
  the conversation so far is held in memory so the next message continues
  the same slot-filling turn instead of starting over.

Sticky routing: the orchestrator checks has_pending(user_id) BEFORE running
intent classification, so short follow-ups ("iya", "jadi 5kg") stay routed
here instead of being misclassified as small talk on their own. See
orchestrator.py's _node_route.

Session state is in-process memory only (self._sessions / self._pending_confirm),
keyed by user_id. Fine for a single Railway instance / demo; resets on
restart or redeploy.

Confirmation policy (deliberate, not symmetric):
  update_stock              — writes immediately, no confirmation.
                               Reversible/low-risk.
  record_sale/record_expense — money-moving. Requires an explicit yes from
                               the owner before anything is written. The
                               confirmation question text and the yes/no
                               classification are both plain Python (not
                               LLM-generated) so this step can't be talked
                               around by a confused or hallucinating model.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.settings import settings
from data.firestore_client import (
    get_inventory,
    get_sales_history,
    update_inventory_stock,
    record_transaction,
    record_expense,
)
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
    },
    {
        "type": "function",
        "function": {
            "name": "record_sale",
            "description": (
                "Record a sale/transaction for one product — the owner is telling "
                "you they just sold something TO A CUSTOMER. Only call this once "
                "you know the exact product name, the quantity sold, AND the unit "
                "price. Do NOT use this for the owner buying/restocking supplies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product sold, as the owner referred to it.",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "How many units were sold.",
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Price per unit in Rupiah (a plain number, e.g. 15000).",
                    },
                },
                "required": ["product_name", "quantity", "unit_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_expense",
            "description": (
                "Record an ad-hoc OPERATIONAL expense the owner just paid for "
                "(fuel, packaging, utilities, transport, snacks for staff, etc.) "
                "— money spent that is consumed/used, NOT inventory bought to "
                "resell later. Only call this once you know a description of "
                "what it was for AND the amount in Rupiah. No fixed list to "
                "match against — any description is fine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of what the expense was for.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount spent, in Rupiah (a plain number, e.g. 20000).",
                    },
                },
                "required": ["description", "amount"],
            },
        },
    },
]

_SYSTEM_TEMPLATE = """\
You are Kira, helping a warung (small shop) owner record business data \
through chat instead of filling out a form.

You have three tools:
- update_stock: the owner is telling you a new stock quantity for an item.
- record_sale: the owner is telling you they sold something to a customer.
- record_expense: the owner is telling you they spent money on an \
operational cost (fuel, packaging, utilities, etc.) — something consumed, \
not inventory bought to resell.

RULES:
1. If a message is missing information needed for a tool, do NOT call it — \
ask a short, natural follow-up question for exactly the missing piece(s). \
Do not ask about anything else.
2. Never guess a name, quantity, price, or amount that wasn't stated or \
clearly implied.
3. The owner's actual inventory items are: {item_list}.
   The owner's actual products (for sales) are: {product_list}.
   For update_stock/record_sale, if they mention an item/product that \
doesn't clearly match one of these lists, do NOT call any function — tell \
them it isn't in their records and they should add it via the "Catat Data" \
page first. record_expense has no such list — any description is valid.
4. CRITICAL — buying supplies vs buying inventory to resell: if the owner \
says they BOUGHT/PURCHASED something and it's clearly for resale (mentions \
"buat dijual", "buat jualan", "untuk stok", "restock", or the item matches \
one of the inventory items above), that is NOT record_expense — it changes \
what they have to sell, not an operating cost. Do NOT call record_expense \
for this. Instead, explain that restocking isn't recordable via chat yet — \
tell them to use "update stok [item] jadi [total setelah barang diterima]" \
once they know the new total, or ask them to clarify if they actually mean \
an operational expense (not for resale).
5. Indonesian shorthand numbers: "rb"/"ribu" = x1,000, "jt"/"juta" = x1,000,000 \
(e.g. "30rb" = 30000, "1,5jt" = 1500000).
6. Once you have everything a tool needs (matching the lists above where \
applicable), call it immediately — don't ask for confirmation yourself, \
just call it.
7. Keep any text reply short — one or two sentences.

LANGUAGE: {language_instruction}
"""

_LANGUAGE_INSTRUCTIONS = {
    "id": "Respond in informal but respectful Indonesian.",
    "en": "Respond in warm, conversational English.",
}

_YES_WORDS = {
    "ya", "iya", "yaa", "iyaa", "yoi", "yap", "yup", "yep", "ok", "oke", "okay",
    "benar", "betul", "bener", "sip", "siap", "lanjut", "gas", "cocok",
    "yes", "yeah", "yea", "correct", "right", "confirm", "confirmed",
}
_NO_WORDS = {
    "tidak", "gak", "ga", "nggak", "enggak", "bukan", "salah", "batal",
    "no", "nope", "wrong", "cancel", "not",
}


def _classify_yes_no(text: str) -> Optional[bool]:
    """Deterministic yes/no classification for a confirmation reply.

    Plain keyword match on purpose — this gate protects a money-moving
    write, so it should not depend on another LLM call succeeding.
    Returns True/False, or None if the reply is genuinely ambiguous.
    """
    normalized = text.strip().lower().strip("!.,")
    words = set(normalized.split())
    if normalized in _YES_WORDS or words & _YES_WORDS:
        return True
    if normalized in _NO_WORDS or words & _NO_WORDS:
        return False
    return None


class DataEntryAgent(BaseAgent):
    name = "data_entry_agent"

    def __init__(self):
        self._client = None
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._pending_confirm: Dict[str, Dict[str, Any]] = {}

    def has_pending(self, user_id: str) -> bool:
        """True if user_id is mid slot-filling OR awaiting yes/no confirmation."""
        return bool(self._sessions.get(user_id)) or user_id in self._pending_confirm

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

        if user_id in self._pending_confirm:
            return self._handle_confirmation(request)

        items = get_inventory(user_id)
        item_names = [i["item"] for i in items]
        products = get_sales_history(user_id)
        product_names = [p["item"] for p in products]

        if not item_names and not product_names:
            text = self._lang(
                "Saya tidak menemukan data bisnis untuk akun ini.",
                "I couldn't find any business data for this account.",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        system = _SYSTEM_TEMPLATE.format(
            item_list=", ".join(item_names) or "(none on record)",
            product_list=", ".join(product_names) or "(none on record)",
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
                "Bisa tolong dijelaskan lebih detail?",
                "Could you give me a bit more detail?",
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

        if call.function.name == "record_sale":
            return self._handle_record_sale(user_id, language, args, product_names)
        if call.function.name == "record_expense":
            return self._handle_record_expense(user_id, language, args)

        return self._handle_update_stock(user_id, language, args, item_names)

    # ── update_stock: validate, write immediately, no confirmation ─────

    def _handle_update_stock(
        self, user_id: str, language: str, args: Dict[str, Any], item_names: List[str],
    ) -> AgentResponse:
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

        self._sessions.pop(user_id, None)
        text = self._lang(
            f"Oke, stok {result['updated_item']} sudah saya update jadi {result['new_stock']:g}.",
            f"Done — {result['updated_item']} stock updated to {result['new_stock']:g}.",
            language,
        )
        return AgentResponse(agent_name=self.name, result={"response_text": text})

    # ── record_sale / record_expense: validate, then require confirmation ─

    def _handle_record_sale(
        self, user_id: str, language: str, args: Dict[str, Any], product_names: List[str],
    ) -> AgentResponse:
        product_name = str(args.get("product_name", "")).strip()
        matched = next((n for n in product_names if n.lower() == product_name.lower()), None)
        try:
            quantity = float(args.get("quantity"))
        except (TypeError, ValueError):
            quantity = None
        try:
            unit_price = float(args.get("unit_price"))
        except (TypeError, ValueError):
            unit_price = None

        if matched is None or not quantity or quantity <= 0 or not unit_price or unit_price <= 0:
            self._sessions.pop(user_id, None)
            text = self._lang(
                f"Maaf, saya tidak yakin soal produk '{product_name}' atau angkanya. "
                f"Produk yang tercatat: {', '.join(product_names)}. Coba lagi?",
                f"Sorry, I'm not confident about product '{product_name}' or the numbers. "
                f"Recorded products: {', '.join(product_names)}. Could you try again?",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        revenue = quantity * unit_price
        self._sessions.pop(user_id, None)
        self._pending_confirm[user_id] = {
            "action":       "record_sale",
            "product_name": matched,
            "quantity":     quantity,
            "unit_price":   unit_price,
            "revenue":      revenue,
            "language":     language,
        }
        text = self._lang(
            f"Oke, saya catat penjualan {matched} {quantity:g} pcs @ Rp{unit_price:,.0f} "
            f"= Rp{revenue:,.0f}. Benar?",
            f"Got it — recording a sale of {matched} x{quantity:g} @ Rp{unit_price:,.0f} "
            f"= Rp{revenue:,.0f}. Is that right?",
            language,
        )
        return AgentResponse(agent_name=self.name, result={"response_text": text})

    def _handle_record_expense(
        self, user_id: str, language: str, args: Dict[str, Any],
    ) -> AgentResponse:
        description = str(args.get("description", "")).strip()
        try:
            amount = float(args.get("amount"))
        except (TypeError, ValueError):
            amount = None

        if not description or not amount or amount <= 0:
            self._sessions.pop(user_id, None)
            text = self._lang(
                "Maaf, saya tidak yakin soal deskripsi atau jumlahnya. Coba lagi?",
                "Sorry, I'm not confident about the description or the amount. Could you try again?",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        self._sessions.pop(user_id, None)
        self._pending_confirm[user_id] = {
            "action":      "record_expense",
            "description": description,
            "amount":      amount,
            "language":    language,
        }
        text = self._lang(
            f"Oke, saya catat pengeluaran '{description}' sebesar Rp{amount:,.0f}. Benar?",
            f"Got it — recording an expense of '{description}' for Rp{amount:,.0f}. Is that right?",
            language,
        )
        return AgentResponse(agent_name=self.name, result={"response_text": text})

    def _handle_confirmation(self, request: AgentRequest) -> AgentResponse:
        user_id = request.user_id
        pending = self._pending_confirm[user_id]
        language = pending["language"]
        verdict = _classify_yes_no(request.payload)

        if verdict is None:
            text = self._lang(
                "Maaf, saya kurang paham — ya atau tidak untuk catat ini?",
                "Sorry, I didn't quite catch that — yes or no to record this?",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        del self._pending_confirm[user_id]

        if not verdict:
            text = self._lang(
                "Oke, tidak jadi saya catat. Kalau mau coba lagi, sebutkan detailnya.",
                "Okay, I won't record that. Let me know the details if you'd like to try again.",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        if pending["action"] == "record_expense":
            try:
                result = record_expense(user_id, pending["description"], pending["amount"])
            except ValueError as exc:
                return AgentResponse(
                    agent_name=self.name,
                    result={"response_text": str(exc)},
                    success=False, error=str(exc),
                )
            text = self._lang(
                f"Sip, pengeluaran '{pending['description']}' sudah saya catat "
                f"(Rp{pending['amount']:,.0f}). Saldo kas sekarang Rp{result['new_cash_balance']:,.0f}.",
                f"Done — expense '{pending['description']}' recorded "
                f"(Rp{pending['amount']:,.0f}). Cash balance is now Rp{result['new_cash_balance']:,.0f}.",
                language,
            )
            return AgentResponse(agent_name=self.name, result={"response_text": text})

        try:
            result = record_transaction(
                user_id, pending["product_name"], pending["quantity"], pending["unit_price"],
            )
        except ValueError as exc:
            return AgentResponse(
                agent_name=self.name,
                result={"response_text": str(exc)},
                success=False, error=str(exc),
            )

        text = self._lang(
            f"Sip, penjualan {pending['product_name']} sudah saya catat "
            f"(Rp{pending['revenue']:,.0f}). Saldo kas sekarang Rp{result['new_cash_balance']:,.0f}.",
            f"Done — sale of {pending['product_name']} recorded "
            f"(Rp{pending['revenue']:,.0f}). Cash balance is now Rp{result['new_cash_balance']:,.0f}.",
            language,
        )
        return AgentResponse(agent_name=self.name, result={"response_text": text})
