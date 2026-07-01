"""
BookkeeperAgent — turns sales data into P&L and cashflow for a specific user.

Data source:
  get_business_state(user_id) → cash_balance, avg_daily_sales, daily_operating_expenses
  get_sales_history(user_id)  → revenue_7d and cost_7d per product

P&L computation:
  All 7-day product totals are averaged to produce an estimated daily P&L.
  This is stated in the output so the owner knows it is an estimate
  (actual daily entries would require a full POS/accounting integration).
"""
from data.firestore_client import get_business_state, get_sales_history
from .base import BaseAgent, AgentRequest, AgentResponse


class BookkeeperAgent(BaseAgent):
    name = "bookkeeper_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        state = get_business_state(request.user_id)
        sales = get_sales_history(request.user_id)

        if not state or not sales:
            # Graceful empty result — data layer returned nothing for this user_id
            empty = {
                "period": "today", "revenue": 0, "cogs": 0,
                "gross_profit": 0, "gross_margin_pct": 0.0,
                "operating_expenses": 0, "net_profit": 0,
                "cashflow": {"opening_balance": 0, "cash_in": 0, "cash_out": 0, "closing_balance": 0},
                "alert": "No data found for this business.",
            }
            return AgentResponse(
                agent_name=self.name,
                result={"financials": empty, "summary": "No data found."},
            )

        # ── Revenue & COGS (7-day totals → daily average) ──────
        revenue_7d = sum(s["revenue_7d"] for s in sales)
        cost_7d    = sum(s["cost_7d"]    for s in sales)
        revenue    = round(revenue_7d / 7)
        cogs       = round(cost_7d / 7)
        gross_profit  = revenue - cogs
        gross_margin  = round(gross_profit / revenue * 100, 1) if revenue else 0.0

        # ── Operating expenses & net profit ────────────────────
        op_expenses = state.get("daily_operating_expenses", 0)
        net_profit  = gross_profit - op_expenses

        # ── Cashflow (today's estimated movement) ──────────────
        cash_balance = state.get("cash_balance", 0)
        cash_in      = revenue
        cash_out     = cogs + op_expenses
        closing      = cash_balance + cash_in - cash_out

        # ── Alert when the shop is running at a daily net loss ─
        alert = None
        if net_profit < 0:
            alert = self._lang(
                f"Peringatan: estimasi rugi bersih hari ini Rp{abs(net_profit):,}",
                f"Warning: estimated net loss today of Rp{abs(net_profit):,}",
                request.language,
            )

        financials = {
            "period":             "today (est. from 7-day average)",
            "revenue":            revenue,
            "cogs":               cogs,
            "gross_profit":       gross_profit,
            "gross_margin_pct":   gross_margin,
            "operating_expenses": op_expenses,
            "net_profit":         net_profit,
            "cashflow": {
                "opening_balance": cash_balance,
                "cash_in":         cash_in,
                "cash_out":        cash_out,
                "closing_balance": closing,
            },
            "alert": alert,
        }

        summary = self._lang(
            f"Laba bersih hari ini: Rp{net_profit:,} (margin {gross_margin}%)",
            f"Net profit today: Rp{net_profit:,} (margin {gross_margin}%)",
            request.language,
        )

        return AgentResponse(
            agent_name=self.name,
            result={"financials": financials, "summary": summary},
        )
