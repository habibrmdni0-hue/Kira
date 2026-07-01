"""
InventoryAgent — tracks stock levels, predicts stockouts, drafts reorder messages.

Data source: data.firestore_client.get_inventory(user_id)
  Returns per-user inventory from Firestore (or LOCAL_FALLBACK when
  credentials are not configured). Different user_ids return different stock.

The days_remaining calculation and alert logic are unchanged from the original.
"""
from data.firestore_client import get_inventory
from .base import BaseAgent, AgentRequest, AgentResponse


class InventoryAgent(BaseAgent):
    name = "inventory_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        stock_items = get_inventory(request.user_id)

        alerts = []
        for item in stock_items:
            days_left = (
                round(item["stock"] / item["daily_usage"], 1)
                if item["daily_usage"] > 0
                else 999
            )
            item["days_remaining"] = days_left
            if item["stock"] <= item["reorder_point"]:
                alerts.append({
                    "item":           item["item"],
                    "days_remaining": days_left,
                    "action": self._lang(
                        f"Segera pesan {item['item']} — stok habis dalam {days_left} hari",
                        f"Reorder {item['item']} now — runs out in {days_left} days",
                        request.language,
                    ),
                })

        summary = (
            self._lang(
                f"{len(alerts)} item perlu dipesan segera.",
                f"{len(alerts)} items need reordering.",
                request.language,
            )
            if alerts
            else self._lang(
                "Stok semua item aman.",
                "All stock levels are healthy.",
                request.language,
            )
        )

        return AgentResponse(
            agent_name=self.name,
            result={"stock": stock_items, "alerts": alerts, "summary": summary},
        )
