"""
StrategyAgent — identifies profitable/loss-making products and recommends
pricing adjustments and promotional strategies.

Data source: data.firestore_client.get_sales_history(user_id)
  Returns per-user product revenue/cost data from Firestore (or LOCAL_FALLBACK).
  Different user_ids return different products and sales figures.

Margin classification thresholds:
  margin < 0%   → losing_money (costs more to make than it earns)
  0% ≤ margin < 10%  → low_margin (profitable but fragile)
  10% ≤ margin < 60% → healthy
  margin ≥ 60%  → star (high-margin, promote more)

These thresholds are intentionally fixed — they represent real business
rules, not agent-specific magic numbers.
"""
from data.firestore_client import get_sales_history
from .base import BaseAgent, AgentRequest, AgentResponse

# Margin classification boundaries
_STAR_THRESHOLD    = 60.0   # margin ≥ this → star product
_HEALTHY_THRESHOLD = 10.0   # margin ≥ this → healthy
_LOSS_THRESHOLD    = 0.0    # margin < this → losing money


def _classify(margin_pct: float) -> str:
    if margin_pct < _LOSS_THRESHOLD:
        return "losing_money"
    if margin_pct < _HEALTHY_THRESHOLD:
        return "low_margin"
    if margin_pct < _STAR_THRESHOLD:
        return "healthy"
    return "star"


class StrategyAgent(BaseAgent):
    name = "strategy_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        sales = get_sales_history(request.user_id)

        # Compute margin for each product from actual revenue/cost data
        product_analysis = []
        for s in sales:
            rev = s["revenue_7d"]
            cost = s["cost_7d"]
            if rev == 0:
                continue
            margin_pct = round((rev - cost) / rev * 100, 1)
            product_analysis.append({
                "product":    s["item"],
                "revenue":    rev,
                "cost":       cost,
                "margin_pct": margin_pct,
                "status":     _classify(margin_pct),
            })

        losers = [p for p in product_analysis if p["status"] == "losing_money"]
        stars  = [p for p in product_analysis if p["status"] == "star"]

        # Recommendations: address losers first (urgent), then highlight stars
        recommendations = []
        for p in losers:
            recommendations.append(
                self._lang(
                    f"Naikkan harga {p['product']} atau kurangi porsi — sekarang rugi {abs(p['margin_pct']):.1f}%.",
                    f"Raise price or reduce portion for {p['product']} — currently losing {abs(p['margin_pct']):.1f}%.",
                    request.language,
                )
            )
        for p in stars:
            recommendations.append(
                self._lang(
                    f"Promosikan {p['product']} lebih banyak — margin tinggi {p['margin_pct']:.1f}%.",
                    f"Promote {p['product']} more — high margin at {p['margin_pct']:.1f}%.",
                    request.language,
                )
            )

        return AgentResponse(
            agent_name=self.name,
            result={
                "product_analysis": product_analysis,
                "recommendations":  recommendations,
                "summary": self._lang(
                    f"{len(losers)} produk merugi, {len(stars)} produk bintang ditemukan.",
                    f"{len(losers)} loss-making product(s), {len(stars)} star product(s) identified.",
                    request.language,
                ),
            },
        )
