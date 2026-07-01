"""
data/forecast.py — internal forward projections derived exclusively from
Firestore data (revenue history, inventory, products).

IMPORTANT: These are NOT external market predictions. Every number here
comes from the business's own historical data: cash balance, observed
daily revenue, stock levels, and product margins. No market data, no
external signals, no ML models.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from data.firestore_client import get_business_state, get_daily_sales_timeseries


# ──────────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────────

def forecast_business(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Compute forward projections for user_id from internal data only.

    Returns None if user_id is not found.

    Returned keys:
        cash_runway_days                        — cash / daily burn
        cash_forecast                           — human-readable cash narrative
        revenue_trend_7d                        — "declining"|"stable"|"growing"
        revenue_trend_pct                       — % change: last 7d vs prior 7d
        avg_daily_revenue_last7                 — Rp, 7-day average
        projected_monthly_loss_from_losing_products  — Rp (0 if none)
        stockout_forecast                       — items running out within 7 days
        summary                                 — one-line summary in business language
    """
    state = get_business_state(user_id)
    if not state:
        return None

    history  = get_daily_sales_timeseries(user_id)   # [{date, revenue, transactions}]
    language = state.get("language", "id")
    is_id    = language == "id"
    products = state.get("_products_raw", [])

    # ── 1. Cash runway ─────────────────────────────────────────
    cash       = state["cash_balance"]
    daily_cogs = sum(p["cost_7d"] for p in products) / 7 if products else 0
    daily_burn = state.get("daily_operating_expenses", 0) + daily_cogs
    cash_runway = round(cash / daily_burn, 1) if daily_burn else 999.0

    # ── 2. Revenue trend from sales_history ────────────────────
    if len(history) >= 14:
        last7 = [h["revenue"] for h in history[-7:]]
        prev7 = [h["revenue"] for h in history[-14:-7]]
        avg_last7 = sum(last7) / 7
        avg_prev7 = sum(prev7) / 7
        trend_pct = round((avg_last7 - avg_prev7) / avg_prev7 * 100, 1) if avg_prev7 else 0.0
    else:
        avg_last7 = float(state.get("avg_daily_sales", 0))
        avg_prev7 = avg_last7
        trend_pct = 0.0

    if trend_pct < -3:
        trend_label = "declining"
    elif trend_pct > 3:
        trend_label = "growing"
    else:
        trend_label = "stable"

    # ── 3. Projected monthly loss from loss-making products ────
    monthly_loss = 0
    for p in products:
        rev7, cost7 = p["revenue_7d"], p["cost_7d"]
        if rev7 > 0 and (rev7 - cost7) / rev7 < 0:
            monthly_loss += int((cost7 - rev7) / 7 * 30)

    # ── 4. Stockout forecast (items running out within 7 days) ─
    today            = date.today()
    stockout_forecast: List[Dict[str, Any]] = []
    for item in state.get("inventory", []):
        usage = item["daily_usage"]
        if usage <= 0:
            continue
        days_left = round(item["stock"] / usage, 1)
        if days_left > 7:
            continue
        if days_left <= 1:
            deadline = "hari ini" if is_id else "today"
        elif days_left <= 2:
            deadline = "besok" if is_id else "tomorrow"
        else:
            n        = int(math.ceil(days_left))
            deadline = f"{n} hari lagi" if is_id else f"in {n} days"
        stockout_forecast.append({
            "item":              item["item"],
            "days_until_stockout": days_left,
            "action_deadline":   deadline,
            "stockout_date":     (today + timedelta(days=int(days_left))).isoformat(),
        })
    stockout_forecast.sort(key=lambda x: x["days_until_stockout"])

    # ── 5. Cash forecast narrative ─────────────────────────────
    avg_rev_rounded = round(avg_last7 / 1000) * 1000
    burn_rounded    = round(daily_burn / 1000) * 1000
    if is_id:
        condition = "kritis — habis" if cash_runway < 3 else "bertahan"
        cash_forecast = (
            f"Dengan saldo Rp{cash:,} dan pengeluaran harian sekitar Rp{int(burn_rounded):,}, "
            f"kas {condition} dalam {cash_runway} hari. "
            f"Rata-rata penjualan 7 hari terakhir: Rp{int(avg_rev_rounded):,}/hari "
            f"({'+' if trend_pct >= 0 else ''}{trend_pct}% vs minggu lalu)."
        )
    else:
        condition = "critically low — runs out" if cash_runway < 3 else "lasts"
        cash_forecast = (
            f"With Rp{cash:,} cash and daily burn around Rp{int(burn_rounded):,}, "
            f"cash {condition} in {cash_runway} days. "
            f"7-day average revenue: Rp{int(avg_rev_rounded):,}/day "
            f"({'+' if trend_pct >= 0 else ''}{trend_pct}% vs prior week)."
        )

    # ── 6. One-line summary ────────────────────────────────────
    parts: List[str] = []
    if is_id:
        if cash_runway < 3:
            parts.append(f"kas kritis ({cash_runway} hari)")
        if trend_pct < -3:
            parts.append(f"penjualan turun {abs(trend_pct)}% minggu ini")
        elif trend_pct > 3:
            parts.append(f"penjualan naik {trend_pct}% minggu ini")
        if stockout_forecast:
            parts.append(f"{len(stockout_forecast)} stok habis dalam 7 hari")
        if monthly_loss > 0:
            parts.append(f"proyeksi rugi produk Rp{monthly_loss:,}/bulan")
        if not parts:
            parts.append("semua indikator dalam kondisi baik")
        summary = "Ringkasan forecast: " + ", ".join(parts) + "."
    else:
        if cash_runway < 3:
            parts.append(f"cash critical ({cash_runway} days)")
        if trend_pct < -3:
            parts.append(f"revenue down {abs(trend_pct)}% this week")
        elif trend_pct > 3:
            parts.append(f"revenue up {trend_pct}% this week")
        if stockout_forecast:
            parts.append(f"{len(stockout_forecast)} items running out within 7 days")
        if monthly_loss > 0:
            parts.append(f"projected product loss Rp{monthly_loss:,}/mo")
        if not parts:
            parts.append("all indicators healthy")
        summary = "Forecast summary: " + ", ".join(parts) + "."

    return {
        "cash_runway_days":                          cash_runway,
        "cash_forecast":                             cash_forecast,
        "revenue_trend_7d":                          trend_label,
        "revenue_trend_pct":                         trend_pct,
        "avg_daily_revenue_last7":                   round(avg_last7),
        "projected_monthly_loss_from_losing_products": monthly_loss,
        "stockout_forecast":                         stockout_forecast,
        "summary":                                   summary,
    }
