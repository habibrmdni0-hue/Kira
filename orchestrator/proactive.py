"""
Proactive layer — what makes Kira AGENTIC, not just a chatbot.

Pipeline:
  1. DETECT   — rule-based scan of business state → List[Detection]
  2. FILTER   — drop anything below REASONING_THRESHOLD (severity_score < 50)
                so we don't waste GPU time on non-issues
  3. ENRICH   — for each qualifying detection, call the 70B reasoning model
                to produce: why_it_matters / business_impact / next_step
  4. RETURN   — list of enriched suggestions, sorted by severity descending

run_proactive_check(user_id) is fully decoupled from the request/response
flow and can be called from a cron job, a push scheduler, or a background
worker without any user input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from agents.reasoning_agent import ReasoningAgent
from data.firestore_client import get_business_state


# ──────────────────────────────────────────────────────────────
# Severity threshold — detections below this score are too minor
# to justify a 70B reasoning call. Tune upward to be more
# selective; lower to catch more edge cases.
# ──────────────────────────────────────────────────────────────
REASONING_THRESHOLD = 50


# ──────────────────────────────────────────────────────────────
# Internal detection object (not exposed publicly)
# ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    category: str              # "inventory" | "finance" | "strategy"
    situation_type: str        # "stockout_critical" | "stockout_high" | "stockout_medium"
                               # "losing_product" | "low_margin" | "low_cash"
    severity: str              # "critical" | "high" | "medium"
    severity_score: int        # 0–100; used for threshold filtering and sort order
    item_name: str             # the product / balance item this detection is about
    numbers: Dict[str, Any]    # the actual figures that triggered this detection
    language: str
    business_name: str
    owner: str
    extra_context: Dict[str, Any] = field(default_factory=dict)


# _MOCK_BUSINESS_STATE removed — replaced by data.firestore_client.get_business_state()
# All business data now comes from the unified Firestore data layer.


# ──────────────────────────────────────────────────────────────
# Detection logic
# ──────────────────────────────────────────────────────────────

def _inventory_severity(days_remaining: float) -> Tuple[int, str]:
    if days_remaining <= 1:  return 100, "critical"
    if days_remaining <= 2:  return 90,  "critical"
    if days_remaining <= 3:  return 75,  "high"
    if days_remaining <= 7:  return 50,  "medium"
    return 0, "low"


def _cash_severity(days_covered: float) -> Tuple[int, str]:
    if days_covered < 1:  return 95, "critical"
    if days_covered < 2:  return 75, "high"
    if days_covered < 5:  return 50, "medium"
    return 0, "low"


def _margin_severity(margin_pct: float) -> Tuple[int, str]:
    if margin_pct < 0:   return 70, "high"
    if margin_pct < 5:   return 50, "medium"
    return 0, "low"


def _detect_situations(state: Dict[str, Any], language: str) -> List[Detection]:
    """
    Scans a business state snapshot and returns all detected situations,
    including low-severity ones. Threshold filtering happens in the caller.
    """
    detections: List[Detection] = []
    business     = state["business_name"]
    owner        = state["owner"]
    avg_daily    = state.get("avg_daily_sales", 1)

    # ── Inventory: stockout proximity ──────────────────────────
    for item in state.get("inventory", []):
        if item["daily_usage"] <= 0:
            continue
        days = round(item["stock"] / item["daily_usage"], 1)
        score, severity = _inventory_severity(days)
        if score == 0:
            continue  # healthy — no detection needed

        sit_type = (
            "stockout_critical" if severity == "critical" else
            "stockout_high"     if severity == "high"     else
            "stockout_medium"
        )
        detections.append(Detection(
            category="inventory",
            situation_type=sit_type,
            severity=severity,
            severity_score=score,
            item_name=item["item"],
            numbers={
                "stock":          item["stock"],
                "unit":           item["unit"],
                "daily_usage":    item["daily_usage"],
                "days_remaining": days,
                "min_reorder_qty": round(item["daily_usage"] * 7, 1),  # 1-week buffer
            },
            language=language,
            business_name=business,
            owner=owner,
            extra_context={"avg_daily_sales": avg_daily},
        ))

    # ── Finance: cash coverage ─────────────────────────────────
    cash   = state.get("cash_balance", 0)
    days_c = round(cash / avg_daily, 1) if avg_daily > 0 else 999
    score_c, sev_c = _cash_severity(days_c)
    if score_c > 0:
        detections.append(Detection(
            category="finance",
            situation_type="low_cash",
            severity=sev_c,
            severity_score=score_c,
            item_name="Cash Balance",
            numbers={
                "cash_balance":    cash,
                "avg_daily_sales": avg_daily,
                "days_covered":    days_c,
            },
            language=language,
            business_name=business,
            owner=owner,
        ))

    # ── Strategy: loss-making or very-low-margin products ──────
    for sale in state.get("recent_sales_7d", []):
        if sale["revenue"] == 0:
            continue
        margin = (sale["revenue"] - sale["cost"]) / sale["revenue"]
        score_m, sev_m = _margin_severity(margin * 100)
        if score_m == 0:
            continue

        loss_7d       = sale["cost"] - sale["revenue"]  # positive = losing money
        daily_loss    = round(loss_7d / 7, 0)
        monthly_loss  = round(daily_loss * 30, 0)

        detections.append(Detection(
            category="strategy",
            situation_type="losing_product" if margin < 0 else "low_margin",
            severity=sev_m,
            severity_score=score_m,
            item_name=sale["item"],
            numbers={
                "revenue_7d":    sale["revenue"],
                "cost_7d":       sale["cost"],
                "loss_7d":       loss_7d,
                "margin_pct":    round(margin * 100, 1),
                "daily_loss":    int(daily_loss),
                "monthly_loss":  int(monthly_loss),
            },
            language=language,
            business_name=business,
            owner=owner,
            extra_context={"avg_daily_sales": avg_daily},
        ))

    return detections


# ──────────────────────────────────────────────────────────────
# Enrichment — 70B reasoning call per qualifying detection
# ──────────────────────────────────────────────────────────────

_ENRICH_SYSTEM_ID = """\
Kamu adalah Kira, asisten bisnis AI yang sangat berpengalaman untuk warung kecil.

Diberikan satu situasi bisnis spesifik dengan data nyata, tugasmu adalah menganalisis
dan memberikan penjelasan TERSTRUKTUR dalam tiga bagian persis:

KENAPA INI PENTING:
[2-3 kalimat yang mengacu pada angka nyata dari data. Jelaskan akar masalahnya.]

DAMPAK JIKA DIABAIKAN:
[Dampak konkret dengan estimasi Rupiah jika memungkinkan. Spesifik, bukan generik.]

LANGKAH SELANJUTNYA:
[SATU tindakan konkret. Bahasa informal seperti teman yang sudah bertahun-tahun \
bantu warung. Tidak ada jargon bisnis.]

Penting: gunakan angka dari data yang diberikan. Jangan mengada-ada angka baru.
"""

_ENRICH_SYSTEM_EN = """\
You are Kira, a highly experienced AI business advisor for small warung shops.

Given one specific business situation with real data, produce a STRUCTURED analysis
in exactly three sections:

WHY THIS MATTERS:
[2-3 sentences referencing the actual numbers in the data. Explain the root cause.]

IMPACT IF IGNORED:
[Concrete business impact with Rp estimates where possible. Specific, not generic.]

NEXT STEP:
[ONE concrete action. Tone: a trusted friend who has helped small shops for years.
No business jargon — plain language only.]

Important: use only the numbers provided in the data. Do not fabricate new figures.
"""


def _build_enrichment_user_prompt(d: Detection, state: Dict[str, Any], lang: str) -> str:
    n = d.numbers
    ctx = d.extra_context

    if d.situation_type in ("stockout_critical", "stockout_high", "stockout_medium"):
        if lang == "id":
            return (
                f"Warung: {d.business_name}\n"
                f"Situasi: Stok {d.item_name} hampir habis\n\n"
                f"DATA:\n"
                f"  Stok saat ini : {n['stock']} {n['unit']}\n"
                f"  Pemakaian/hari: {n['daily_usage']} {n['unit']}\n"
                f"  Sisa waktu    : {n['days_remaining']} hari\n"
                f"  Qty reorder   : minimal {n['min_reorder_qty']} {n['unit']} (stok 1 minggu)\n"
                f"  Avg omzet/hari: Rp{ctx.get('avg_daily_sales', 0):,}"
            )
        return (
            f"Business: {d.business_name}\n"
            f"Situation: {d.item_name} stock running critically low\n\n"
            f"DATA:\n"
            f"  Current stock : {n['stock']} {n['unit']}\n"
            f"  Daily usage   : {n['daily_usage']} {n['unit']}\n"
            f"  Days remaining: {n['days_remaining']} days\n"
            f"  Reorder qty   : at least {n['min_reorder_qty']} {n['unit']} (1-week buffer)\n"
            f"  Avg daily rev : Rp{ctx.get('avg_daily_sales', 0):,}"
        )

    if d.situation_type in ("losing_product", "low_margin"):
        if lang == "id":
            return (
                f"Warung: {d.business_name}\n"
                f"Situasi: Produk {'merugi' if d.situation_type == 'losing_product' else 'margin rendah'} — {d.item_name}\n\n"
                f"DATA (7 hari terakhir):\n"
                f"  Pendapatan: Rp{n['revenue_7d']:,}\n"
                f"  Biaya HPP : Rp{n['cost_7d']:,}\n"
                f"  Kerugian  : Rp{n['loss_7d']:,} total / Rp{n['daily_loss']:,} per hari\n"
                f"  Margin    : {n['margin_pct']}%\n"
                f"  Est. rugi bulan ini: Rp{n['monthly_loss']:,}"
            )
        return (
            f"Business: {d.business_name}\n"
            f"Situation: {'Loss-making' if d.situation_type == 'losing_product' else 'Low-margin'} product — {d.item_name}\n\n"
            f"DATA (last 7 days):\n"
            f"  Revenue   : Rp{n['revenue_7d']:,}\n"
            f"  COGS      : Rp{n['cost_7d']:,}\n"
            f"  Loss      : Rp{n['loss_7d']:,} total / Rp{n['daily_loss']:,} per day\n"
            f"  Margin    : {n['margin_pct']}%\n"
            f"  Est. monthly loss: Rp{n['monthly_loss']:,}"
        )

    if d.situation_type == "low_cash":
        if lang == "id":
            return (
                f"Warung: {d.business_name}\n"
                f"Situasi: Saldo kas rendah\n\n"
                f"DATA:\n"
                f"  Saldo kas saat ini : Rp{n['cash_balance']:,}\n"
                f"  Avg omzet/hari     : Rp{n['avg_daily_sales']:,}\n"
                f"  Kas cukup untuk    : {n['days_covered']} hari operasional"
            )
        return (
            f"Business: {d.business_name}\n"
            f"Situation: Low cash balance\n\n"
            f"DATA:\n"
            f"  Current cash balance: Rp{n['cash_balance']:,}\n"
            f"  Avg daily sales     : Rp{n['avg_daily_sales']:,}\n"
            f"  Cash covers         : {n['days_covered']} days of operations"
        )

    # Fallback for any future situation types
    return f"Business: {d.business_name}\nSituation: {d.situation_type}\nData: {d.numbers}"


def _parse_enrichment(raw: str, lang: str) -> Dict[str, str]:
    """Extract the three structured sections from the 70B model's response."""
    if lang == "id":
        headers = ("KENAPA INI PENTING:", "DAMPAK JIKA DIABAIKAN:", "LANGKAH SELANJUTNYA:")
        keys    = ("why_it_matters", "business_impact", "next_step")
    else:
        headers = ("WHY THIS MATTERS:", "IMPACT IF IGNORED:", "NEXT STEP:")
        keys    = ("why_it_matters", "business_impact", "next_step")

    result: Dict[str, str] = {k: "" for k in keys}
    current_key: Optional[str] = None

    for line in raw.splitlines():
        stripped = line.strip()
        matched = False
        for header, key in zip(headers, keys):
            if stripped.upper().startswith(header.upper()):
                current_key = key
                remainder = stripped[len(header):].strip()
                if remainder:
                    result[current_key] = remainder
                matched = True
                break
        if not matched and current_key and stripped:
            sep = " " if result[current_key] else ""
            result[current_key] += sep + stripped

    # Fallback: if parsing failed, put everything in why_it_matters
    if not any(result.values()):
        result["why_it_matters"] = raw.strip()

    return result


# ──────────────────────────────────────────────────────────────
# Mock enrichment — realistic deep-analysis responses per
# detection type, grounded in the detection's actual numbers.
# Only used when MOCK_MODE=true. Kept in this file so it stays
# next to the data structures that generate it.
# ──────────────────────────────────────────────────────────────

def _mock_enrich(d: Detection, lang: str) -> Dict[str, str]:
    n = d.numbers

    if d.situation_type in ("stockout_critical", "stockout_high"):
        if lang == "id":
            return {
                "why_it_matters": (
                    f"{d.item_name} hanya tersisa {n['stock']} {n['unit']} dengan pemakaian "
                    f"{n['daily_usage']} {n['unit']}/hari — stok habis dalam {n['days_remaining']} hari. "
                    f"Bahan ini kemungkinan besar adalah komponen utama produk terlaris warung Anda, "
                    f"sehingga kehabisan stok langsung berdampak pada kemampuan produksi."
                ),
                "business_impact": (
                    f"Jika tidak dipesan hari ini, warung berpotensi tidak bisa memproduksi menu "
                    f"berbahan {d.item_name} mulai besok. Dengan rata-rata omzet "
                    f"Rp{d.extra_context.get('avg_daily_sales', 0):,}/hari, satu hari tutup atau "
                    f"menu kosong bisa berarti kehilangan Rp{int(d.extra_context.get('avg_daily_sales', 0) * 0.3):,}–"
                    f"Rp{int(d.extra_context.get('avg_daily_sales', 0) * 0.5):,} dari menu yang bergantung pada bahan ini."
                ),
                "next_step": (
                    f"Hubungi supplier {d.item_name} sekarang dan minta kirim minimal "
                    f"{n['min_reorder_qty']} {n['unit']} hari ini. Kalau supplier tidak bisa antar, "
                    f"minta seseorang beli di grosir terdekat — harga sedikit lebih mahal tidak masalah, "
                    f"daripada menu kosong dan pelanggan kecewa."
                ),
            }
        return {
            "why_it_matters": (
                f"{d.item_name} has only {n['stock']} {n['unit']} left with a daily usage of "
                f"{n['daily_usage']} {n['unit']}/day — it runs out in {n['days_remaining']} days. "
                f"This is likely a core ingredient for your top-selling menu items, meaning a stockout "
                f"directly halts production."
            ),
            "business_impact": (
                f"If not restocked today, you risk being unable to serve dishes that use {d.item_name} "
                f"starting tomorrow. With average daily revenue of "
                f"Rp{d.extra_context.get('avg_daily_sales', 0):,}, a single day of empty-menu "
                f"disruption could cost Rp{int(d.extra_context.get('avg_daily_sales', 0) * 0.3):,}–"
                f"Rp{int(d.extra_context.get('avg_daily_sales', 0) * 0.5):,} in lost sales."
            ),
            "next_step": (
                f"Call your {d.item_name} supplier now and request delivery of at least "
                f"{n['min_reorder_qty']} {n['unit']} today. If the supplier can't deliver, "
                f"send someone to the nearest wholesaler — paying slightly more is far better "
                f"than losing customers to an empty menu."
            ),
        }

    if d.situation_type == "stockout_medium":
        if lang == "id":
            return {
                "why_it_matters": (
                    f"{d.item_name} tersisa {n['stock']} {n['unit']} — cukup untuk sekitar "
                    f"{n['days_remaining']} hari lagi. Ini masih aman untuk hari ini, tapi "
                    f"kalau pemesanan tertunda atau supplier telat, bisa jadi masalah."
                ),
                "business_impact": (
                    f"Tanpa reorder minggu ini, risiko kehabisan stok di akhir minggu meningkat. "
                    f"Pesan terlambat sering berarti harga lebih tinggi atau kualitas lebih rendah "
                    f"karena terpaksa beli di pengecer, bukan grosir."
                ),
                "next_step": (
                    f"Masukkan pemesanan {d.item_name} di agenda hari ini — minimal "
                    f"{n['min_reorder_qty']} {n['unit']} untuk cadangan 1 minggu. Tidak perlu panik, "
                    f"tapi jangan ditunda lebih dari besok."
                ),
            }
        return {
            "why_it_matters": (
                f"{d.item_name} has {n['stock']} {n['unit']} left — roughly {n['days_remaining']} days. "
                f"Safe for today, but any supplier delay or unexpected demand spike could cause a shortfall."
            ),
            "business_impact": (
                f"Without a reorder this week, you risk end-of-week stockout pressure. Last-minute "
                f"buying at retail prices rather than wholesale usually costs 15–25% more."
            ),
            "next_step": (
                f"Schedule a {d.item_name} order today for at least {n['min_reorder_qty']} {n['unit']} "
                f"(1-week buffer). No urgency, but don't let it slip past tomorrow."
            ),
        }

    if d.situation_type == "losing_product":
        if lang == "id":
            return {
                "why_it_matters": (
                    f"{d.item_name} menghasilkan pendapatan Rp{n['revenue_7d']:,} dalam 7 hari terakhir, "
                    f"tapi biaya bahan bakunya Rp{n['cost_7d']:,} — artinya setiap hari Anda menjual "
                    f"{d.item_name}, warung rugi Rp{n['daily_loss']:,}. Margin negatif {abs(n['margin_pct'])}% "
                    f"ini berarti semakin laku produk ini, semakin besar kerugiannya."
                ),
                "business_impact": (
                    f"Jika tidak diperbaiki bulan ini, kerugian dari {d.item_name} saja bisa mencapai "
                    f"Rp{n['monthly_loss']:,} — uang itu seharusnya bisa jadi keuntungan bersih "
                    f"atau modal untuk produk lain. Produk lain yang untung secara diam-diam "
                    f"'menyubsidi' kerugian ini tanpa Anda sadari."
                ),
                "next_step": (
                    f"Naikan harga {d.item_name} sebesar 20–25% minggu ini, atau kurangi porsi/ukuran "
                    f"sekitar 15%. Coba dulu 3 hari dan lihat respons pelanggan — biasanya mereka "
                    f"tidak terlalu protes selama rasanya tetap enak. Kalau harga tidak bisa naik, "
                    f"pertimbangkan untuk tidak menjual {d.item_name} sama sekali."
                ),
            }
        return {
            "why_it_matters": (
                f"{d.item_name} earned Rp{n['revenue_7d']:,} over the last 7 days but cost "
                f"Rp{n['cost_7d']:,} to produce — meaning every day you sell {d.item_name}, "
                f"the shop loses Rp{n['daily_loss']:,}. A {abs(n['margin_pct'])}% negative margin "
                f"means the more popular this item is, the bigger the loss."
            ),
            "business_impact": (
                f"Left unchanged for a month, {d.item_name} alone could drain Rp{n['monthly_loss']:,} "
                f"from your profits. Your other profitable products are silently cross-subsidising "
                f"this loss without you realising it."
            ),
            "next_step": (
                f"Raise the price of {d.item_name} by 20–25% this week, or reduce the portion size "
                f"by about 15%. Trial it for 3 days — customers rarely complain if the taste stays "
                f"the same. If you can't raise the price, seriously consider removing it from the menu."
            ),
        }

    if d.situation_type == "low_margin":
        if lang == "id":
            return {
                "why_it_matters": (
                    f"{d.item_name} masih menguntungkan tapi marginnya sangat tipis ({n['margin_pct']}%). "
                    f"Sedikit kenaikan harga bahan baku dari supplier sudah bisa bikin produk ini merugi."
                ),
                "business_impact": (
                    f"Produk margin tipis sangat rentan terhadap fluktuasi harga bahan baku. "
                    f"Jika harga supplier naik 5–10% saja, {d.item_name} langsung jadi produk merugi."
                ),
                "next_step": (
                    f"Review harga jual {d.item_name} dan coba naikkan Rp500–Rp1.000 per unit. "
                    f"Atau cari supplier alternatif untuk bahan bakunya agar HPP bisa turun."
                ),
            }
        return {
            "why_it_matters": (
                f"{d.item_name} is profitable but with a very thin margin of {n['margin_pct']}%. "
                f"Any small supplier price increase could push it into loss territory."
            ),
            "business_impact": (
                f"Thin-margin products are fragile. A 5–10% rise in ingredient costs would make "
                f"{d.item_name} immediately loss-making."
            ),
            "next_step": (
                f"Review {d.item_name}'s selling price and consider a small Rp500–Rp1,000 increase "
                f"per unit, or find an alternative supplier to reduce your COGS."
            ),
        }

    if d.situation_type == "low_cash":
        if lang == "id":
            return {
                "why_it_matters": (
                    f"Saldo kas warung saat ini Rp{n['cash_balance']:,}, sementara rata-rata pengeluaran "
                    f"operasional harian sekitar Rp{n['avg_daily_sales']:,}. Artinya kas hanya cukup "
                    f"untuk {n['days_covered']} hari ke depan tanpa pemasukan tambahan."
                ),
                "business_impact": (
                    f"Jika kas habis sebelum omzet masuk, warung bisa kesulitan beli bahan baku "
                    f"esok hari. Ini bisa memaksa pembelian hutang atau harga eceran yang lebih mahal, "
                    f"yang akan menekan margin lebih jauh lagi."
                ),
                "next_step": (
                    f"Hari ini, tunda pembelian stok yang tidak mendesak dan fokus kumpulkan "
                    f"semua piutang dari pelanggan yang belum bayar. Target: pastikan kas di atas "
                    f"Rp{int(n['avg_daily_sales'] * 3):,} (3 hari operasional) sebelum minggu depan."
                ),
            }
        return {
            "why_it_matters": (
                f"Current cash balance is Rp{n['cash_balance']:,} against average daily operations "
                f"of Rp{n['avg_daily_sales']:,} — the cash will only last {n['days_covered']} more days "
                f"without incoming revenue."
            ),
            "business_impact": (
                f"If cash runs dry before revenue comes in, you may be unable to purchase ingredients "
                f"the next day, forcing credit purchases or retail-price buying — both of which "
                f"compress margins further."
            ),
            "next_step": (
                f"Today, delay any non-urgent stock purchases and actively collect any outstanding "
                f"customer credit. Target: build cash reserves back above "
                f"Rp{int(n['avg_daily_sales'] * 3):,} (3 days of operations) before end of week."
            ),
        }

    # Generic fallback for future situation types
    return {
        "why_it_matters": f"A situation requiring attention was detected for {d.item_name}.",
        "business_impact": "This may affect business operations if not addressed.",
        "next_step": "Review the detected issue and take appropriate action.",
    }


def _enrich_with_reasoning(
    detection: Detection,
    state: Dict[str, Any],
    language: str,
) -> Dict[str, str]:
    """
    Calls the 70B reasoning model to produce structured analysis for a single
    detection. In MOCK_MODE, returns a realistic mock instead of hitting the API.
    """
    if settings.mock_mode:
        return _mock_enrich(detection, language)

    system = _ENRICH_SYSTEM_ID if language == "id" else _ENRICH_SYSTEM_EN
    user   = _build_enrichment_user_prompt(detection, state, language)

    reasoner = ReasoningAgent()
    raw      = reasoner.call(system, user, language)
    return _parse_enrichment(raw, language)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_proactive_check(user_id: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scheduled proactive check — NOT triggered by user input.

    `language`, if given as "id" or "en", overrides the business's stored
    default language for the enrichment narrative (why_it_matters /
    business_impact / next_step). Falls back to the business's stored
    language when not provided.

    Steps:
      1. Detect all situations from the business state snapshot
      2. Filter to those above REASONING_THRESHOLD (avoids wasting GPU on noise)
      3. Enrich each qualifying detection with 70B reasoning analysis
      4. Return sorted by severity (most critical first)

    Returns a list of enriched suggestion dicts, each containing:
      - detection: the raw structured detection (for logging / downstream use)
      - analysis:  the reasoning agent's natural-language output (user-facing)
      - message:   convenience alias → analysis["next_step"] (for simple display)
    """
    state = get_business_state(user_id)
    if not state:
        return []

    language = language if language in ("id", "en") else state.get("language", "id")

    # 1. Detect
    all_detections = _detect_situations(state, language)

    # 2. Filter by severity threshold
    qualifying = [d for d in all_detections if d.severity_score >= REASONING_THRESHOLD]

    if not qualifying:
        return []

    # 3. Sort — most critical situations first
    qualifying.sort(key=lambda d: d.severity_score, reverse=True)

    # 4. Enrich each detection with reasoning-agent analysis
    suggestions: List[Dict[str, Any]] = []
    for i, detection in enumerate(qualifying):
        analysis = _enrich_with_reasoning(detection, state, language)
        suggestions.append({
            "priority":  i + 1,
            "category":  detection.category,
            "severity":  detection.severity,
            "severity_score": detection.severity_score,
            "item":      detection.item_name,
            "user_id":   user_id,
            "business":  state["business_name"],
            # Raw detection data — useful for logging, dashboards, or downstream agents
            "detection": {
                "situation_type": detection.situation_type,
                "numbers":        detection.numbers,
            },
            # Reasoning agent's structured analysis — the user-facing content
            "analysis": analysis,
            # Convenience alias for simple single-line display
            "message": analysis.get("next_step", ""),
        })

    return suggestions
