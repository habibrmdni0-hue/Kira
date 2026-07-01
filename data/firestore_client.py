"""
data/firestore_client.py — Kira's unified data access layer.

THE ONLY PLACE that touches Firestore. All four agents (inventory, strategy,
bookkeeper, proactive) call into this module instead of maintaining their own
hardcoded data dicts.

Connection strategy (selected automatically at first call):
  FIREBASE_CREDENTIALS_JSON env var set → parse JSON inline (Railway deploy)
  FIREBASE_CREDENTIALS_PATH set + file exists → read JSON from file (local dev)
  Otherwise → LOCAL_FALLBACK (mirrors seed data exactly, for local dev)

MOCK_MODE does NOT gate this module. Data reads are always live (Firestore or
local fallback). Only LLM calls respect MOCK_MODE. These are independent.

Firestore schema (collection: "businesses", document id = user_id):
  business_name:             str
  owner_name:                str
  language:                  str        "id" | "en"
  cash_balance:              number     Rp — current cash
  avg_daily_sales:           number     Rp — rolling 30-day average
  daily_operating_expenses:  number     Rp — rent, utilities, etc.
  inventory: [{
    name:          str,   stock: number, unit: str,
    daily_usage:   number, reorder_point: number
  }]
  products: [{
    name: str, revenue_7d: number, cost_7d: number, units_sold_7d: number
  }]
"""
from __future__ import annotations

import json
import os
import random as _random
from datetime import date as _date, timedelta as _td
from typing import Any, Dict, List, Optional

from config.settings import settings


# ──────────────────────────────────────────────────────────────
# Sales-history generator — used both here (LOCAL_FALLBACK) and
# in seed_firestore.py.  Pure stdlib, no project imports needed.
# Fixed reference date (2026-07-01) + fixed per-business seed
# ensures identical output on every run.
# ──────────────────────────────────────────────────────────────

_HISTORY_REF_DATE = _date(2026, 7, 1)   # today in the demo world


def generate_sales_history(
    base_revenue: int,
    trend_per_day: float,
    weekend_multiplier: float,
    base_transactions: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Return 30 daily sales entries ending on _HISTORY_REF_DATE.

    Args:
        base_revenue:        Day-0 target revenue (Rp)
        trend_per_day:       Linear multiplicative drift per day
                             (-0.005 = ≈5% decline over 30 days)
        weekend_multiplier:  Revenue boost on Sat/Sun (e.g. 1.20)
        base_transactions:   Day-0 transaction count
        seed:                RNG seed — different per business
    """
    rng   = _random.Random(seed)
    start = _HISTORY_REF_DATE - _td(days=29)   # 30 entries: start … ref

    result: List[Dict[str, Any]] = []
    for i in range(30):
        d              = start + _td(days=i)
        trend_factor   = 1.0 + trend_per_day * i
        weekend_factor = weekend_multiplier if d.weekday() >= 5 else 1.0
        noise_rev      = 1.0 + rng.uniform(-0.08, 0.08)
        noise_tx       = 1.0 + rng.uniform(-0.12, 0.12)

        rev = max(0, round(base_revenue * trend_factor * weekend_factor * noise_rev / 1000) * 1000)
        tx  = max(5, round(base_transactions * trend_factor * weekend_factor * noise_tx))

        result.append({"date": d.isoformat(), "revenue": rev, "transactions": tx})

    return result


# ──────────────────────────────────────────────────────────────
# Local fallback — identical to what seed_firestore.py writes.
# Provides per-user data differentiation even without credentials.
# ──────────────────────────────────────────────────────────────

_LOCAL_FALLBACK: Dict[str, Dict[str, Any]] = {
    # ── user_001: Warung Bu Sari ───────────────────────────────
    # Low cash, several stockouts imminent, one losing product
    "user_001": {
        "business_name": "Warung Bu Sari",
        "owner_name": "Bu Sari",
        "language": "id",
        "cash_balance": 250_000,
        "avg_daily_sales": 415_000,
        "daily_operating_expenses": 45_000,
        "inventory": [
            {"name": "Gula Pasir",    "stock": 2.0,  "unit": "kg",    "daily_usage": 1.5,  "reorder_point": 3.0},
            {"name": "Minyak Goreng", "stock": 5.0,  "unit": "liter", "daily_usage": 0.8,  "reorder_point": 2.0},
            {"name": "Tepung Terigu", "stock": 0.5,  "unit": "kg",    "daily_usage": 0.3,  "reorder_point": 1.0},
            {"name": "Telur",         "stock": 30.0, "unit": "butir", "daily_usage": 12.0, "reorder_point": 12.0},
        ],
        "products": [
            {"name": "Gorengan",     "revenue_7d": 315_000, "cost_7d": 378_000, "units_sold_7d": 70},
            {"name": "Minuman Es",   "revenue_7d": 546_000, "cost_7d": 126_000, "units_sold_7d": 130},
            {"name": "Nasi Bungkus", "revenue_7d": 840_000, "cost_7d": 588_000, "units_sold_7d": 140},
            {"name": "Rokok Eceran", "revenue_7d": 210_000, "cost_7d": 189_000, "units_sold_7d": 70},
        ],
        # ~400k/day, downward trend (explains the cash problem). seed=42
        "sales_history": generate_sales_history(420_000, -0.005, 1.20, 45, seed=42),
    },

    # ── user_002: Toko Pak Budi ────────────────────────────────
    # Medium cash, inventory pressure (Oil/Noodles), no losing products
    "user_002": {
        "business_name": "Toko Pak Budi",
        "owner_name": "Pak Budi",
        "language": "en",
        "cash_balance": 520_000,
        "avg_daily_sales": 245_000,
        "daily_operating_expenses": 35_000,
        "inventory": [
            {"name": "Rice",    "stock": 25.0, "unit": "kg",   "daily_usage": 8.0,  "reorder_point": 10.0},
            {"name": "Oil",     "stock": 3.0,  "unit": "liter","daily_usage": 1.2,  "reorder_point": 2.5},
            {"name": "Noodles", "stock": 10.0, "unit": "pack", "daily_usage": 5.0,  "reorder_point": 7.0},
        ],
        "products": [
            {"name": "Cooked Meals",   "revenue_7d": 980_000, "cost_7d": 630_000, "units_sold_7d": 140},
            {"name": "Drinks",         "revenue_7d": 420_000, "cost_7d": 105_000, "units_sold_7d": 300},
            {"name": "Packaged Goods", "revenue_7d": 315_000, "cost_7d": 294_000, "units_sold_7d": 90},
        ],
        # ~245k/day, flat. seed=43
        "sales_history": generate_sales_history(245_000, 0.0, 1.15, 30, seed=43),
    },

    # ── user_003: Kedai Kang Asep ──────────────────────────────
    # Healthy cash, Daging Sapi about to run out, two star products,
    # one loss-maker. Deliberately different from Bu Sari and Pak Budi
    # to prove per-user differentiation.
    "user_003": {
        "business_name": "Kedai Kang Asep",
        "owner_name": "Kang Asep",
        "language": "id",
        "cash_balance": 850_000,
        "avg_daily_sales": 320_000,
        "daily_operating_expenses": 30_000,
        "inventory": [
            {"name": "Daging Sapi", "stock": 1.5,  "unit": "kg",    "daily_usage": 0.8,  "reorder_point": 2.0},
            {"name": "Bumbu Bakso", "stock": 500.0,"unit": "gram",  "daily_usage": 200.0,"reorder_point": 300.0},
            {"name": "Mi Kuning",   "stock": 3.0,  "unit": "kg",    "daily_usage": 0.5,  "reorder_point": 1.0},
            {"name": "Tahu",        "stock": 50.0, "unit": "potong","daily_usage": 15.0, "reorder_point": 20.0},
        ],
        "products": [
            {"name": "Bakso Sapi", "revenue_7d": 1_400_000, "cost_7d": 560_000, "units_sold_7d": 200},
            {"name": "Mie Ayam",   "revenue_7d":   490_000, "cost_7d": 245_000, "units_sold_7d": 70},
            {"name": "Es Teh",     "revenue_7d":   350_000, "cost_7d":  70_000, "units_sold_7d": 250},
            {"name": "Gorengan",   "revenue_7d":    80_000, "cost_7d":  96_000, "units_sold_7d": 40},
        ],
        # ~330k/day avg, upward trend. seed=44
        "sales_history": generate_sales_history(280_000, 0.012, 1.25, 35, seed=44),
    },
}


# ──────────────────────────────────────────────────────────────
# Firestore connection — lazy singleton
# ──────────────────────────────────────────────────────────────

_db = None          # firestore.Client once connected
_db_tried = False   # avoid re-trying after a failed init


def _get_db():
    """
    Return a Firestore client, or None if credentials are not configured.
    Connection is attempted once and the result is cached for the process lifetime.

    Credential resolution order:
      1. FIREBASE_CREDENTIALS_JSON env var (inline JSON — Railway / cloud deploy)
      2. FIREBASE_CREDENTIALS_PATH file path (local dev)
      3. Neither set → return None, use LOCAL_FALLBACK
    """
    global _db, _db_tried
    if _db_tried:
        return _db

    _db_tried = True

    creds_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
    creds_path     = getattr(settings, "firebase_credentials_path", "") or ""

    if not creds_json_str and (not creds_path or not os.path.isfile(creds_path)):
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore as fb_firestore

        if not firebase_admin._apps:
            if creds_json_str:
                cred   = credentials.Certificate(json.loads(creds_json_str))
                source = "FIREBASE_CREDENTIALS_JSON env var"
            else:
                cred   = credentials.Certificate(creds_path)
                source = creds_path
            firebase_admin.initialize_app(cred)

        _db = fb_firestore.client()
        print(f"[firestore_client] Connected to Firestore ({source})")
    except Exception as exc:
        print(f"[firestore_client] Could not connect to Firestore: {exc}")
        print("[firestore_client] Using local fallback data.")
        _db = None

    return _db


def firestore_status() -> str:
    """Return 'connected' if Firestore is live, 'fallback' if using local data."""
    return "connected" if _get_db() is not None else "fallback"


# ──────────────────────────────────────────────────────────────
# Internal: fetch raw Firestore document OR local fallback
# ──────────────────────────────────────────────────────────────

def _fetch_raw(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the raw Firestore document dict for user_id, or the local fallback.
    Returns None if the user_id is not found in either source.
    """
    db = _get_db()

    if db is not None:
        try:
            doc = db.collection("businesses").document(user_id).get()
            if doc.exists:
                return doc.to_dict()
            # Document not in Firestore — try local fallback for this user_id
            # (useful when Firestore is connected but seed hasn't been run yet)
        except Exception as exc:
            print(f"[firestore_client] Read error for {user_id}: {exc}")
            # Fall through to local fallback

    return _LOCAL_FALLBACK.get(user_id)


# ──────────────────────────────────────────────────────────────
# Internal: transform raw document → agent-compatible shape
# ──────────────────────────────────────────────────────────────

def _transform(raw: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Convert a raw Firestore/fallback document (uses 'name' field in arrays)
    into the dict shape that agents and proactive.py currently expect ('item' field).

    The returned dict also contains a '_products_raw' key so get_sales_history
    can extract structured product data without re-fetching.
    """
    inventory = [
        {
            "item":          row["name"],
            "stock":         float(row["stock"]),
            "unit":          row["unit"],
            "daily_usage":   float(row["daily_usage"]),
            "reorder_point": float(row.get("reorder_point", row["daily_usage"] * 2)),
        }
        for row in raw.get("inventory", [])
    ]

    recent_sales_7d = [
        {
            "item":    p["name"],
            "revenue": p["revenue_7d"],
            "cost":    p["cost_7d"],
        }
        for p in raw.get("products", [])
    ]

    return {
        "business_name":            raw.get("business_name", user_id),
        "owner":                    raw.get("owner_name", ""),
        "language":                 raw.get("language", "id"),
        "cash_balance":             raw.get("cash_balance", 0),
        "avg_daily_sales":          raw.get("avg_daily_sales", 0),
        "daily_operating_expenses": raw.get("daily_operating_expenses", 0),
        "inventory":                inventory,
        "recent_sales_7d":          recent_sales_7d,
        "_products_raw":            raw.get("products", []),        # for get_sales_history
        "sales_history":            raw.get("sales_history", []),   # daily time-series
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_business_state(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Full business state for user_id.

    Returned shape (matches proactive.py's existing state expectations):
      business_name, owner, language, cash_balance, avg_daily_sales,
      daily_operating_expenses, inventory[{item, stock, unit, daily_usage,
      reorder_point}], recent_sales_7d[{item, revenue, cost}]

    Returns None if user_id is not found.
    """
    raw = _fetch_raw(user_id)
    if raw is None:
        return None
    return _transform(raw, user_id)


def get_inventory(user_id: str) -> List[Dict[str, Any]]:
    """
    Inventory items for user_id.

    Each item: {item, stock, unit, daily_usage, reorder_point}

    Returns [] if user_id is not found.
    """
    state = get_business_state(user_id)
    return state["inventory"] if state else []


def get_sales_history(user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Product sales data for user_id.

    Each entry: {item, revenue_7d, cost_7d, units_sold_7d}

    The `days` parameter is reserved for future date-range queries on Firestore;
    currently all stored data covers 7 days regardless of the value passed.

    Returns [] if user_id is not found.
    """
    state = get_business_state(user_id)
    if not state:
        return []

    return [
        {
            "item":          p["name"],
            "revenue_7d":    p["revenue_7d"],
            "cost_7d":       p["cost_7d"],
            "units_sold_7d": p.get("units_sold_7d", 0),
        }
        for p in state.get("_products_raw", [])
    ]


def get_daily_sales_timeseries(user_id: str) -> List[Dict[str, Any]]:
    """
    30-day daily revenue time series for user_id, suitable for charting.

    Each entry: {"date": "YYYY-MM-DD", "revenue": int, "transactions": int}

    Returns [] if user_id is not found or has no sales_history yet.
    """
    state = get_business_state(user_id)
    return state.get("sales_history", []) if state else []
