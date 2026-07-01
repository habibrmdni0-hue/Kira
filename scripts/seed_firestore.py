"""
scripts/seed_firestore.py — populate Firestore with demo business data.

Run this ONCE after setting up credentials. It creates or overwrites
documents in the `businesses` collection. Safe to re-run — Firestore
upserts overwrite existing documents with the same user_id.

Usage:
    python scripts/seed_firestore.py

Prerequisites:
    1. FIREBASE_CREDENTIALS_PATH set in .env (path to service account JSON)
    2. firebase-admin installed:  pip install firebase-admin

What this creates:
    businesses/user_001  — Warung Bu Sari    (id, stressed)
    businesses/user_002  — Toko Pak Budi     (en, moderate)
    businesses/user_003  — Kedai Kang Asep   (id, healthy)

After seeding, run `python -X utf8 main.py` to verify that Demo 9 shows
genuinely different data for each user_id.
"""
import os
import sys

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

CREDS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")

if not CREDS_PATH:
    print("ERROR: FIREBASE_CREDENTIALS_PATH is not set in .env")
    print("  Add it to .env:  FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccount.json")
    sys.exit(1)

if not os.path.isfile(CREDS_PATH):
    print(f"ERROR: Credentials file not found: {CREDS_PATH}")
    print("  Download the service account JSON from Firebase Console:")
    print("  Project Settings → Service accounts → Generate new private key")
    sys.exit(1)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    print("ERROR: firebase-admin is not installed.")
    print("  Install it:  pip install firebase-admin")
    sys.exit(1)

# ── Connect ────────────────────────────────────────────────────
cred = credentials.Certificate(CREDS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Seed data (mirrors LOCAL_FALLBACK in firestore_client.py) ─
BUSINESSES = {
    # ── user_001: Warung Bu Sari ───────────────────────────────
    # Stressed: low cash, multiple imminent stockouts, one losing product
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
    },

    # ── user_002: Toko Pak Budi ────────────────────────────────
    # Moderate: medium cash, Oil + Noodles pressure, no losing products
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
    },

    # ── user_003: Kedai Kang Asep ──────────────────────────────
    # Healthy cash. Critical: Daging Sapi running out fast (1.9 days).
    # Two star products. One losing product (Gorengan, like Bu Sari but
    # different numbers). Proves genuinely different data per user_id.
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
    },
}


# ── Write to Firestore ─────────────────────────────────────────
print(f"\nSeeding {len(BUSINESSES)} businesses to Firestore (collection: 'businesses')...\n")

for user_id, data in BUSINESSES.items():
    doc_ref = db.collection("businesses").document(user_id)
    doc_ref.set(data)
    n_inv  = len(data["inventory"])
    n_prod = len(data["products"])
    print(f"  ✓  {user_id}  |  {data['business_name']}  |  {n_inv} inventory items, {n_prod} products")

print(f"\nDone. Run `python -X utf8 main.py` to verify per-user differentiation (Demo 9).\n")
