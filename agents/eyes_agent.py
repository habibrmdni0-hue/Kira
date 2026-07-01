"""
EyesAgent — hybrid OCR pipeline for warung receipts and handwritten notes.

TWO-TIER ARCHITECTURE (mirrors the threshold-gating pattern in proactive.py):

  Tier 1  LOCAL  (every scan, free, fast)
  ─────────────────────────────────────────────────────────────────────
  1. Preprocess image with OpenCV:
       grayscale → denoise → adaptive threshold → deskew
  2. Run pytesseract and capture per-word confidence scores
  3. Compute overall confidence (average of non-empty word scores)
  4. If confidence ≥ EYES_CONFIDENCE_THRESHOLD → parse text, return result

  Tier 2  CLOUD FALLBACK  (only when local confidence is too low)
  ─────────────────────────────────────────────────────────────────────
  5. Send image (base64) + structured extraction prompt to a vision LLM
     via an OpenAI-compatible endpoint (VISION_BASE_URL/VISION_MODEL)
  6. Parse the model's JSON response and return it

Both tiers produce the same output schema so callers never know which ran.
MOCK_MODE bypasses both tiers with deterministic mock data.

AMD / cloud vision endpoint:
  Set VISION_BASE_URL + VISION_MODEL in .env — no code changes needed.
  Works with GPT-4o-mini, LLaVA on Fireworks AI, or any compatible endpoint.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from .base import BaseAgent, AgentRequest, AgentResponse


# ──────────────────────────────────────────────────────────────
# Lazy imports — optional at module import time so the
# orchestrator doesn't crash when packages aren't installed.
# ImportError surfaces only when handle() is actually called
# in non-mock mode.
# ──────────────────────────────────────────────────────────────

def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for local OCR. "
            "Run: pip install opencv-python"
        )

def _numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required: pip install numpy")

def _tesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is required for local OCR. "
            "Run: pip install pytesseract  (also install the Tesseract binary — "
            "see README for platform instructions)"
        )


# ──────────────────────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────────────────────

def _load_image_bytes(payload: str) -> Tuple[bytes, str]:
    """
    Load image bytes + base64 string from payload.
    Accepts: data URI, file path, or raw base64 string.
    """
    if payload.startswith("data:image"):
        b64 = payload.split(",", 1)[1]
        return base64.b64decode(b64), b64

    if os.path.isfile(payload):
        with open(payload, "rb") as f:
            raw = f.read()
        return raw, base64.b64encode(raw).decode()

    try:
        raw = base64.b64decode(payload, validate=False)
        return raw, payload
    except Exception:
        raise ValueError("Payload is not a data URI, file path, or valid base64 string")


# ──────────────────────────────────────────────────────────────
# Tier 1 — preprocessing
# ──────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes):
    """
    grayscale → denoise → adaptive threshold → deskew.
    Returns a preprocessed numpy array ready for Tesseract.
    """
    cv2 = _cv2()
    np  = _numpy()

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — payload is not a valid image")

    # Step 1: grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: denoise (mild — preserves thin handwriting strokes)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 3: adaptive threshold — handles phone-photo receipts with uneven lighting
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # Step 4: deskew
    return _deskew(thresh)


def _deskew(image):
    """Detect rotation via minimum-area-rectangle and correct up to ±45°."""
    cv2 = _cv2()
    np  = _numpy()

    # Dark pixels = text (adaptive threshold produces white background)
    coords = np.column_stack(np.where(image < 200))
    if len(coords) < 20:
        return image  # not enough content to estimate angle safely

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image  # negligible skew — skip warp to preserve quality

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ──────────────────────────────────────────────────────────────
# Tier 1 — OCR + confidence scoring
# ──────────────────────────────────────────────────────────────

def _run_local_ocr(image, language: str) -> Tuple[str, float]:
    """
    Run pytesseract and return (raw_text, overall_confidence 0-100).

    Uses ind+eng for Indonesian requests so Indonesian item names OCR correctly
    while numbers (always ASCII) stay accurate.
    PSM 6 = uniform block of text — best for structured receipts.
    """
    pytesseract = _tesseract()

    lang_code  = "ind+eng" if language == "id" else "eng"
    tess_cfg   = "--psm 6"

    data = pytesseract.image_to_data(
        image,
        lang=lang_code,
        config=tess_cfg,
        output_type=pytesseract.Output.DICT,
    )
    # Confidence column: -1 = non-text region, 0-100 = word confidence
    word_confs = [int(c) for c in data["conf"] if int(c) >= 0]
    overall    = round(sum(word_confs) / len(word_confs), 1) if word_confs else 0.0

    raw_text = pytesseract.image_to_string(image, lang=lang_code, config=tess_cfg)
    return raw_text.strip(), overall


# ──────────────────────────────────────────────────────────────
# Tier 1 — receipt text parser
# ──────────────────────────────────────────────────────────────

def _parse_receipt_text(text: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Parse OCR text into structured items + total.

    Handles common Indonesian warung receipt formats:
      Printed: "Gula Pasir  10 kg  @ 14.500  =  145.000"
      Handwritten: "gula 10kg 14500 = 145rb"
      Two-column: "Tepung  3  33000"
    """
    items: List[Dict[str, Any]] = []
    total: float = 0.0

    def _to_num(s: str) -> float:
        """14.500 or 14,500 or Rp14.500 or 14rb → float"""
        s = re.sub(r'[Rr][Pp]', '', s)
        s = s.replace('.', '').replace(',', '').strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    def _expand_shorthand(s: str) -> str:
        """rb (ribu) and jt (juta) shorthand → full numbers"""
        s = re.sub(r'(\d+(?:\.\d+)?)\s*rb\b',
                   lambda m: str(int(float(m.group(1)) * 1000)), s, flags=re.IGNORECASE)
        s = re.sub(r'(\d+(?:\.\d+)?)\s*jt\b',
                   lambda m: str(int(float(m.group(1)) * 1_000_000)), s, flags=re.IGNORECASE)
        return s

    lines = _expand_shorthand(text).splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ── Total line ──────────────────────────────────────────
        if re.search(r'\b(total|jumlah|bayar|grand\s*total)\b', line, re.IGNORECASE):
            nums = re.findall(r'\d[\d.]*', line)
            for n in nums:
                candidate = _to_num(n)
                if candidate > total:
                    total = candidate
            continue

        # ── Pattern A: NAME  QTY [unit] [@] UNIT_PRICE [=] SUBTOTAL ──
        # "Gula Pasir  10 kg  @ 14.500  =  145.000"
        m = re.match(
            r'^([A-Za-z][A-Za-z\s]{1,30}?)\s+'         # item name
            r'(\d+(?:\.\d+)?)\s*'                        # quantity
            r'(?:kg|ltr?|liter|gr?|g|pcs|bks|btl|pack|bh|butir|biji)?\s*'  # optional unit
            r'@?\s*([\d.]{4,})\s*[=\-]?\s*([\d.]{4,})', # unit_price  subtotal
            line, re.IGNORECASE,
        )
        if m:
            name      = m.group(1).strip()
            qty       = float(m.group(2))
            uprice    = _to_num(m.group(3))
            subtotal  = _to_num(m.group(4))
            if len(name) >= 2 and qty > 0 and uprice > 0:
                items.append({"name": name, "quantity": qty, "unit_price": uprice})
                total += subtotal or (qty * uprice)
            continue

        # ── Pattern B: NAME  QTY  PRICE  (two-column warung notes) ──
        # "Tepung  3  11000" or "Telur  30 biji  58000"
        m2 = re.match(
            r'^([A-Za-z][A-Za-z\s]{1,30}?)\s+'
            r'(\d+(?:\.\d+)?)\s*'
            r'(?:kg|ltr?|liter|gr?|g|pcs|bks|btl|pack|bh|butir|biji)?\s+'
            r'([\d.]{4,})$',
            line, re.IGNORECASE,
        )
        if m2:
            name  = m2.group(1).strip()
            qty   = float(m2.group(2))
            price = _to_num(m2.group(3))
            if len(name) >= 2 and qty > 0 and price > 0:
                # Heuristic: if price / qty is a clean integer it's likely a unit price
                if qty > 1 and price % qty == 0:
                    items.append({"name": name, "quantity": qty, "unit_price": price / qty})
                    total += price
                else:
                    items.append({"name": name, "quantity": qty, "unit_price": price})
                    total += qty * price
            continue

    # Last-resort total: take largest number seen if no TOTAL line found
    if not total and text:
        candidates = [_to_num(n) for n in re.findall(r'\d[\d.]{4,}', text)]
        if candidates:
            total = max(candidates)

    return items, total


# ──────────────────────────────────────────────────────────────
# Tier 2 — cloud vision fallback
# ──────────────────────────────────────────────────────────────
#
# AMD / cloud endpoint:
#   Set VISION_BASE_URL and VISION_MODEL in .env.
#   Any OpenAI-compatible vision endpoint works:
#     gpt-4o, gpt-4o-mini, LLaVA on Fireworks, etc.
#   No code changes needed to switch providers.
# ──────────────────────────────────────────────────────────────

_VISION_SYSTEM = "You are a precise receipt OCR engine. Output only valid JSON, no prose."

_VISION_PROMPT_ID = """\
Ekstrak semua item dari gambar nota/struk ini.
Kembalikan HANYA JSON valid dalam format berikut — tanpa penjelasan lain:

{
  "items": [{"name": "nama item", "quantity": 1.0, "unit_price": 14500}],
  "total": 268000
}

Aturan: quantity dan unit_price harus angka. Harga dalam Rupiah tanpa simbol.
Jika tidak terbaca, gunakan 0.
"""

_VISION_PROMPT_EN = """\
Extract all items from this receipt or handwritten note.
Return ONLY valid JSON in this exact format — no explanation:

{
  "items": [{"name": "item name", "quantity": 1.0, "unit_price": 14500}],
  "total": 268000
}

Rules: quantity and unit_price must be numbers. Prices in Rupiah as integers.
Use 0 for anything illegible.
"""


def _run_cloud_ocr(image_b64: str, language: str) -> Dict[str, Any]:
    """
    Send image to cloud vision model and return parsed extraction dict.
    Raises on network or API failure (caller handles graceful degradation).
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=settings.vision_base_url,
        api_key=settings.vision_api_key,
    )
    prompt = _VISION_PROMPT_ID if language == "id" else _VISION_PROMPT_EN

    response = client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0,
    )

    raw = response.choices[0].message.content
    # Extract JSON even if model wraps it in markdown fences
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return {"items": [], "total": 0}
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {"items": [], "total": 0}


# ──────────────────────────────────────────────────────────────
# Mock data — realistic outputs for both tiers
# Used when MOCK_MODE=true; no API or Tesseract call made.
# ──────────────────────────────────────────────────────────────

_MOCK_TIER1_ITEMS = [
    {"name": "Gula Pasir",    "quantity": 10, "unit_price": 14500},
    {"name": "Minyak Goreng", "quantity": 5,  "unit_price": 18000},
    {"name": "Tepung Terigu", "quantity": 3,  "unit_price": 11000},
]

# Cloud tier finds an extra item the blurry local OCR missed
_MOCK_TIER2_ITEMS = [
    {"name": "Gula Pasir",    "quantity": 10, "unit_price": 14500},
    {"name": "Minyak Goreng", "quantity": 5,  "unit_price": 18000},
    {"name": "Tepung Terigu", "quantity": 3,  "unit_price": 11000},
    {"name": "Telur",         "quantity": 1,  "unit_price": 58000},  # missed by local OCR
]

_MOCK_TIER1_RAW_TEXT = (
    "Toko Sembako Maju\n"
    "15 Januari 2024\n"
    "Gula Pasir  10 kg  @ 14.500  =  145.000\n"
    "Minyak Goreng  5 lt  @ 18.000  =  90.000\n"
    "Tepung Terigu  3 kg  @ 11.000  =  33.000\n"
    "TOTAL  268.000"
)

# Deliberately garbled — simulates what Tesseract returns from a dark, blurry photo
_MOCK_TIER2_RAW_TEXT = (
    "[Tier 1 output — confidence 28.1, below threshold]\n"
    "T0k0 S3mb4k0 M4ju\n"
    "Gul4 P4s1r  10 kg  @ l4.5O0  =  l45.O00\n"
    "M1ny4k G0r3ng  5 lt  @ l8.O0O  =  9O.OOO\n"
    "T3pung T3r1gu  3 kg  @ ll.OO0  =  33.OOO\n"
    "T3lur  1 tr4y  @ 58.OOO  =  58.OOO\n"
    "[Cloud vision recovered structured data above]"
)


def _mock_tier1_extraction() -> Dict[str, Any]:
    total = sum(i["quantity"] * i["unit_price"] for i in _MOCK_TIER1_ITEMS)
    return {
        "source_tier": "local",
        "confidence":  82.4,
        "items":       _MOCK_TIER1_ITEMS,
        "total":       int(total),
        "raw_text":    _MOCK_TIER1_RAW_TEXT,
    }


def _mock_tier2_extraction() -> Dict[str, Any]:
    total = sum(i["quantity"] * i["unit_price"] for i in _MOCK_TIER2_ITEMS)
    return {
        "source_tier": "cloud_fallback",
        "confidence":  28.1,          # local confidence that triggered escalation
        "items":       _MOCK_TIER2_ITEMS,
        "total":       int(total),
        "raw_text":    _MOCK_TIER2_RAW_TEXT,
    }


# ──────────────────────────────────────────────────────────────
# EyesAgent
# ──────────────────────────────────────────────────────────────

class EyesAgent(BaseAgent):
    name = "eyes_agent"

    def handle(self, request: AgentRequest) -> AgentResponse:
        if settings.mock_mode:
            extraction = self._mock_extract(request)
        else:
            extraction = self._real_extract(request)

        return self._build_response(extraction, request.language)

    # ── Mock path ──────────────────────────────────────────────

    def _mock_extract(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Simulate tier selection based on payload keywords.
        Keywords that suggest a bad image force tier 2 escalation.
        All other payloads simulate a clean scan handled by tier 1.
        """
        payload_lower = request.payload.lower()
        bad_image_signals = ("blurry", "blur", "buram", "jelek", "kotor",
                             "messy", "gelap", "dark", "low_quality", "blurred")
        if any(sig in payload_lower for sig in bad_image_signals):
            return _mock_tier2_extraction()
        return _mock_tier1_extraction()

    # ── Real path ──────────────────────────────────────────────

    def _real_extract(self, request: AgentRequest) -> Dict[str, Any]:
        if request.input_type != "image":
            return {
                "source_tier": "local",
                "confidence":  0.0,
                "items":       [],
                "total":       0,
                "raw_text":    "[No image — set input_type='image' and pass base64/path as payload]",
            }

        image_bytes, image_b64 = _load_image_bytes(request.payload)

        # ── Tier 1: preprocess + local OCR ────────────────────
        raw_text  = ""
        confidence = 0.0
        try:
            preprocessed = _preprocess(image_bytes)
            raw_text, confidence = _run_local_ocr(preprocessed, request.language)
        except Exception as exc:
            # Tesseract binary missing, cv2 failure, etc.
            # confidence stays 0 → falls through to tier 2
            raw_text = f"[Local OCR error: {exc}]"

        if confidence >= settings.eyes_confidence_threshold:
            items, total = _parse_receipt_text(raw_text)
            return {
                "source_tier": "local",
                "confidence":  confidence,
                "items":       items,
                "total":       int(total),
                "raw_text":    raw_text,
            }

        # ── Tier 2: cloud vision fallback ─────────────────────
        try:
            cloud = _run_cloud_ocr(image_b64, request.language)
            return {
                "source_tier": "cloud_fallback",
                "confidence":  confidence,      # original local score (kept for audit)
                "items":       cloud.get("items", []),
                "total":       int(cloud.get("total", 0)),
                "raw_text":    raw_text,         # local OCR text (kept for debugging)
            }
        except Exception as cloud_exc:
            # Cloud also failed — return best-effort from local OCR
            items, total = _parse_receipt_text(raw_text)
            return {
                "source_tier": "local",          # degraded
                "confidence":  confidence,
                "items":       items,
                "total":       int(total),
                "raw_text":    raw_text,
                "error":       f"Cloud fallback failed: {cloud_exc}",
            }

    # ── Response assembly (common to both paths) ───────────────

    def _build_response(self, extraction: Dict[str, Any], language: str) -> AgentResponse:
        items    = extraction.get("items", [])
        total    = extraction.get("total", 0)
        tier     = extraction.get("source_tier", "local")
        conf     = extraction.get("confidence", 0)
        n_items  = len(items)

        if language == "id":
            tier_label = "lokal" if tier == "local" else "cloud (fallback)"
            summary = (
                f"Nota dibaca via OCR {tier_label} (conf {conf:.0f}%): "
                f"{n_items} item, total Rp{total:,}."
            )
        else:
            tier_label = tier.replace("_", " ")
            summary = (
                f"Receipt parsed via {tier_label} OCR (conf {conf:.0f}%): "
                f"{n_items} item(s), total Rp{total:,}."
            )

        # Keep 'summary' for the synthesize node in orchestrator.py
        return AgentResponse(
            agent_name=self.name,
            result={**extraction, "summary": summary},
        )
