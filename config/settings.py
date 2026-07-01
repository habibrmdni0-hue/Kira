import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ─────────────────────────────────────────────────────────
    # Router LLM — small, fast model for intent classification
    # ─────────────────────────────────────────────────────────
    router_base_url: str = os.getenv("ROUTER_BASE_URL", "https://api.openai.com/v1")
    router_api_key: str  = os.getenv("ROUTER_API_KEY",  "mock-key")
    router_model: str    = os.getenv("ROUTER_MODEL",    "gpt-4o-mini")

    # ─────────────────────────────────────────────────────────
    # Reasoning LLM — AMD MI300X GPU inference endpoint
    # Swap REASONING_BASE_URL + REASONING_MODEL in .env to
    # point at AMD Developer Cloud or Fireworks AI.
    # No code changes required; the client is OpenAI-compatible.
    # ─────────────────────────────────────────────────────────
    reasoning_base_url: str = os.getenv("REASONING_BASE_URL", "https://api.openai.com/v1")
    reasoning_api_key: str  = os.getenv("REASONING_API_KEY",  "mock-key")
    reasoning_model: str    = os.getenv(
        "REASONING_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct"
    )

    # ─────────────────────────────────────────────────────────
    # Vision LLM — cloud OCR fallback for EyesAgent (Tier 2)
    # Used only when local Tesseract confidence is too low.
    # Any OpenAI-compatible vision endpoint works:
    #   gpt-4o / gpt-4o-mini, LLaVA on Fireworks AI, etc.
    # Keep this separate from reasoning_* — vision models are
    # typically smaller and cheaper than the 70B reasoning model.
    # ─────────────────────────────────────────────────────────
    vision_base_url: str = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1")
    vision_api_key: str  = os.getenv("VISION_API_KEY",  "mock-key")
    vision_model: str    = os.getenv("VISION_MODEL",    "gpt-4o-mini")

    # ─────────────────────────────────────────────────────────
    # Voice LLM — conversational synthesizer (last pipeline step)
    # voice_agent receives structured outputs from ALL specialist
    # agents and turns them into one natural-language response.
    # Use a FAST, CHEAP conversational model here — the heavy 70B
    # analysis work is already done by reasoning_agent.
    # ─────────────────────────────────────────────────────────
    voice_base_url: str = os.getenv("VOICE_BASE_URL", "https://api.openai.com/v1")
    voice_api_key: str  = os.getenv("VOICE_API_KEY",  "mock-key")
    voice_model: str    = os.getenv("VOICE_MODEL",    "gpt-4o-mini")

    # ─────────────────────────────────────────────────────────
    # EyesAgent confidence threshold
    # Local OCR confidence below this value triggers Tier 2.
    # Range: 0–100. Lower = more escalations to cloud (better
    # recall, higher cost). Higher = fewer cloud calls (faster,
    # cheaper, but may miss messy handwriting).
    # ─────────────────────────────────────────────────────────
    eyes_confidence_threshold: float = float(
        os.getenv("EYES_CONFIDENCE_THRESHOLD", "60")
    )

    # ─────────────────────────────────────────────────────────
    # Firebase / Firestore — business data layer
    # Path to the service account JSON downloaded from:
    #   Firebase Console → Project Settings → Service accounts
    # If not set or file not found, the data layer falls back to
    # LOCAL_FALLBACK data (same values as the seed script).
    # MOCK_MODE does NOT affect Firestore reads — data is always
    # read from the real source regardless of LLM mock state.
    # ─────────────────────────────────────────────────────────
    firebase_credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")

    # ─────────────────────────────────────────────────────────
    # MOCK_MODE — bypass all LLM/OCR calls with deterministic
    # mock responses. Default true so tests run without API keys.
    # Does NOT affect Firestore data reads (separate concern).
    # ─────────────────────────────────────────────────────────
    mock_mode: bool = os.getenv("MOCK_MODE", "true").lower() == "true"


settings = Settings()
