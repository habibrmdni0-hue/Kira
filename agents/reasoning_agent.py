"""
ReasoningAgent — calls a large open-source model (Llama 3.1 70B / Qwen 2.5 72B)
via an OpenAI-compatible API endpoint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AMD MI300X GPU INFERENCE — HOW TO PLUG IN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set these three .env variables — NO code changes required:

  REASONING_BASE_URL  = <AMD Developer Cloud or Fireworks AI endpoint>
  REASONING_API_KEY   = <your API key>
  REASONING_MODEL     = <model name on that endpoint>

Examples:
  # Fireworks AI (MI300X-backed)
  REASONING_BASE_URL=https://api.fireworks.ai/inference/v1
  REASONING_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct

  # AMD Developer Cloud (direct)
  REASONING_BASE_URL=https://api.amd.com/v1
  REASONING_MODEL=meta-llama/Llama-3.1-70B-Instruct

The client is a standard openai.OpenAI() instance — it works with any
endpoint that speaks the OpenAI Chat Completions API.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from config.settings import settings
from .base import BaseAgent, AgentRequest, AgentResponse


class ReasoningAgent(BaseAgent):
    name = "reasoning_agent"

    def __init__(self):
        self._client = None  # lazy-init to avoid import errors in mock mode

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=settings.reasoning_base_url,
                api_key=settings.reasoning_api_key,
            )
        return self._client

    def call(self, system_prompt: str, user_prompt: str, language: str = "en") -> str:
        """Direct reasoning call — used by the proactive layer and other agents."""
        if settings.mock_mode:
            return self._mock_response(user_prompt, language)

        client = self._get_client()
        response = client.chat.completions.create(
            model=settings.reasoning_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def handle(self, request: AgentRequest) -> AgentResponse:
        system = (
            "Kamu adalah analis bisnis warung berpengalaman. Jawab dalam Bahasa Indonesia."
            if request.language == "id"
            else "You are an expert warung business analyst. Answer in English."
        )
        result_text = self.call(system, request.payload, request.language)
        return AgentResponse(agent_name=self.name, result={"analysis": result_text})

    @staticmethod
    def _mock_response(prompt: str, language: str) -> str:
        prompt_lower = prompt.lower()
        if "stockout" in prompt_lower or "stok" in prompt_lower or "stock" in prompt_lower:
            return (
                "⚠️ PERINGATAN STOK:\n"
                "• Gula Pasir: habis dalam ~1.3 hari — segera pesan minimal 5 kg\n"
                "• Tepung Terigu: hampir habis (0.5 kg sisa) — pesan 3 kg\n"
                "REKOMENDASI: Hubungi Toko Sembako Maju hari ini sebelum tutup."
                if language == "id"
                else
                "⚠️ STOCK ALERT:\n"
                "• Sugar: runs out in ~1.3 days — order at least 5 kg immediately\n"
                "• Flour: almost gone (0.5 kg left) — order 3 kg\n"
                "RECOMMENDATION: Contact your supplier today before closing time."
            )
        if "rugi" in prompt_lower or "loss" in prompt_lower or "margin" in prompt_lower:
            return (
                "📉 ANALISIS PRODUK MERUGI:\n"
                "Gorengan menjual Rp315.000 tapi biaya produksi Rp378.000 → rugi Rp63.000/hari.\n"
                "SOLUSI: Naikkan harga jual 20% ATAU kurangi ukuran porsi 15%.\n"
                "Prioritas: ubah harga minggu ini sebelum kerugian terakumulasi."
                if language == "id"
                else
                "📉 LOSS-MAKING PRODUCT ANALYSIS:\n"
                "Fried snacks earn Rp315,000 but cost Rp378,000 → losing Rp63,000/day.\n"
                "SOLUTION: Raise selling price by 20% OR reduce portion size by 15%.\n"
                "Priority: adjust pricing this week before losses compound."
            )
        return (
            "Berdasarkan data bisnis Anda, semua indikator utama dalam batas normal. "
            "Pantau terus stok dan laba harian untuk deteksi dini masalah."
            if language == "id"
            else
            "Based on your business data, all key indicators are within normal range. "
            "Keep monitoring daily stock and profit for early problem detection."
        )
