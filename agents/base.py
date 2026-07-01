from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentRequest:
    payload: str
    user_id: str
    language: str                        # "id" | "en"
    input_type: str = "text"             # "text" | "image" | "voice"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
        }


class BaseAgent(ABC):
    """All Kira agents implement this interface."""

    name: str = "base_agent"

    @abstractmethod
    def handle(self, request: AgentRequest) -> AgentResponse:
        """Process a request and return a structured response."""

    def _lang(self, id_text: str, en_text: str, language: str) -> str:
        """Helper: pick the right language string."""
        return id_text if language == "id" else en_text
