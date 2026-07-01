"""
KiraOrchestrator — the backbone that coordinates Kira's six specialist agents.

Architecture (LangGraph StateGraph):

  ┌──────────┐    ┌────────┐    ┌──────────────────────────────┐    ┌───────────┐
  │  intake  │───▶│ route  │───▶│  dispatch_agents             │───▶│ synthesize│
  │ (validate│    │ (LLM   │    │  (runs selected agents,      │    │ (build    │
  │  & parse)│    │  class.)│    │   collects responses)        │    │  reply)   │
  └──────────┘    └────────┘    └──────────────────────────────┘    └───────────┘

The proactive flow (run_proactive_check) bypasses this graph entirely and
calls the reasoning agent directly — it is designed to run on a schedule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agents import (
    AgentRequest,
    EyesAgent,
    BookkeeperAgent,
    InventoryAgent,
    StrategyAgent,
    VoiceAgent,
    ReasoningAgent,
)
from orchestrator.router import route as llm_route


# ──────────────────────────────────────────────────────────────
# Public request model (intake contract)
# ──────────────────────────────────────────────────────────────

@dataclass
class KiraRequest:
    payload: str
    user_id: str
    language: str = "id"        # "id" | "en"
    input_type: str = "text"    # "text" | "image" | "voice"


# ──────────────────────────────────────────────────────────────
# LangGraph state schema
# ──────────────────────────────────────────────────────────────

class KiraState(TypedDict):
    # Set at intake
    payload: str
    user_id: str
    language: str
    input_type: str

    # Set by router
    intent: str
    agents_to_invoke: List[str]

    # Populated as agents run
    agent_results: Dict[str, Any]

    # Final output
    final_response: str
    error: Optional[str]


# ──────────────────────────────────────────────────────────────
# Agent registry
# ──────────────────────────────────────────────────────────────

_AGENT_REGISTRY = {
    "eyes_agent":       EyesAgent(),
    "bookkeeper_agent": BookkeeperAgent(),
    "inventory_agent":  InventoryAgent(),
    "strategy_agent":   StrategyAgent(),
    "voice_agent":      VoiceAgent(),
    "reasoning_agent":  ReasoningAgent(),
}


# ──────────────────────────────────────────────────────────────
# Graph nodes
# ──────────────────────────────────────────────────────────────

def _node_intake(state: KiraState) -> KiraState:
    """Validate and normalise the incoming request."""
    language = state.get("language", "id")
    if language not in ("id", "en"):
        language = "id"
    return {**state, "language": language, "agent_results": {}, "error": None}


def _node_route(state: KiraState) -> KiraState:
    """Classify intent and decide which agents to invoke."""
    try:
        intent, agents = llm_route(state["payload"], state["language"])
    except Exception as exc:
        # Routing failure is non-fatal — fall back to voice agent
        intent, agents = "routing error — fallback", ["voice_agent"]
    return {**state, "intent": intent, "agents_to_invoke": agents}


def _node_dispatch_agents(state: KiraState) -> KiraState:
    """
    Run every selected agent and collect responses.

    Execution order is intentionally two-phase:
      Phase 1 — all specialist agents (inventory, bookkeeper, strategy,
                 reasoning, eyes) run with the raw user payload.
      Phase 2 — voice_agent runs LAST with a context dict that contains
                 every specialist agent's result.  This is what makes
                 voice_agent a genuine synthesizer rather than an echo.

    In production, Phase 1 agents can run in parallel (asyncio.gather or
    ThreadPoolExecutor). Sequential here keeps the demo dependency-free.
    """
    results: Dict[str, Any] = {}
    request = AgentRequest(
        payload=state["payload"],
        user_id=state["user_id"],
        language=state["language"],
        input_type=state["input_type"],
    )

    # ── Phase 1: specialist agents ─────────────────────────────
    for agent_name in state["agents_to_invoke"]:
        if agent_name == "voice_agent":
            continue  # handled in Phase 2
        agent = _AGENT_REGISTRY.get(agent_name)
        if agent is None:
            results[agent_name] = {"error": f"Unknown agent: {agent_name}"}
            continue
        try:
            response = agent.handle(request)
            results[agent_name] = response.to_dict()
        except Exception as exc:
            results[agent_name] = {"agent": agent_name, "success": False, "error": str(exc)}

    # ── Phase 2: voice_agent — synthesizes specialist results ──
    if "voice_agent" in state["agents_to_invoke"]:
        # Extract the `result` dict from each specialist response so
        # voice_agent gets clean data, not the full response envelope.
        voice_context = {
            name: res.get("result", {})
            for name, res in results.items()
        }
        voice_request = AgentRequest(
            payload=state["payload"],
            user_id=state["user_id"],
            language=state["language"],
            input_type=state["input_type"],
            context=voice_context,
        )
        voice_agent = _AGENT_REGISTRY["voice_agent"]
        try:
            voice_response = voice_agent.handle(voice_request)
            results["voice_agent"] = voice_response.to_dict()
        except Exception as exc:
            results["voice_agent"] = {
                "agent": "voice_agent", "success": False, "error": str(exc)
            }

    return {**state, "agent_results": results}


def _node_synthesize(state: KiraState) -> KiraState:
    """
    Build the final user-facing response.

    When voice_agent ran (Phase 2 of dispatch), it already synthesized
    all specialist results into one coherent natural-language response —
    that output IS the final response, and this node just surfaces it.

    When voice_agent was not invoked (edge case: routing chose only
    specialist agents), we fall back to the old structured bullet-list
    format so the pipeline never returns an empty response.
    """
    results  = state.get("agent_results", {})
    language = state["language"]

    # ── Primary path: voice_agent synthesized everything ───────
    voice_text = results.get("voice_agent", {}).get("result", {}).get("response_text", "")
    if voice_text:
        return {**state, "final_response": voice_text}

    # ── Fallback: no voice_agent — build structured output ─────
    lines: List[str] = []

    inv = results.get("inventory_agent", {}).get("result", {})
    for alert in inv.get("alerts", []):
        lines.append(f"⚠️  {alert['action']}")

    strat = results.get("strategy_agent", {}).get("result", {})
    for rec in strat.get("recommendations", []):
        lines.append(f"💡 {rec}")

    fin_summary = results.get("bookkeeper_agent", {}).get("result", {}).get("summary")
    if fin_summary:
        lines.append(f"📊 {fin_summary}")

    eye_summary = results.get("eyes_agent", {}).get("result", {}).get("summary")
    if eye_summary:
        lines.append(f"📄 {eye_summary}")

    analysis = results.get("reasoning_agent", {}).get("result", {}).get("analysis")
    if analysis:
        lines.append(f"\n🧠 {'Analisis mendalam' if language == 'id' else 'Deep analysis'}:\n{analysis}")

    final = "\n".join(lines) if lines else (
        "Tidak ada hasil yang ditemukan." if language == "id" else "No results found."
    )
    return {**state, "final_response": final}


# ──────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(KiraState)

    graph.add_node("intake",          _node_intake)
    graph.add_node("route",           _node_route)
    graph.add_node("dispatch_agents", _node_dispatch_agents)
    graph.add_node("synthesize",      _node_synthesize)

    graph.set_entry_point("intake")
    graph.add_edge("intake",          "route")
    graph.add_edge("route",           "dispatch_agents")
    graph.add_edge("dispatch_agents", "synthesize")
    graph.add_edge("synthesize",      END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────
# Public orchestrator class
# ──────────────────────────────────────────────────────────────

class KiraOrchestrator:
    """
    Entry point for all reactive (user-triggered) requests.

    Usage:
        orchestrator = KiraOrchestrator()
        result = orchestrator.run(KiraRequest(
            payload="Berapa stok gula saya?",
            user_id="user_001",
            language="id",
        ))
        print(result["final_response"])
    """

    def __init__(self):
        self._graph = _build_graph()

    def run(self, request: KiraRequest) -> Dict[str, Any]:
        """Process a user request through the full agent pipeline."""
        initial_state: KiraState = {
            "payload":        request.payload,
            "user_id":        request.user_id,
            "language":       request.language,
            "input_type":     request.input_type,
            "intent":         "",
            "agents_to_invoke": [],
            "agent_results":  {},
            "final_response": "",
            "error":          None,
        }
        final_state = self._graph.invoke(initial_state)
        return {
            "user_id":        final_state["user_id"],
            "language":       final_state["language"],
            "intent":         final_state["intent"],
            "agents_invoked": final_state["agents_to_invoke"],
            "agent_results":  final_state["agent_results"],
            "final_response": final_state["final_response"],
        }
