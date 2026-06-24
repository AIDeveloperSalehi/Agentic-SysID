"""
Intake agent — parses the user's plant description and initialises the dossier.

LLM agent: uses the Anthropic SDK to parse natural-language input and produce
a PlantContract + initial Dossier.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    Assets,
    Budget,
    Dossier,
    EntryPath,
    PhysicsAvailability,
    PlantContract,
    Report,
)
from tools.model_registry import ModelRegistry
from core.llm_logger import LLMLogger


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "intake.md"


class IntakeAgent:
    """
    LLM-backed intake agent.

    Usage
    -----
    agent   = IntakeAgent(registry, budget_total=200.0)
    dossier = agent(dossier)   # dossier passed in is the initial empty one
    """

    def __init__(
        self,
        registry: ModelRegistry,
        budget_total: float = 200.0,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        llm_logger: Optional[LLMLogger] = None,
    ):
        self._registry   = registry
        self._budget     = budget_total
        self._model      = model
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._llm_logger = llm_logger

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        """Invoked by the LangGraph orchestrator."""
        plant_desc = dossier.status   # description stored in status by the pipeline runner
        report = self.run(plant_desc)
        meta = report.metadata

        entry_path = EntryPath(meta.get("entry_path", "white-box"))
        physics    = PhysicsAvailability(meta.get("physics", "none"))
        contract_id = meta.get("plant_contract_id", "")

        return dossier.update(
            entry_path=entry_path,
            status=f"intake done: {entry_path.value}",
            assets=Assets(
                plant_contract_id=contract_id,
                physics=physics,
            ),
            budget=Budget(total=self._budget),
            last_report=report,
        )

    # ── LLM run ───────────────────────────────────────────────────────────────

    def run(self, plant_description: str) -> Report:
        """Parse plant description, store PlantContract, return Report."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else \
            "You are the Intake agent. Parse the plant and call create_plant_contract, then post_report."

        tools = [
            {
                "name": "create_plant_contract",
                "description": "Store a PlantContract in the registry. Returns the contract ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name":         {"type": "string"},
                        "input_names":  {"type": "array",  "items": {"type": "string"}},
                        "output_names": {"type": "array",  "items": {"type": "string"}},
                        "state_names":  {"type": "array",  "items": {"type": "string"}},
                        "input_limits": {"type": "object"},
                        "sample_time":  {"type": "number"},
                        "description":  {"type": "string"},
                    },
                    "required": ["name", "input_names", "output_names",
                                 "input_limits", "sample_time"],
                },
            },
            _post_report_schema(),
        ]

        messages = [{"role": "user", "content": f"Plant description:\n{plant_description}"}]
        contract_id: Optional[str] = None

        _sep = "─" * 72
        log.debug("[IntakeAgent] %s", _sep)
        log.debug("[IntakeAgent] SYSTEM PROMPT\n%s\n%s", system_prompt, _sep)
        log.debug("[IntakeAgent] TASK MESSAGE\nPlant description:\n%s\n%s", plant_description, _sep)

        for iteration in range(8):
            log.debug("[IntakeAgent] iteration %d — calling LLM (%s)", iteration + 1, self._model)
            resp = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            if self._llm_logger is not None:
                self._llm_logger.log(
                    agent="IntakeAgent",
                    iteration=iteration + 1,
                    system=system_prompt,
                    messages=messages,
                    model=self._model,
                    tools=tools,
                    response=resp,
                )

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    log.debug("[IntakeAgent] LLM text:\n%s", block.text)

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                log.warning("[IntakeAgent] LLM returned no tool calls — stopping loop")
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []

            final_report: Optional[Report] = None
            for tc in tool_uses:
                log.debug("[IntakeAgent] tool call → %s\n  input: %s",
                          tc.name, json.dumps(tc.input, indent=2))
                if tc.name == "create_plant_contract":
                    inp = tc.input
                    # Build and store PlantContract
                    contract = PlantContract(
                        name=inp["name"],
                        input_names=inp["input_names"],
                        output_names=inp["output_names"],
                        state_names=inp.get("state_names", []),
                        input_limits={
                            k: tuple(v) for k, v in inp["input_limits"].items()
                        },
                        sample_time=float(inp["sample_time"]),
                        description=inp.get("description", ""),
                    )
                    # Store as a model artifact for easy retrieval
                    from core.schemas import ModelArtifact, ModelType
                    artifact = ModelArtifact(
                        id=contract.id,
                        model_type=ModelType.WHITE_BOX,
                        structure_description=f"PlantContract:{contract.name}",
                        metadata={"plant_contract": contract.model_dump()},
                    )
                    self._registry.store_model(artifact)
                    contract_id = contract.id
                    log.debug("[IntakeAgent] tool result ← create_plant_contract: contract_id=%s name=%s",
                              contract_id, contract.name)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps({"contract_id": contract_id, "name": contract.name}),
                    })

                elif tc.name == "post_report":
                    final_report = _build_report("IntakeAgent", tc.input)
                    if contract_id and "plant_contract_id" not in final_report.metadata:
                        final_report.metadata["plant_contract_id"] = contract_id
                    log.debug("[IntakeAgent] post_report → status=%s summary=%s",
                              tc.input.get("status"), tc.input.get("summary"))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": "Report posted.",
                    })

            messages.append({"role": "user", "content": results})
            if final_report:
                return final_report

        return Report(
            agent="IntakeAgent",
            status=AgentStatus.FAILED,
            summary="Intake did not complete within iteration limit.",
            metadata={"plant_contract_id": contract_id or ""},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post_report_schema() -> dict:
    return {
        "name": "post_report",
        "description": "Post the final report. Call ONCE when done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status":   {"type": "string", "enum": ["done", "needs_user_input", "failed"]},
                "summary":  {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["status", "summary"],
        },
    }


def _build_report(agent: str, raw: dict) -> Report:
    from agents.base_agent import _build_report as base_build
    return base_build(agent, raw)
