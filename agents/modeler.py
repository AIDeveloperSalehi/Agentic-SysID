"""
Modeler agent — derives the symbolic ODE structure and stores it in the registry.

LLM agent: uses physics knowledge to write the governing equations, then calls
the ID analyst service to check identifiability and reparameterize.
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
    Artifacts,
    Assets,
    Dossier,
    ModelArtifact,
    ModelType,
    PhysicsAvailability,
    Report,
)
from agents.id_analyst import IDAnalyst
from tools.model_registry import ModelRegistry
from core.llm_logger import LLMLogger


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "modeler.md"


class ModelerAgent:
    """
    LLM-backed modeler agent.

    Derives ODE structure → checks identifiability → stores model → posts report.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        retrieval_service=None,
        llm_logger: Optional[LLMLogger] = None,
    ):
        self._registry   = registry
        self._id_analyst = IDAnalyst()
        self._model      = model
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._retrieval  = retrieval_service
        self._llm_logger = llm_logger

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        # Build task description from dossier
        contract_id = dossier.assets.plant_contract_id or ""
        contract_info = self._load_contract_info(contract_id)
        task_msg = (
            f"Plant contract ID: {contract_id}\n"
            f"Contract info: {json.dumps(contract_info, indent=2)}\n\n"
            f"Derive the white-box ODE model for this plant."
        )
        if dossier.open_critiques:
            critique = dossier.open_critiques[0]
            task_msg += f"\n\nCritique from validation: {critique.description}"

        report = self.run(task_msg)

        meta = report.metadata
        model_id = meta.get("model_id", "")
        improvable = meta.get("improvable", True)

        # Mark critique as addressed if one was open
        critiques = dossier.open_critiques
        if critiques:
            critiques = [c.model_copy(update={"status": "addressed"})
                         if c.addressed_to == "modeler" else c
                         for c in critiques]

        return dossier.update(
            status=f"modeler done: model={model_id}",
            artifacts=dossier.artifacts.model_copy(update={
                "current_model_id": model_id,
                "model_history":    dossier.artifacts.model_history + ([model_id] if model_id else []),
            }),
            open_critiques=critiques,
            last_report=report.model_copy(update={
                "metadata": {**report.metadata, "improvable": improvable},
            }),
        )

    # ── LLM run ───────────────────────────────────────────────────────────────

    def run(self, task_msg: str) -> Report:
        """Derive ODE, check identifiability, store model, return Report."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else \
            "You are the Modeler. Derive the ODE and call store_model, then post_report."

        memory_tool = []
        if self._retrieval is not None:
            memory_tool = [{
                "name": "query_memory",
                "description": (
                    "Search past identification runs and reference documents for similar plants. "
                    "Call this FIRST, before deriving the ODE, to find proven structures and "
                    "parameter ranges from prior runs. Returns prior ODEs, fitted parameter values, "
                    "bounds that worked, and which path (white/grey/surrogate) succeeded."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Describe the plant dynamics you're looking for, e.g. 'second-order rotary pendulum with friction and external torque'",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 3)",
                        },
                    },
                    "required": ["query"],
                },
            }]

        tools = memory_tool + [
            {
                "name": "check_identifiability",
                "description": (
                    "Check whether the ODE parameters are structurally identifiable "
                    "from the available output measurements. Returns identifiability "
                    "status and suggested reparameterization."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "normalized_rhs": {"type": "string",
                                           "description": "RHS of highest-deriv ODE"},
                        "params":        {"type": "array", "items": {"type": "string"},
                                          "description": "Parameter names in the ODE"},
                        "state_vars":    {"type": "array", "items": {"type": "string"}},
                        "input_vars":    {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["normalized_rhs", "params", "state_vars", "input_vars"],
                },
            },
            {
                "name": "store_model",
                "description": "Store the model structure in the registry. Returns model_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "description":        {"type": "string"},
                        "normalized_rhs":     {"type": "string",
                                               "description": "Reparameterized RHS (highest derivative = RHS)"},
                        "fit_params":         {"type": "array", "items": {"type": "string"},
                                               "description": "Lumped params to fit"},
                        "param_bounds":       {"type": "object",
                                               "description": "{param: [lo, hi]}"},
                        "state_vars":         {"type": "array", "items": {"type": "string"},
                                               "description": "State variables in ascending derivative order. len = system order."},
                        "input_vars":         {"type": "array", "items": {"type": "string"}},
                        "output_vars":        {"type": "array", "items": {"type": "string"}},
                        "system_order":       {"type": "integer",
                                               "description": "ODE order (1=first-order, 2=second-order, …). Defaults to len(state_vars)."},
                        "output_state_index": {"type": "integer",
                                               "description": "Index in state_vars of the measured output (default 0). Set to 1 if only velocity is observed."},
                        "improvable":         {"type": "boolean"},
                    },
                    "required": ["description", "normalized_rhs", "fit_params",
                                 "state_vars", "input_vars", "output_vars"],
                },
            },
            _post_report_schema(),
        ]

        messages = [{"role": "user", "content": task_msg}]
        model_id: Optional[str] = None

        _sep = "─" * 72
        log.debug("[ModelerAgent] %s", _sep)
        log.debug("[ModelerAgent] SYSTEM PROMPT\n%s\n%s", system_prompt, _sep)
        log.debug("[ModelerAgent] TASK MESSAGE\n%s\n%s", task_msg, _sep)

        for iteration in range(10):
            log.debug("[ModelerAgent] iteration %d — calling LLM (%s)", iteration + 1, self._model)
            resp = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            if self._llm_logger is not None:
                self._llm_logger.log(
                    agent="ModelerAgent",
                    iteration=iteration + 1,
                    system=system_prompt,
                    messages=messages,
                    model=self._model,
                    tools=tools,
                    response=resp,
                )

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    log.debug("[ModelerAgent] LLM text:\n%s", block.text)

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                log.warning("[ModelerAgent] LLM returned no tool calls — stopping loop")
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            final_report: Optional[Report] = None

            for tc in tool_uses:
                log.debug("[ModelerAgent] tool call → %s\n  input: %s",
                          tc.name, json.dumps(tc.input, indent=2))
                if tc.name == "query_memory":
                    query  = tc.input.get("query", "")
                    top_k  = int(tc.input.get("top_k", 3))
                    result_content = self._retrieval.query(query, top_k=top_k)
                    log.debug("[ModelerAgent] query_memory → %d chars returned", len(result_content))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_content,
                    })

                elif tc.name == "check_identifiability":
                    inp = tc.input
                    id_report = self._id_analyst.analyze(
                        normalized_rhs=inp["normalized_rhs"],
                        fit_params=inp["params"],
                        state_vars=inp["state_vars"],
                        input_vars=inp["input_vars"],
                    )
                    result_content = json.dumps({
                        "identifiable": id_report.identifiable.value,
                        "non_identifiable_params": id_report.non_identifiable_params,
                        "recommendation": id_report.recommendation,
                    })
                    log.debug("[ModelerAgent] tool result ← check_identifiability:\n  %s",
                              result_content)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_content,
                    })

                elif tc.name == "store_model":
                    inp = tc.input
                    state_vars   = inp["state_vars"]
                    system_order = inp.get("system_order", len(state_vars))
                    artifact = ModelArtifact(
                        model_type=ModelType.WHITE_BOX,
                        structure_description=inp["description"],
                        parameters={},
                        metadata={
                            "normalized_rhs":     inp["normalized_rhs"],
                            "fit_params":         inp["fit_params"],
                            "param_bounds":       inp.get("param_bounds", {}),
                            "state_vars":         state_vars,
                            "input_vars":         inp["input_vars"],
                            "output_vars":        inp["output_vars"],
                            "system_order":       system_order,
                            "output_state_index": inp.get("output_state_index", 0),
                            "improvable":         inp.get("improvable", True),
                        },
                    )
                    model_id = self._registry.store_model(artifact)
                    log.debug("[ModelerAgent] tool result ← store_model: model_id=%s rhs=%s params=%s",
                              model_id, inp["normalized_rhs"], inp["fit_params"])
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps({"model_id": model_id}),
                    })

                elif tc.name == "post_report":
                    final_report = _build_report("ModelerAgent", tc.input)
                    if model_id and "model_id" not in final_report.metadata:
                        final_report.metadata["model_id"] = model_id
                    log.debug("[ModelerAgent] post_report → status=%s summary=%s",
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
            agent="ModelerAgent",
            status=AgentStatus.FAILED,
            summary="Modeler did not complete within iteration limit.",
            metadata={"model_id": model_id or ""},
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_contract_info(self, contract_id: str) -> dict:
        """Load plant contract metadata from the registry."""
        if not contract_id:
            return {}
        try:
            artifact = self._registry.load_model(contract_id)
            return artifact.metadata.get("plant_contract", {})
        except Exception:
            return {}


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
                "metadata": {"type": "object",
                             "description": "Must include model_id and improvable."},
            },
            "required": ["status", "summary"],
        },
    }


def _build_report(agent: str, raw: dict) -> Report:
    from agents.base_agent import _build_report as base_build
    return base_build(agent, raw)
