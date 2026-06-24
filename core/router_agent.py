"""
LLM Router Agent (Option B — fully agentic routing).

Replaces the deterministic _route_after_validation function.
After each validation the RouterAgent reads the full attempt log, the dossier
state, and the last verdict, then decides where to route next.

The RouterAgent is conditioned on every prior agent's reasoning: it can see
what the Estimator found, what strategies GreyBoxAgent tried and why, and
whether the SurrogateAgent's results were better or worse than grey-box.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from core.schemas import Dossier, Report, AgentStatus

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router.md"

VALID_NODES     = {"estimator", "greybox_so", "surrogate_so", "ship"}
MAX_RE_ESTIMATE = 3   # max times router may send back to Estimator


class RouterAgent:
    """
    LLM-backed routing agent.

    Called after every validation.  Makes a single LLM call, reads the route_to
    tool call, stores the decision and reasoning in last_report.recommended_next,
    and returns the updated dossier.

    The static routing function _route_after_router in orchestrator.py then
    reads dossier.last_report.recommended_next to resolve the LangGraph edge.
    """

    def __init__(
        self,
        model:   str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        self._model   = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        decision, reasoning = self._decide(dossier)

        logger.info("[RouterAgent] → %s | %s", decision, reasoning)

        report = Report(
            agent="RouterAgent",
            status=AgentStatus.DONE,
            summary=f"Router → {decision}: {reasoning}",
            recommended_next=decision,
            metadata={
                "next_node": decision,
                "reasoning": reasoning,
            },
        )
        return dossier.update(
            status=f"router → {decision}",
            last_report=report,
        )

    # ── Decision logic ────────────────────────────────────────────────────────

    def _decide(self, dossier: Dossier) -> tuple[str, str]:
        """Returns (next_node, reasoning)."""
        # Fast path: budget exhausted → ship
        if dossier.budget.exhausted:
            return "ship", "Budget exhausted — shipping best model seen so far."

        # Fast path: PASS verdict → ship
        if dossier.last_verdict and dossier.last_verdict.verdict.value == "pass":
            return "ship", "Validation passed — shipping current model."

        # Black rung after surrogate → ship (cannot escalate further)
        from core.schemas import Rung
        if dossier.current_rung == Rung.BLACK:
            return "ship", (
                "On black rung with failed surrogate — cannot escalate further. "
                "Shipping best model from attempt log."
            )

        # Ask the LLM for a reasoned routing decision
        task_msg      = _build_task_message(dossier)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else _FALLBACK_SYSTEM

        logger.debug("[RouterAgent] task:\n%s", task_msg)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            resp   = client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": task_msg}],
                tools=[_route_to_schema()],
                tool_choice={"type": "any"},
            )

            for block in resp.content:
                if block.type == "tool_use" and block.name == "route_to":
                    node      = block.input.get("next_node", "ship")
                    reasoning = block.input.get("reasoning", "")
                    if node not in VALID_NODES:
                        logger.warning("[RouterAgent] invalid node '%s' — defaulting to ship", node)
                        node = "ship"
                    # Enforce re-estimate quota
                    if node == "estimator" and dossier.re_estimate_count >= MAX_RE_ESTIMATE:
                        logger.warning("[RouterAgent] re-estimate quota exhausted — routing to greybox_so")
                        node = "greybox_so"
                        reasoning = f"Re-estimate quota exhausted ({MAX_RE_ESTIMATE}). " + reasoning
                    return node, reasoning

            # No tool call — fall back
            logger.warning("[RouterAgent] LLM returned no tool call — falling back to rule-based")
        except Exception as exc:
            logger.error("[RouterAgent] LLM call failed (%s) — falling back to rule-based", exc)

        return _rule_based_fallback(dossier)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_task_message(dossier: Dossier) -> str:
    from core.schemas import Rung
    lines = ["## Pipeline state"]
    lines.append(f"- Current rung: {dossier.current_rung.value}")
    lines.append(f"- Budget remaining: {dossier.budget.remaining:.1f} / {dossier.budget.total:.1f}")
    lines.append(f"- Re-estimate count: {dossier.re_estimate_count} / {MAX_RE_ESTIMATE}")
    if dossier.artifacts.best_val_rmse is not None:
        lines.append(
            f"- Best val RMSE in history: {dossier.artifacts.best_val_rmse:.4f} "
            f"(model={dossier.artifacts.best_model_id})"
        )

    if dossier.last_verdict:
        v = dossier.last_verdict
        lines.append(f"\n## Last validation result")
        lines.append(f"- Verdict: {v.verdict.value}")
        lines.append(f"- Gap type: {v.gap_type.value}")
        lines.append(f"- Worst RMSE: {v.metrics.get('rmse', '?'):.4f}")
        lines.append(f"- Worst scenario: {v.metrics.get('worst_case_scenario', '?')} "
                     f"at amplitude {v.metrics.get('worst_case_amplitude_fraction', '?')}")
        lines.append(f"- Whiteness p: {v.metrics.get('residual_whiteness_p', '?'):.4f}")
        lines.append(f"- Max feature corr: {v.metrics.get('max_feature_correlation', '?'):.4f}")
        if v.failure_hypothesis:
            lines.append(f"- Failure hypothesis: {v.failure_hypothesis[:300]}")

    # Dominant residual features live in last_report.metadata (posted by ValidationAgent)
    if dossier.last_report and dossier.last_report.metadata.get("dominant_features"):
        lines.append(f"- Dominant residual features: {dossier.last_report.metadata['dominant_features']}")

    if dossier.attempt_log:
        lines.append("\n## Complete attempt history")
        lines.append("| # | rung | agent | model_class | n_train | train_rmse | val_rmse | gap | reasoning |")
        lines.append("|---|------|-------|-------------|---------|-----------|----------|-----|-----------|")
        for i, a in enumerate(dossier.attempt_log, 1):
            tr   = f"{a.train_rmse:.4f}" if a.train_rmse == a.train_rmse else "n/a"
            vr   = f"{a.val_rmse:.4f}"   if a.val_rmse is not None else "n/a"
            note = (a.agent_reasoning[:100] + "…") if len(a.agent_reasoning) > 100 else a.agent_reasoning
            lines.append(f"| {i} | {a.rung} | {a.agent} | {a.model_class} | {a.n_train} | {tr} | {vr} | {a.gap_type} | {note} |")

    re_est_available = dossier.re_estimate_count < MAX_RE_ESTIMATE
    lines.append("\n## Available next steps")
    if re_est_available and dossier.current_rung.value == "white":
        lines.append(
            "- `estimator`: re-run the Estimator with higher amplitude and longer segments. "
            "Use this when the model structure is correct but parameters are inaccurate "
            "(e.g. validation fails at large amplitudes but passes at small ones, "
            "or max_feature_correlation is low-to-moderate without a dominant structural feature). "
            f"Only available {MAX_RE_ESTIMATE - dossier.re_estimate_count} more time(s)."
        )
    lines.append("- `greybox_so`: run GreyBoxAgent (physics-based correction)")
    lines.append("- `surrogate_so`: run SurrogateAgent (black-box data-driven)")
    lines.append("- `ship`: ship the best model found so far")
    lines.append("\nCall `route_to` with your decision and reasoning.")
    return "\n".join(lines)


def _route_to_schema() -> dict:
    return {
        "name": "route_to",
        "description": "Decide where the pipeline should route next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "next_node": {
                    "type": "string",
                    "enum": ["estimator", "greybox_so", "surrogate_so", "ship"],
                    "description": "The next pipeline node to execute",
                },
                "reasoning": {
                    "type": "string",
                    "description": "2-3 sentences explaining the routing decision, referencing specific RMSE numbers",
                },
            },
            "required": ["next_node", "reasoning"],
        },
    }


def _rule_based_fallback(dossier: Dossier) -> tuple[str, str]:
    """Deterministic fallback if the LLM call fails."""
    from core.schemas import GapType, Rung, VerdictResult

    if not dossier.last_verdict:
        return "ship", "No verdict available — shipping defensively."

    verdict = dossier.last_verdict
    gap     = verdict.gap_type
    rung    = dossier.current_rung
    history = dossier.artifacts.model_history

    if verdict.verdict == VerdictResult.PASS:
        return "ship", "Validation passed."
    if dossier.budget.exhausted:
        return "ship", "Budget exhausted."
    if rung == Rung.BLACK:
        return "ship", "Black rung failure — shipping best model."
    if rung == Rung.WHITE:
        from core.schemas import PhysicsAvailability
        if gap == GapType.FIXABLE and dossier.re_estimate_count < MAX_RE_ESTIMATE:
            return "estimator", (
                f"Fixable gap (parameter inaccuracy) on white rung — "
                f"re-estimating with better data "
                f"({dossier.re_estimate_count + 1}/{MAX_RE_ESTIMATE})."
            )
        if gap == GapType.STRUCTURED_RESIDUAL:
            return "greybox_so", "Structured residual on white rung → grey-box."
        if gap == GapType.UNMODELABLE:
            if dossier.assets.physics in (PhysicsAvailability.FULL, PhysicsAvailability.PARTIAL):
                return "greybox_so", "Unmodelable but physics available → grey-box."
            return "surrogate_so", "Unmodelable, no physics → surrogate."
        return "greybox_so", "Fixable gap quota exhausted or unhandled gap → grey-box."
    if rung == Rung.GREY:
        if gap == GapType.STRUCTURED_RESIDUAL:
            if len(history) < 4:
                return "greybox_so", "Grey rung first attempt → retry grey-box with more data."
            return "surrogate_so", "Grey rung already retried → surrogate."
        return "surrogate_so", "Unmodelable/fixable on grey rung → surrogate."
    return "ship", "Unhandled case — shipping best model."


_FALLBACK_SYSTEM = (
    "You are the Routing Agent. Read the attempt log and current state, then call "
    "route_to with one of: greybox_so, surrogate_so, ship. Prefer the simplest "
    "model that has the best validation RMSE. Do not escalate if escalation will "
    "make things worse."
)
