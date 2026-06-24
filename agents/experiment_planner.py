"""
ExperimentPlannerAgent — LLM agent that decides *what experiments to run* before
each estimator invocation.

Sits between the router (or modeler) and the estimator.  Receives the full
dossier history (prior experiment amplitudes, methods, training costs, validation
failures) and emits an ExperimentPlan that the estimator will follow instead of
its hard-coded rule-based defaults.

This separates concerns cleanly:
  - ExperimentPlannerAgent: WHAT to excite (amplitude, methods, segment length)
  - EstimatorAgent:         HOW to fit (NLS, multi-shooting, state refinement)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from core.schemas import Dossier, ExperimentPlan, Report, AgentStatus

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiment_planner.md"


class ExperimentPlannerAgent:
    """
    LLM-backed experiment planning agent.

    Called once before each estimator invocation.  Makes a single LLM call,
    reads the plan_experiment tool response, stores the result as
    dossier.experiment_plan, and returns the updated dossier.
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
        plan, reasoning = self._plan(dossier)
        logger.info(
            "[ExperimentPlannerAgent] plan: methods=%s amp=%.2f–%.2f seg_len=%d | %s",
            plan.methods, plan.base_amplitude, plan.max_amplitude, plan.seg_len, reasoning,
        )

        report = Report(
            agent="ExperimentPlannerAgent",
            status=AgentStatus.DONE,
            summary=f"Experiment plan: {plan.methods} amp {plan.base_amplitude:.2f}–{plan.max_amplitude:.2f} seg_len {plan.seg_len}",
            metadata={"experiment_plan": plan.model_dump()},
        )
        return dossier.update(
            status="experiment_planner done",
            experiment_plan=plan,
            last_report=report,
        )

    # ── Planning logic ────────────────────────────────────────────────────────

    def _plan(self, dossier: Dossier) -> tuple[ExperimentPlan, str]:
        """Returns (ExperimentPlan, reasoning)."""
        task_msg      = _build_task_message(dossier)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else _FALLBACK_SYSTEM

        logger.debug("[ExperimentPlannerAgent] task:\n%s", task_msg)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            resp   = client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": task_msg}],
                tools=[_plan_experiment_schema()],
                tool_choice={"type": "any"},
            )

            for block in resp.content:
                if block.type == "tool_use" and block.name == "plan_experiment":
                    inp  = block.input
                    sched = inp.get("amplitude_schedule") or None
                    plan = ExperimentPlan(
                        methods             = inp.get("methods", ["prbs", "multisine", "steps"]),
                        base_amplitude      = float(inp.get("base_amplitude", 0.70)),
                        max_amplitude       = float(inp.get("max_amplitude",  0.92)),
                        seg_len             = int(inp.get("seg_len", 50)),
                        amplitude_schedule  = [float(a) for a in sched] if sched else None,
                        reasoning           = inp.get("reasoning", ""),
                    )
                    return plan, plan.reasoning

            logger.warning("[ExperimentPlannerAgent] LLM returned no tool call — using rule-based fallback")
        except Exception as exc:
            logger.error("[ExperimentPlannerAgent] LLM call failed (%s) — using rule-based fallback", exc)

        return _rule_based_fallback(dossier)


# ── Message builder ───────────────────────────────────────────────────────────

def _build_task_message(dossier: Dossier) -> str:
    re_est = dossier.re_estimate_count
    lines  = ["## Pipeline state"]
    lines.append(f"- Re-estimation number: {re_est} (0 = first pass, 1 = first retry, ...)")
    lines.append(f"- Budget remaining: {dossier.budget.remaining:.1f} / {dossier.budget.total:.1f}")

    # Last validation result
    if dossier.last_verdict:
        v = dossier.last_verdict
        m = v.metrics
        lines.append("\n## Last validation result")
        lines.append(f"- Verdict: {v.verdict.value}")
        lines.append(f"- Gap type: {v.gap_type.value}")
        lines.append(f"- Worst RMSE: {m.get('rmse', '?')}")
        lines.append(f"- Worst scenario: {m.get('worst_case_scenario', '?')} "
                     f"at amplitude {m.get('worst_case_amplitude_fraction', '?')}")
        amp_dep = m.get("amplitude_dependent_failure")
        best_tier = m.get("best_amplitude_tier", 0.0)
        if amp_dep is not None:
            lines.append(f"- Amplitude-dependent failure: {amp_dep}  "
                         f"(True = passes at low amp, fails at high amp only)")
        if best_tier and best_tier > 0:
            lines.append(f"- Highest passing amplitude: {best_tier:.2f}  "
                         f"(model is valid up to this amplitude)")
        if v.failure_hypothesis:
            lines.append(f"- Failure hypothesis: {v.failure_hypothesis[:400]}")

    # Full history of what was tried
    if dossier.attempt_log:
        lines.append("\n## History of estimation attempts")
        lines.append("Each entry shows: re_est index, methods used (if known), amplitude range, "
                     "training cost, validation RMSE, and what the estimator observed.")
        for i, a in enumerate(dossier.attempt_log):
            if a.agent != "estimator":
                continue
            tr  = f"{a.train_rmse:.4f}" if a.train_rmse == a.train_rmse else "n/a"
            vr  = f"{a.val_rmse:.4f}"   if a.val_rmse is not None else "n/a"
            meta_str = ""
            if a.agent_reasoning:
                meta_str = f" | {a.agent_reasoning[:120]}"
            lines.append(f"  Attempt {i+1}: train_rmse={tr} val_rmse={vr}{meta_str}")

    # Current experiment_plan (if any — don't repeat it)
    if dossier.experiment_plan:
        p = dossier.experiment_plan
        lines.append(f"\n## Previous experiment plan (what was just tried)")
        lines.append(f"- Methods: {p.methods}")
        lines.append(f"- Amplitude: {p.base_amplitude:.2f} → {p.max_amplitude:.2f}")
        lines.append(f"- Seg len: {p.seg_len}")
        lines.append(f"- Reasoning: {p.reasoning}")

    lines.append("\n## Your task")
    lines.append(
        "Design the next identification experiment. Call `plan_experiment` with:\n"
        "- `methods`: ordered list of excitation methods (one per inner iteration)\n"
        "- `base_amplitude`: lowest amplitude fraction in your plan (used as reference)\n"
        "- `max_amplitude`: highest amplitude fraction in your plan\n"
        "- `amplitude_schedule`: REQUIRED — explicit list of amplitudes, one per inner iteration.\n"
        "  This OVERRIDES the base→max ramp. Must span at least LOW (<0.50), MEDIUM (0.50–0.73), "
        "and HIGH (>0.73) tiers. Weight more entries toward the weak regime but never drop a tier.\n"
        "- `seg_len`: multi-shooting segment length in samples (50 = short, 200 = long)\n"
        "- `reasoning`: one sentence explaining why this plan differs from what was tried before\n\n"
        "CRITICAL RULE: When amplitude_dependent_failure=True (model passes at low amplitude, "
        "fails at high), the fix is NOT more high-amplitude training data. "
        "Restoring-force and spring-constant terms average to zero over full spinning cycles and "
        "CANNOT be identified from high-amplitude spinning data. "
        "Add more LOW and MEDIUM amplitude iterations to improve these coefficients.\n\n"
        "Available methods: `prbs`, `multisine`, `steps`, `chirp`."
    )
    return "\n".join(lines)


# ── Tool schema ───────────────────────────────────────────────────────────────

def _plan_experiment_schema() -> dict:
    return {
        "name": "plan_experiment",
        "description": "Specify the next identification experiment for the estimator to run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["prbs", "multisine", "steps", "chirp"]},
                    "description": "Ordered list of excitation methods for each inner loop iteration",
                    "minItems": 1,
                    "maxItems": 6,
                },
                "base_amplitude": {
                    "type": "number",
                    "description": "Starting amplitude fraction (0.0–1.0)",
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "max_amplitude": {
                    "type": "number",
                    "description": "Ceiling amplitude fraction (0.0–1.0); inner loop ramps toward this",
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "seg_len": {
                    "type": "integer",
                    "description": "Multi-shooting segment length in samples (50–200)",
                    "minimum": 20,
                    "maximum": 250,
                },
                "amplitude_schedule": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                    "description": (
                        "Explicit per-iteration amplitude fractions (one per inner loop iteration). "
                        "MUST include at least one value ≤0.50 (low tier, for restoring-force "
                        "identifiability), one in 0.50–0.73 (medium tier), and one ≥ max_amplitude "
                        "(high tier, for input-gain identifiability). "
                        "Weight more entries toward the failing regime without dropping any tier."
                    ),
                    "minItems": 3,
                    "maxItems": 6,
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence: why these choices and how they differ from prior attempts",
                },
            },
            "required": ["methods", "base_amplitude", "max_amplitude", "seg_len",
                         "amplitude_schedule", "reasoning"],
        },
    }


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_fallback(dossier: Dossier) -> tuple[ExperimentPlan, str]:
    """Stratified fallback that always covers low, medium, and high amplitude tiers."""
    re_est = dossier.re_estimate_count
    if re_est == 0:
        plan = ExperimentPlan(
            methods             = ["prbs", "steps", "prbs", "steps", "prbs"],
            base_amplitude      = 0.35,
            max_amplitude       = 0.75,
            seg_len             = 50,
            amplitude_schedule  = [0.30, 0.55, 0.35, 0.65, 0.75],
            reasoning           = "First estimation: stratified low→medium coverage for broad parameter identifiability.",
        )
    else:
        amp_dep   = False
        best_tier = 0.0
        if dossier.last_verdict:
            amp_dep   = bool(dossier.last_verdict.metrics.get("amplitude_dependent_failure", False))
            best_tier = float(dossier.last_verdict.metrics.get("best_amplitude_tier", 0.0))

        if amp_dep and best_tier > 0:
            # Passes at low amplitude → weight more medium iterations to improve K_s,
            # keep high for K_u but don't go all-in on the spinning regime.
            sched = [0.30, best_tier, 0.40, min(best_tier + 0.10, 0.72), 0.85]
            reason = (
                f"Amplitude-dependent failure: model valid up to {best_tier:.2f}. "
                "Adding more low/medium data to improve restoring-force coefficient "
                "rather than increasing amplitude further."
            )
        else:
            # All amplitudes fail: cover the full range with low-tier emphasis.
            sched  = [0.30, 0.50, 0.35, 0.65, 0.85]
            reason = f"Re-estimation {re_est}: all amplitudes failing — broad stratified coverage."

        plan = ExperimentPlan(
            methods             = ["prbs", "multisine", "steps", "prbs", "multisine"],
            base_amplitude      = min(sched),
            max_amplitude       = max(sched),
            seg_len             = min(50 + 50 * re_est, 120),
            amplitude_schedule  = sched,
            reasoning           = reason,
        )
    return plan, plan.reasoning


_FALLBACK_SYSTEM = (
    "You are the Experiment Planner Agent. Design the next identification experiment "
    "by calling plan_experiment. Choose methods and amplitude to cover the regime where "
    "validation failed, avoiding repeating what was already tried."
)
