"""
Grey-Box LLM Agent.

Replaces the deterministic GreyBoxSubOrchestrator as the graph node for grey-box
correction.  An LLM reasons over the attempt log, residual diagnosis, and
validation results to decide which strategy to apply, whether to collect more
data, and when the best model has been found.

Conditioning: every agent's reasoning (stored in AttemptEntry.agent_reasoning)
is visible here, so this agent knows what the estimator tried, why the white-box
failed, and what the validation said.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    AttemptEntry,
    Dossier,
    Report,
    Rung,
)
from agents.greybox.sub_orchestrator import GreyBoxSubOrchestrator
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "greybox.md"
MAX_ITERATIONS = 12


class GreyBoxAgent:
    """
    LLM-driven grey-box identification agent.

    The LLM decides which correction strategy to apply, whether to collect
    more data, and when to stop.  Numerical work is delegated to
    GreyBoxSubOrchestrator helper methods.
    """

    def __init__(
        self,
        plant_api:  PlantAPI,
        registry:   ModelRegistry,
        db:         ExperimentDatabase,
        model:      str = "claude-sonnet-4-6",
        api_key:    Optional[str] = None,
        n_samples:  int = 600,
    ):
        self._model    = model
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._toolkit  = GreyBoxSubOrchestrator(plant_api, registry, db, n_samples=n_samples)
        self._registry = registry
        self._db       = db
        self._api      = plant_api

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id    = dossier.artifacts.current_model_id or ""
        contract_id = dossier.assets.plant_contract_id or ""
        dataset_ids = list(dossier.artifacts.dataset_ids)

        task_msg    = _build_task_message(dossier, model_id, dataset_ids)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else _FALLBACK_SYSTEM

        result = self._run_llm_loop(task_msg, system_prompt, model_id, contract_id, dataset_ids, dossier)

        final_model_id = result.get("model_id", model_id)
        reasoning      = result.get("reasoning", "")
        run_ids        = result.get("run_ids", [])
        train_rmse     = result.get("train_rmse", float("nan"))
        rung_str       = result.get("rung", "grey")
        rung           = Rung(rung_str) if rung_str in (r.value for r in Rung) else Rung.GREY

        report = Report(
            agent="GreyBoxAgent",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=final_model_id, type="model", store="registry")],
            summary=f"GreyBoxAgent: selected model={final_model_id}, train_rmse={train_rmse:.4f}. {reasoning}",
            metadata={
                "model_id":   final_model_id,
                "train_rmse": train_rmse,
                "reasoning":  reasoning,
                "run_ids":    run_ids,
            },
        )

        new_dataset_ids = list(dict.fromkeys(dataset_ids + run_ids))
        return dossier.update(
            current_rung=rung,
            status=f"greybox_agent done: model={final_model_id}",
            artifacts=dossier.artifacts.model_copy(update={
                "current_model_id": final_model_id,
                "model_history":    dossier.artifacts.model_history + [final_model_id],
                "dataset_ids":      new_dataset_ids,
            }),
            last_report=report,
        )

    # ── LLM loop ─────────────────────────────────────────────────────────────

    def _run_llm_loop(
        self,
        task_msg:     str,
        system_prompt: str,
        model_id:     str,
        contract_id:  str,
        dataset_ids:  List[str],
        dossier:      Dossier,
    ) -> dict:
        import anthropic
        client   = anthropic.Anthropic(api_key=self._api_key)
        messages = [{"role": "user", "content": task_msg}]
        tools    = _greybox_tools()

        accumulated_run_ids: List[str] = []
        best_result: dict = {"model_id": model_id, "train_rmse": float("inf"), "rung": "grey", "reasoning": ""}
        eval_count = 0

        _sep = "─" * 72
        logger.debug("[GreyBoxAgent] %s\nSYSTEM:\n%s\n%s\nTASK:\n%s\n%s",
                     _sep, system_prompt, _sep, task_msg, _sep)

        for iteration in range(MAX_ITERATIONS):
            logger.debug("[GreyBoxAgent] iteration %d", iteration + 1)
            resp = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    logger.info("[GreyBoxAgent] LLM: %s", block.text.strip())

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                logger.warning("[GreyBoxAgent] no tool calls — stopping")
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            done    = False

            for tc in tool_uses:
                logger.info("[GreyBoxAgent] tool → %s  args=%s", tc.name,
                            json.dumps(tc.input, indent=2))
                out = self._dispatch(
                    tc.name, tc.input, model_id, contract_id,
                    dataset_ids, accumulated_run_ids, eval_count,
                )
                if tc.name == "collect_data" and "run_id" in out:
                    dataset_ids = dataset_ids + [out["run_id"]]
                    accumulated_run_ids.append(out["run_id"])
                if tc.name in ("run_coulomb_extension", "run_sindy_correction",
                               "run_gp_correction", "run_sequence_correction",
                               "run_re_estimation"):
                    if "model_id" in out and not out.get("error"):
                        if out.get("train_rmse", float("inf")) < best_result.get("train_rmse", float("inf")):
                            best_result.update(out)
                if tc.name == "evaluate_model" and "worst_rmse" in out:
                    eval_count += 1
                if tc.name == "post_result":
                    best_result["model_id"]  = tc.input.get("model_id", best_result.get("model_id", model_id))
                    best_result["reasoning"] = tc.input.get("reasoning", "")
                    best_result["run_ids"]   = accumulated_run_ids
                    done = True

                out_str = json.dumps(out)
                logger.info("[GreyBoxAgent] result ← %s", out_str[:300])
                results.append({"type": "tool_result", "tool_use_id": tc.id, "content": out_str})

            messages.append({"role": "user", "content": results})
            if done:
                break

        best_result.setdefault("run_ids", accumulated_run_ids)
        return best_result

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _dispatch(
        self,
        name:        str,
        args:        dict,
        model_id:    str,
        contract_id: str,
        dataset_ids: List[str],
        run_ids_acc: List[str],
        eval_count:  int,
    ) -> dict:
        try:
            if name == "get_residual_diagnosis":
                return self._toolkit.diagnose(model_id, dataset_ids, contract_id)

            elif name == "collect_data":
                n         = int(args.get("n_samples", 600))
                st        = args.get("strategy", "prbs")
                seed      = 200 + len(run_ids_acc) * 7
                amplitude = args.get("amplitude", None)
                f_lo      = args.get("f_lo", None)
                f_hi      = args.get("f_hi", None)
                return self._toolkit.collect_data(
                    contract_id, n_samples=n, strategy=st, seed=seed,
                    amplitude=amplitude, f_lo=f_lo, f_hi=f_hi,
                )

            elif name == "run_coulomb_extension":
                return self._toolkit.run_with_strategy(
                    "coulomb", model_id, dataset_ids, contract_id)

            elif name == "run_sindy_correction":
                fitting_domain = args.get("fitting_domain", "output")
                return self._toolkit.run_with_strategy(
                    "sindy", model_id, dataset_ids, contract_id,
                    fitting_domain=fitting_domain)

            elif name == "run_gp_correction":
                return self._toolkit.run_with_strategy(
                    "gp", model_id, dataset_ids, contract_id)

            elif name == "run_sequence_correction":
                mc          = args.get("model_class", "rnn")
                epochs      = int(args.get("n_epochs", 200))
                seq_len     = int(args.get("seq_len", 50))
                hidden_size = int(args.get("hidden_size", 64))
                return self._toolkit.run_with_strategy(
                    "sequence", model_id, dataset_ids, contract_id,
                    seq_model_class=mc, seq_epochs=epochs, seq_len=seq_len,
                    seq_hidden_size=hidden_size)

            elif name == "run_re_estimation":
                return self._toolkit.run_re_estimation(
                    new_rhs=args.get("new_rhs", ""),
                    new_params=args.get("new_params", []),
                    physics_model_id=model_id,
                    dataset_ids=dataset_ids,
                    contract_id=contract_id,
                    param_bounds=args.get("param_bounds", None),
                )

            elif name == "evaluate_model":
                return self._evaluate(args.get("model_id", model_id), contract_id)

            elif name == "post_result":
                return {"status": "posted"}

            else:
                return {"error": f"unknown tool: {name}"}
        except Exception as exc:
            logger.error("[GreyBoxAgent] tool %s raised: %s", name, exc)
            return {"error": str(exc)}

    def _evaluate(self, model_id: str, contract_id: str) -> dict:
        """Run deterministic validation and return a concise summary.

        Uses deterministic=True to avoid spawning a nested LLM probe loop —
        the greybox agent just needs a fast quality signal, not a full adversarial
        analysis.  The LLM probe loop runs only in the main pipeline verdict.
        """
        from agents.validation import ValidationAgent
        validator = ValidationAgent(self._api, self._registry, self._db)
        try:
            verdict, report = validator.run(model_id, contract_id, deterministic=True)
            scenario_rmse = {
                k: v for k, v in report.metadata.get("metrics", {}).items()
                if k.startswith("rmse") or k == "rmse"
            }
            return {
                "verdict":      verdict.verdict.value,
                "gap_type":     verdict.gap_type.value,
                "worst_rmse":   verdict.metrics.get("rmse", float("nan")),
                "scenario_rmse": {
                    sc: round(float(v), 4)
                    for sc, v in zip(
                        ["slow_sinusoidal", "large_amplitude", "chirp_sweep"],
                        [verdict.metrics.get(f"rmse_{i}", float("nan")) for i in range(3)],
                    )
                },
                "whiteness_p":  round(verdict.metrics.get("residual_whiteness_p", 0), 4),
                "max_feat_corr": round(verdict.metrics.get("max_feature_correlation", 0), 4),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_task_message(dossier: Dossier, model_id: str, dataset_ids: List[str]) -> str:
    lines = ["## Current dossier state"]
    lines.append(f"- Current rung: {dossier.current_rung.value}")
    lines.append(f"- Physics: {dossier.assets.physics.value}")
    lines.append(f"- Training dataset IDs: {dataset_ids}")
    lines.append(f"- Current model ID: {model_id}")
    if dossier.artifacts.best_val_rmse is not None:
        lines.append(f"- Best val RMSE seen so far: {dossier.artifacts.best_val_rmse:.4f} (model={dossier.artifacts.best_model_id})")

    if dossier.last_verdict:
        v = dossier.last_verdict
        lines.append(f"\n## Last validation verdict")
        lines.append(f"- Result: {v.verdict.value} / gap={v.gap_type.value}")
        lines.append(f"- Worst RMSE: {v.metrics.get('rmse', '?'):.4f}")
        lines.append(f"- Residual whiteness p: {v.metrics.get('residual_whiteness_p', '?'):.4f}")
        lines.append(f"- Max feature corr: {v.metrics.get('max_feature_correlation', '?'):.4f}")
        lines.append(f"- Probes run: {v.metrics.get('n_probes', v.metrics.get('n_scenarios', '?'))}")
        if v.failure_hypothesis:
            lines.append(f"\n## Validation agent failure diagnosis")
            lines.append(v.failure_hypothesis)
        if v.worst_case_inputs:
            wc = v.worst_case_inputs
            lines.append(f"\n## Worst-case probe")
            lines.append(f"- Scenario: {wc.get('scenario_type', '?')}")
            lines.append(f"- Amplitude fraction: {wc.get('amplitude_fraction', '?')}")

    if dossier.attempt_log:
        lines.append("\n## Full attempt history (all prior modelling attempts)")
        lines.append("| # | rung | agent | model_class | n_train | train_rmse | val_rmse | gap | reasoning |")
        lines.append("|---|------|-------|-------------|---------|-----------|----------|-----|-----------|")
        for i, a in enumerate(dossier.attempt_log, 1):
            tr   = f"{a.train_rmse:.4f}" if not (a.train_rmse != a.train_rmse) else "n/a"
            vr   = f"{a.val_rmse:.4f}"   if a.val_rmse is not None else "n/a"
            note = (a.agent_reasoning[:80] + "…") if len(a.agent_reasoning) > 80 else a.agent_reasoning
            lines.append(f"| {i} | {a.rung} | {a.agent} | {a.model_class} | {a.n_train} | {tr} | {vr} | {a.gap_type} | {note} |")

    lines.append("\n## Your task")
    lines.append(
        "Start by calling `get_residual_diagnosis` to understand the current residual structure. "
        "Then choose a correction strategy based on the diagnosis AND the attempt history. "
        "Call `evaluate_model` after training to see actual validation RMSE. "
        "Post the best model you find."
    )
    return "\n".join(lines)


def _greybox_tools() -> list:
    return [
        {
            "name": "get_residual_diagnosis",
            "description": "Compute acceleration-domain residuals and return feature correlations and recommended strategy.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "collect_data",
            "description": (
                "Run a new identification experiment on the plant to gather more training data. "
                "Choose strategy and parameters based on the residual diagnosis:\n"
                "  - Broadband gap or first experiment → prbs or compound\n"
                "  - Frequency-domain error at specific band → chirp or multisine with f_lo/f_hi\n"
                "  - Low-frequency nonlinearity (Coulomb, stiction) → multisine with small f_hi\n"
                "  - Static gain / steady-state → steps\n"
                "Amplitude in plant input units; f_lo/f_hi in Hz for chirp/multisine."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of samples to collect (typical: 400–800)",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["prbs", "chirp", "multisine", "steps", "compound"],
                        "description": (
                            "Signal type:\n"
                            "  prbs     — broadband random binary; best default for initial data\n"
                            "  chirp    — linear frequency sweep from f_lo to f_hi; "
                            "use when you need to characterise a specific frequency range\n"
                            "  multisine— sum of sinusoids at chosen frequencies; "
                            "use for targeted excitation of known resonance/problem bands\n"
                            "  steps    — staircase between fixed levels; "
                            "use to characterise static gain, dead-zones, or saturation\n"
                            "  compound — PRBS + multisine combined; "
                            "best broadband coverage in a single experiment"
                        ),
                    },
                    "amplitude": {
                        "type": "number",
                        "description": (
                            "Peak amplitude in plant input units (e.g. N·m for torque). "
                            "Must be within the contract input limits. "
                            "Ignored for compound (uses fixed fractions). "
                            "Omit to use 70% of the limit (default)."
                        ),
                    },
                    "f_lo": {
                        "type": "number",
                        "description": (
                            "Lower frequency bound in Hz. Used by chirp and multisine. "
                            "Omit to use 0.1 Hz default."
                        ),
                    },
                    "f_hi": {
                        "type": "number",
                        "description": (
                            "Upper frequency bound in Hz. Used by chirp and multisine. "
                            "Must be below Nyquist (= 0.5 / sample_time). "
                            "Omit to use 90% of Nyquist default."
                        ),
                    },
                },
                "required": ["n_samples"],
            },
        },
        {
            "name": "run_coulomb_extension",
            "description": "Extend the ODE with a Coulomb friction term K_c·tanh(vel/ε) and re-estimate all parameters. Returns a WHITE-BOX model.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "run_sindy_correction",
            "description": (
                "Identify a sparse symbolic correction term from the feature library (LASSO). "
                "Returns a GREY-BOX model with y_pred = y_physics + symbolic_correction. "
                "Default fitting_domain='output' fits LASSO directly against (y_meas − y_phys), "
                "which is O(σ) noise — strongly preferred over 'acceleration' which is O(σ/Δt²). "
                "Use 'acceleration' only if you have a specific reason (e.g. the measurement is "
                "already a rate and output-domain residuals are meaningless)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "fitting_domain": {
                        "type": "string",
                        "enum": ["output", "acceleration"],
                        "description": (
                            "Domain in which LASSO is fitted. "
                            "'output': fit against y_meas − y_phys (recommended, O(σ) noise). "
                            "'acceleration': fit against ẍ_meas − ẍ_phys (O(σ/Δt²) noise — noisy)."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "run_gp_correction",
            "description": "Fit a GP on (states, u) → acceleration residual as a non-parametric correction. Returns a GREY-BOX model.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "run_sequence_correction",
            "description": "Train a sequence model (RNN or NARX) on the output residual (y_measured − y_physics). Combines physics baseline with data-driven correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_class": {"type": "string", "enum": ["rnn", "narx"],
                                   "description": "Sequence model class to use"},
                    "n_epochs":    {"type": "integer", "description": "Training epochs"},
                    "seq_len":     {
                        "type": "integer",
                        "description": (
                            "RNN only. Number of time-steps in each training window (default 50). "
                            "Increase (e.g. 100–200) when the dominant dynamics are slow relative "
                            "to the sample rate and the model needs longer context to track them. "
                            "Decrease (e.g. 20–30) when data is limited or training is unstable."
                        ),
                    },
                    "hidden_size": {
                        "type": "integer",
                        "description": (
                            "RNN only. Number of LSTM hidden units (default 64). "
                            "Increase to 128 or 256 when train RMSE is still high after many "
                            "epochs — indicates underfitting due to insufficient model capacity. "
                            "Only increase if n_epochs and seq_len are already well-tuned."
                        ),
                    },
                },
                "required": ["model_class", "n_epochs"],
            },
        },
        {
            "name": "run_re_estimation",
            "description": (
                "Re-parameterize the white-box ODE with a new RHS structure and re-fit ALL "
                "parameters from scratch via NLS. Use ONLY when all additive corrections "
                "(SINDy, GP, Coulomb, sequence) have failed and the failure diagnosis points "
                "to a structural ODE defect — e.g. a missing normalization constant (1/J), "
                "wrong functional form, or a missing parameter that scales the whole model. "
                "Do NOT use for minor coefficient mis-estimation. "
                "Call evaluate_model on the result before posting."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "new_rhs": {
                        "type": "string",
                        "description": (
                            "New ODE RHS using the same state/input variable names as the "
                            "current model. Example: "
                            "'(K_u*torque - K_d*theta_dot - K_s*sin(theta)) / J' "
                            "to add moment of inertia J."
                        ),
                    },
                    "new_params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All parameter names to estimate — include existing ones AND any "
                            "new ones. Example: ['K_u', 'K_d', 'K_s', 'K_c', 'J']"
                        ),
                    },
                    "param_bounds": {
                        "type": "object",
                        "description": (
                            "Bounds for NEW parameters only: {name: [lower, upper]}. "
                            "Existing parameter bounds are inherited automatically. "
                            "Example: {'J': [0.001, 50.0]}"
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Why the current ODE structure is defective and why this new "
                            "structure addresses it."
                        ),
                    },
                },
                "required": ["new_rhs", "new_params", "reasoning"],
            },
        },
        {
            "name": "evaluate_model",
            "description": "Run full adversarial validation on the plant and return per-scenario RMSE. Call at most 3 times — it runs real plant experiments.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model ID to evaluate"},
                },
                "required": ["model_id"],
            },
        },
        {
            "name": "post_result",
            "description": "Finalise and post the best model. Call ONCE when done.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_id":  {"type": "string", "description": "Best model ID"},
                    "reasoning": {"type": "string", "description": "Why this model was chosen"},
                },
                "required": ["model_id", "reasoning"],
            },
        },
    ]


_FALLBACK_SYSTEM = (
    "You are the Grey-Box Identification Agent. Diagnose residuals, apply corrections, "
    "evaluate results, and post the best model. Use the attempt history to avoid repeating "
    "failed strategies."
)
