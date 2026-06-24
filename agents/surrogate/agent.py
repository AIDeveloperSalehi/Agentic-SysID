"""
Surrogate LLM Agent.

Replaces the deterministic SurrogateSubOrchestrator as the graph node for
black-box surrogate fitting.  An LLM reasons over the full attempt log —
including what grey-box strategies were tried and why they failed — to choose
the right model class, epochs, and data collection strategy.

Conditioning: the attempt_log written by GreyBoxAgent and ValidationAgent gives
this agent context about the physics that was understood but couldn't be captured,
the failure modes, and which frequency / amplitude regimes were problematic.
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
    Dossier,
    ModelArtifact,
    ModelType,
    Report,
    Rung,
    SplitFlag,
)
from agents.estimator import _estimate_hidden_states
from agents.surrogate.trainer import SurrogateTrainer
from agents.surrogate.active_sampler import ActiveSampler
from agents.surrogate.model_class_selector import ModelClass, ModelClassSelector, Paradigm
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "surrogate.md"
MAX_ITERATIONS = 8


class SurrogateAgent:
    """
    LLM-driven surrogate identification agent.

    The LLM chooses model class, epochs, and data collection strategy based on
    the attempt log and validation feedback.  The SurrogateTrainer handles the
    actual numerical fitting.
    """

    def __init__(
        self,
        plant_api:  PlantAPI,
        registry:   ModelRegistry,
        db:         ExperimentDatabase,
        model:      str = "claude-sonnet-4-6",
        api_key:    Optional[str] = None,
        n_samples:  int = 300,
    ):
        self._model    = model
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._api      = plant_api
        self._registry = registry
        self._db       = db
        self._trainer  = SurrogateTrainer()
        self._sampler  = ActiveSampler(plant_api, db)
        self._selector = ModelClassSelector()
        self._n        = n_samples

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id    = dossier.artifacts.current_model_id or ""
        contract_id = dossier.assets.plant_contract_id or ""
        dataset_ids = list(dossier.artifacts.dataset_ids)

        # Load model meta to pass physics context
        state_vars, input_vars, output_vars, system_order, output_state_index = (
            self._load_model_meta(model_id)
        )

        task_msg      = _build_task_message(dossier, dataset_ids)
        system_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else _FALLBACK_SYSTEM

        result = self._run_llm_loop(
            task_msg, system_prompt, contract_id, dataset_ids,
            model_id, state_vars, input_vars, output_vars,
            system_order, output_state_index,
        )

        final_model_id = result.get("model_id", model_id)
        reasoning      = result.get("reasoning", "")
        run_ids        = result.get("run_ids", [])
        train_rmse     = result.get("train_rmse", float("nan"))

        report = Report(
            agent="SurrogateAgent",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=final_model_id, type="model", store="registry")],
            summary=(
                f"SurrogateAgent: model={final_model_id}, train_rmse={train_rmse:.4f}. {reasoning}"
            ),
            metadata={
                "model_id":   final_model_id,
                "train_rmse": train_rmse,
                "reasoning":  reasoning,
                "run_ids":    run_ids,
            },
        )

        new_dataset_ids = list(dict.fromkeys(dataset_ids + run_ids))
        return dossier.update(
            current_rung=Rung.BLACK,
            status=f"surrogate_agent done: model={final_model_id}",
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
        contract_id:  str,
        dataset_ids:  List[str],
        model_id:     str,
        state_vars:   List[str],
        input_vars:   List[str],
        output_vars:  List[str],
        system_order: int,
        output_state_index: int,
    ) -> dict:
        import anthropic
        client   = anthropic.Anthropic(api_key=self._api_key)
        messages = [{"role": "user", "content": task_msg}]
        tools    = _surrogate_tools()

        accumulated_run_ids: List[str] = []
        best_result: dict = {"model_id": model_id, "train_rmse": float("inf"), "reasoning": ""}
        eval_count = 0

        _sep = "─" * 72
        logger.debug("[SurrogateAgent] %s\nSYSTEM:\n%s\n%s\nTASK:\n%s\n%s",
                     _sep, system_prompt, _sep, task_msg, _sep)

        for iteration in range(MAX_ITERATIONS):
            logger.debug("[SurrogateAgent] iteration %d", iteration + 1)
            resp = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    logger.info("[SurrogateAgent] LLM: %s", block.text.strip())

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                logger.warning("[SurrogateAgent] no tool calls — stopping")
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            done    = False

            for tc in tool_uses:
                logger.info("[SurrogateAgent] tool → %s  args=%s", tc.name,
                            json.dumps(tc.input, indent=2))

                out = self._dispatch(
                    tc.name, tc.input,
                    contract_id, dataset_ids, accumulated_run_ids, eval_count,
                    model_id, state_vars, input_vars, output_vars,
                    system_order, output_state_index,
                )

                if tc.name == "collect_data" and "run_id" in out:
                    dataset_ids = dataset_ids + [out["run_id"]]
                    accumulated_run_ids.append(out["run_id"])

                if tc.name == "train_model" and "model_id" in out and not out.get("error"):
                    tr = out.get("train_rmse", float("inf"))
                    if tr < best_result.get("train_rmse", float("inf")):
                        best_result.update(out)

                if tc.name == "evaluate_model":
                    eval_count += 1

                if tc.name == "post_result":
                    best_result["model_id"]  = tc.input.get("model_id", best_result.get("model_id", model_id))
                    best_result["reasoning"] = tc.input.get("reasoning", "")
                    best_result["run_ids"]   = accumulated_run_ids
                    done = True

                out_str = json.dumps(out)
                logger.info("[SurrogateAgent] result ← %s", out_str[:300])
                results.append({"type": "tool_result", "tool_use_id": tc.id, "content": out_str})

            messages.append({"role": "user", "content": results})
            if done:
                break

        best_result.setdefault("run_ids", accumulated_run_ids)
        return best_result

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _dispatch(
        self,
        name:          str,
        args:          dict,
        contract_id:   str,
        dataset_ids:   List[str],
        run_ids_acc:   List[str],
        eval_count:    int,
        model_id:      str,
        state_vars:    List[str],
        input_vars:    List[str],
        output_vars:   List[str],
        system_order:  int,
        output_state_index: int,
    ) -> dict:
        try:
            if name == "collect_data":
                return self._collect(
                    contract_id,
                    int(args.get("n_samples", 300)),
                    args.get("strategy", "prbs"),
                    seed=200 + len(run_ids_acc) * 7,
                )

            elif name == "train_model":
                mc_str  = args.get("model_class", "rnn")
                epochs  = int(args.get("n_epochs", 200))
                hidden  = int(args.get("hidden_size", 64))
                return self._train(
                    mc_str, epochs, dataset_ids, contract_id,
                    model_id, state_vars, input_vars, output_vars,
                    system_order, output_state_index,
                    hidden_size=hidden,
                )

            elif name == "evaluate_model":
                return self._evaluate(args.get("model_id", model_id), contract_id)

            elif name == "post_result":
                return {"status": "posted"}

            else:
                return {"error": f"unknown tool: {name}"}
        except Exception as exc:
            logger.error("[SurrogateAgent] tool %s raised: %s", name, exc)
            return {"error": str(exc)}

    # ── Numerical helpers ─────────────────────────────────────────────────────

    def _collect(self, contract_id: str, n_samples: int, strategy: str, seed: int) -> dict:
        from agents.experiment_design import ExperimentDesignAgent
        designer = ExperimentDesignAgent()
        contract = self._load_contract(contract_id)
        method   = strategy if strategy in ("prbs", "chirp", "multisine") else "prbs"
        try:
            seq    = designer.design_for_identification(contract, n_samples=n_samples, method=method, seed=seed)
            u_func = designer.make_u_func(seq["t"], seq["u"])
            result = self._api.apply_input(
                u_func=u_func,
                t_span=(float(seq["t"][0]), float(seq["t"][-1])),
                dt=float(seq["t"][1] - seq["t"][0]),
                purpose="identification",
                input_type=seq["input_type"],
                agent="surrogate_agent",
                split_flag=SplitFlag.TRAIN,
            )
            return {"run_id": result["run_id"], "n_samples": n_samples, "strategy": method}
        except Exception as exc:
            return {"error": str(exc)}

    def _train(
        self,
        model_class_str: str,
        n_epochs:        int,
        dataset_ids:     List[str],
        contract_id:     str,
        model_id:        str,
        state_vars:      List[str],
        input_vars:      List[str],
        output_vars:     List[str],
        system_order:    int,
        output_state_index: int,
        hidden_size:     int = 64,
    ) -> dict:
        # Load all training data
        t_all, u_all, y_all = self._load_training_data(dataset_ids, contract_id)
        if t_all is None:
            return {"error": "no training data available"}

        # Trim edges
        valid        = np.ones(len(t_all), dtype=bool)
        valid[:3]    = False
        valid[-3:]   = False
        t_v = t_all[valid];  u_v = u_all[valid];  y_v = y_all[valid]
        if len(t_v) < 20:
            return {"error": "too few samples after trimming"}

        n_train = len(t_v)
        logger.info("[SurrogateAgent] training %s, n=%d, epochs=%d", model_class_str, n_train, n_epochs)

        # Dispatch to trainer
        mc_map = {mc.value: mc for mc in ModelClass}
        mc = mc_map.get(model_class_str, ModelClass.RNN)
        try:
            if mc == ModelClass.NARX:
                result = self._trainer.fit_narx(y_v, u_v)
            elif mc == ModelClass.RNN:
                result = self._trainer.fit_rnn(y_v, u_v, n_epochs=n_epochs,
                                               hidden_size=hidden_size)
            elif mc == ModelClass.TRANSFORMER:
                result = self._trainer.fit_transformer(y_v, u_v, n_epochs=n_epochs)
            else:
                return {"error": f"unsupported model class: {model_class_str}"}
        except Exception as exc:
            return {"error": f"training failed: {exc}"}

        # Store artifact
        obj_id = f"surrogate_{mc.value}_{n_epochs}ep"
        self._registry.store_object(obj_id, result.predictor)

        artifact = ModelArtifact(
            model_type=ModelType.BLACK_BOX,
            structure_description=f"Surrogate ({mc.value}, IO paradigm, {n_epochs} epochs)",
            parameters={},
            parent_id=model_id or None,
            metadata={
                "normalized_rhs":     "SURROGATE",
                "model_class":        mc.value,
                "surrogate_paradigm": "input_output",
                "state_vars":         state_vars,
                "input_vars":         input_vars,
                "output_vars":        output_vars,
                "system_order":       system_order,
                "output_state_index": output_state_index,
                "fit_params":         [],
                "n_train":            n_train,
                "n_epochs":           n_epochs,
                "train_rmse":         result.train_rmse,
                "surrogate_object_id": obj_id,
            },
        )
        # Update the object key to use the actual artifact ID
        actual_obj_id = artifact.id + "_obj"
        self._registry.store_object(actual_obj_id, result.predictor)
        artifact = artifact.model_copy(update={
            "metadata": {**artifact.metadata, "surrogate_object_id": actual_obj_id}
        })
        surrogate_id = self._registry.store_model(artifact)

        train_rmse = result.train_rmse
        return {
            "model_id":   surrogate_id,
            "model_class": mc.value,
            "n_train":    n_train,
            "n_epochs":   n_epochs,
            "train_rmse": float(train_rmse) if np.isfinite(train_rmse) else 9999.0,
        }

    def _evaluate(self, model_id: str, contract_id: str) -> dict:
        from agents.validation import ValidationAgent
        validator = ValidationAgent(self._api, self._registry, self._db)
        try:
            verdict, report = validator.run(model_id, contract_id)
            return {
                "verdict":      verdict.verdict.value,
                "gap_type":     verdict.gap_type.value,
                "worst_rmse":   round(verdict.metrics.get("rmse", float("nan")), 4),
                "scenario_rmse": {
                    sc: round(float(v), 4)
                    for sc, v in zip(
                        ["slow_sinusoidal", "large_amplitude", "chirp_sweep"],
                        [verdict.metrics.get(f"rmse_{i}", float("nan")) for i in range(3)],
                    )
                },
                "whiteness_p":   round(verdict.metrics.get("residual_whiteness_p", 0.0), 4),
                "max_feat_corr": round(verdict.metrics.get("max_feature_correlation", 0.0), 4),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _load_training_data(self, dataset_ids, contract_id):
        t_parts, u_parts, y_parts = [], [], []
        for run_id in dataset_ids:
            try:
                t_r, u_r, y_r = self._db.load_arrays(run_id)
                t_parts.append(t_r)
                u_parts.append(u_r[0])
                y_parts.append(y_r[0])
            except Exception as exc:
                logger.warning("[SurrogateAgent] skipping run_id=%s — %s", run_id, exc)
        if not t_parts:
            return None, None, None
        return (
            np.concatenate(t_parts),
            np.concatenate(u_parts),
            np.concatenate(y_parts),
        )

    def _load_model_meta(self, model_id):
        defaults = (["x", "x_dot"], ["u"], ["x"], 2, 0)
        if not model_id:
            return defaults
        try:
            meta = self._registry.load_model(model_id).metadata
            sv   = meta.get("state_vars",  defaults[0])
            iv   = meta.get("input_vars",  defaults[1])
            ov   = meta.get("output_vars", defaults[2])
            so   = meta.get("system_order", len(sv))
            osi  = meta.get("output_state_index", 0)
            return sv, iv, ov, so, osi
        except Exception:
            return defaults

    def _load_contract(self, contract_id: str):
        from core.schemas import PlantContract
        if contract_id:
            try:
                artifact = self._registry.load_model(contract_id)
                pc_data  = artifact.metadata.get("plant_contract", {})
                if pc_data:
                    pc_data["input_limits"] = {
                        k: tuple(v) for k, v in pc_data.get("input_limits", {}).items()
                    }
                    return PlantContract(**pc_data)
            except Exception:
                pass
        return self._api._contract


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_task_message(dossier: Dossier, dataset_ids: List[str]) -> str:
    # Estimate total training samples
    lines = ["## Current state"]
    lines.append(f"- Current rung: {dossier.current_rung.value}")
    lines.append(f"- Physics availability: {dossier.assets.physics.value}")
    lines.append(f"- Accumulated dataset IDs: {dataset_ids}  (total IDs: {len(dataset_ids)})")
    if dossier.artifacts.best_val_rmse is not None:
        lines.append(
            f"- Best val RMSE seen so far: {dossier.artifacts.best_val_rmse:.4f} "
            f"(model={dossier.artifacts.best_model_id}) — your surrogate should beat this"
        )

    if dossier.last_verdict:
        v = dossier.last_verdict
        lines.append(f"\n## Last validation verdict")
        lines.append(f"- Result: {v.verdict.value} / gap={v.gap_type.value}")
        lines.append(f"- Worst RMSE: {v.metrics.get('rmse', float('nan')):.4f}")

    if dossier.attempt_log:
        lines.append("\n## Full attempt history (read carefully — includes grey-box reasoning)")
        lines.append("| # | rung | agent | model_class | n_train | train_rmse | val_rmse | gap | reasoning |")
        lines.append("|---|------|-------|-------------|---------|-----------|----------|-----|-----------|")
        for i, a in enumerate(dossier.attempt_log, 1):
            tr   = f"{a.train_rmse:.4f}" if a.train_rmse == a.train_rmse else "n/a"
            vr   = f"{a.val_rmse:.4f}"   if a.val_rmse is not None else "n/a"
            note = (a.agent_reasoning[:80] + "…") if len(a.agent_reasoning) > 80 else a.agent_reasoning
            lines.append(f"| {i} | {a.rung} | {a.agent} | {a.model_class} | {a.n_train} | {tr} | {vr} | {a.gap_type} | {note} |")

    lines.append("\n## Your task")
    lines.append(
        "Choose a surrogate model class and training configuration. "
        "Use the attempt history to understand what physics was captured and what remains unexplained. "
        "Start by training a model, then evaluate it. Adjust based on the result. "
        "Post the best model you find."
    )
    return "\n".join(lines)


def _surrogate_tools() -> list:
    return [
        {
            "name": "collect_data",
            "description": "Run a new identification experiment on the plant.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "n_samples": {"type": "integer"},
                    "strategy":  {"type": "string", "enum": ["prbs", "chirp", "multisine"]},
                },
                "required": ["n_samples"],
            },
        },
        {
            "name": "train_model",
            "description": "Train a black-box surrogate model on all available training data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_class": {"type": "string", "enum": ["narx", "rnn", "transformer"],
                                   "description": "Model class to train"},
                    "n_epochs":    {"type": "integer",
                                   "description": "Training epochs (ignored for NARX)"},
                    "hidden_size": {"type": "integer",
                                   "description": (
                                       "LSTM hidden units for rnn (default 64). "
                                       "Increase to 128 or 256 when train RMSE is still high "
                                       "after many epochs — do NOT jump straight to transformer."
                                   )},
                },
                "required": ["model_class", "n_epochs"],
            },
        },
        {
            "name": "evaluate_model",
            "description": "Run full adversarial validation on the real plant. EXPENSIVE — call at most 3 times.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                },
                "required": ["model_id"],
            },
        },
        {
            "name": "post_result",
            "description": "Finalise and post the best surrogate model. Call ONCE when done.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_id":  {"type": "string"},
                    "reasoning": {"type": "string", "description": "Why this model and configuration"},
                },
                "required": ["model_id", "reasoning"],
            },
        },
    ]


_FALLBACK_SYSTEM = (
    "You are the Surrogate Identification Agent. Choose model class, collect data if needed, "
    "train, evaluate, and post the best surrogate model. Read the attempt history to make "
    "informed choices."
)
