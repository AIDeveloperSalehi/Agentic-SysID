"""
Surrogate sub-orchestrator.

Called when the model fails even after grey-box correction
(current_rung is GREY or WHITE but UNMODELABLE / STRUCTURED_RESIDUAL persists).

Paradigm selection
  Paradigm.ODE (physics available: FULL or PARTIAL)
    Learns  f: (state, u) → highest_derivative  and integrates via ODE solver.
    Model class chosen by sample count: GP (≤500) or NN (>500).

  Paradigm.INPUT_OUTPUT (no physics: NONE)
    Learns directly from observed (y, u) sequences.
    Model class chosen by sample count: NARX (≤400), RNN (≤2000), Transformer (>2000).

No LLM calls are made here.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    Dossier,
    ModelArtifact,
    ModelType,
    PhysicsAvailability,
    Report,
    Rung,
)
from agents.estimator import _estimate_hidden_states, _sg_deriv
from agents.surrogate.active_sampler import ActiveSampler
from agents.surrogate.model_class_selector import ModelClass, ModelClassSelector, Paradigm
from agents.surrogate.trainer import SurrogateTrainer
from agents.surrogate.uncertainty_estimator import UncertaintyEstimator
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI

logger = logging.getLogger(__name__)

# IO-paradigm model classes (sequence predictors)
_IO_CLASSES = {ModelClass.NARX, ModelClass.RNN, ModelClass.TRANSFORMER}


class SurrogateSubOrchestrator:
    """
    Deterministic surrogate sub-orchestrator.

    Interface: __call__(dossier) → Dossier
    """

    def __init__(
        self,
        plant_api:  PlantAPI,
        registry:   ModelRegistry,
        db:         ExperimentDatabase,
        n_samples:  int = 300,
        nn_epochs:  int = 200,
    ):
        self._api      = plant_api
        self._registry = registry
        self._db       = db
        self._n        = n_samples
        self._nn_ep    = nn_epochs
        self._sampler  = ActiveSampler(plant_api, db)
        self._selector = ModelClassSelector()
        self._trainer  = SurrogateTrainer()
        self._unc_est  = UncertaintyEstimator()

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id    = dossier.artifacts.current_model_id or ""
        contract_id = dossier.assets.plant_contract_id or ""

        # ── Step 1: load meta from the failed model ───────────────────────────
        state_vars, input_vars, output_vars, system_order, output_state_index = (
            self._load_model_meta(model_id)
        )

        # ── Step 2: load all accumulated training data ────────────────────────
        t_all, u_all, y_all = self._load_training_data(
            dossier.artifacts.dataset_ids, contract_id
        )
        if t_all is None:
            return _fail(dossier, "SurrogateSO: no training data available")

        # ── Step 3: active sampling — always collect new data on retry ──────────
        # Detect retry: current model is already a surrogate (BLACK_BOX).  On
        # retry, the existing data produced a bad model; collect a fresh batch
        # regardless of how many samples are already available.
        try:
            is_retry = bool(
                model_id
                and self._registry.load_model(model_id).model_type == ModelType.BLACK_BOX
            )
        except Exception:
            is_retry = False

        contract = self._load_contract(contract_id)
        new_ids  = self._sampler.maybe_collect(
            n_available=len(t_all),
            contract=contract,
            existing_run_ids=list(dossier.artifacts.dataset_ids),
            force_collect=is_retry,
        )
        if new_ids:
            t2, u2, y2 = self._load_training_data(new_ids, contract_id)
            if t2 is not None:
                t_all = np.concatenate([t_all, t2])
                u_all = np.concatenate([u_all, u2])
                y_all = np.concatenate([y_all, y2])

        # ── Step 4: paradigm — always IO to avoid noisy state-estimate targets ──
        # ODE paradigm requires double-differentiating noisy θ to get θ̈ as a
        # training target, which amplifies measurement noise.  Since the surrogate
        # is only reached after grey-box has failed (meaning state estimates are
        # already unreliable), we always use the IO paradigm (NARX/RNN/Transformer)
        # which trains directly on the observed (y, u) sequence.
        physics_available = (
            dossier.assets.physics in (PhysicsAvailability.FULL, PhysicsAvailability.PARTIAL)
        )
        paradigm = Paradigm.INPUT_OUTPUT

        # ── Step 5: build training arrays ────────────────────────────────────
        if paradigm == Paradigm.ODE:
            # Need state estimates and derivatives for ODE-paradigm fit
            states_est  = _estimate_hidden_states(t_all, y_all, system_order)
            vel_row     = min(1, system_order - 1)
            y_dot       = states_est[vel_row]
            y_ddot      = _sg_deriv(y_dot, t_all, deriv=1)

            valid        = np.ones(len(t_all), dtype=bool)
            valid[:3]    = False
            valid[-3:]   = False
            t_v    = t_all[valid]
            u_v    = u_all[valid]
            y_v    = y_all[valid]
            ydot_v = y_dot[valid]
            yddot_v = y_ddot[valid]

            if len(t_v) < 10:
                return _fail(dossier, "SurrogateSO: too few valid samples after filtering")

            n_train = len(t_v)
        else:
            # IO paradigm: just need the raw (y, u) sequence; minimal trimming
            valid        = np.ones(len(t_all), dtype=bool)
            valid[:3]    = False
            valid[-3:]   = False
            t_v    = t_all[valid]
            u_v    = u_all[valid]
            y_v    = y_all[valid]

            if len(t_v) < 20:
                return _fail(dossier, "SurrogateSO: too few samples for IO paradigm")

            n_train = len(t_v)
            y_dot   = _sg_deriv(y_v, t_v, deriv=1)   # for uncertainty estimator compat
            ydot_v  = y_dot
            yddot_v = _sg_deriv(y_v, t_v, deriv=2)

        # ── Step 6: select model class ────────────────────────────────────────
        sel = self._selector.select(n_train, paradigm)
        logger.info("SurrogateSO: paradigm=%s model=%s (%s)",
                    paradigm.value, sel.model_class.value, sel.rationale)

        # ── Step 7: fit surrogate ─────────────────────────────────────────────
        try:
            if paradigm == Paradigm.ODE:
                result = self._trainer.fit(
                    model_class=sel.model_class,
                    theta=y_v, theta_dot=ydot_v,
                    u=u_v, theta_ddot=yddot_v,
                    n_epochs=self._nn_ep,
                )
            elif sel.model_class == ModelClass.NARX:
                result = self._trainer.fit_narx(y_v, u_v)
            elif sel.model_class == ModelClass.RNN:
                result = self._trainer.fit_rnn(y_v, u_v, n_epochs=self._nn_ep)
            else:  # TRANSFORMER
                result = self._trainer.fit_transformer(y_v, u_v, n_epochs=self._nn_ep)
        except Exception as exc:
            logger.error("SurrogateSO: trainer raised %s: %s", type(exc).__name__, exc)
            return _fail(dossier, f"SurrogateSO: trainer failed — {exc}")

        # ── Step 8: uncertainty (ODE path only; IO path: heuristic) ──────────
        if paradigm == Paradigm.ODE:
            unc = self._unc_est.estimate(
                model_class=sel.model_class,
                predictor=result.predictor,
                theta=y_v, theta_dot=ydot_v,
                u=u_v, theta_ddot=yddot_v,
            )
        else:
            unc = {"mean_std": 0.1, "note": "IO paradigm — uncertainty heuristic"}

        # ── Step 9: store model artifact + pickled predictor ──────────────────
        surrogate_artifact = ModelArtifact(
            model_type=ModelType.BLACK_BOX,
            structure_description=(
                f"Surrogate ({sel.model_class.value}, {paradigm.value} paradigm): "
                + ("data-driven ODE surrogate" if paradigm == Paradigm.ODE
                   else "input-output sequence model")
            ),
            parameters={},
            parent_id=model_id or None,
            metadata={
                "normalized_rhs":      "SURROGATE",
                "model_class":         sel.model_class.value,
                "surrogate_paradigm":  paradigm.value,
                "state_vars":          state_vars,
                "input_vars":          input_vars,
                "output_vars":         output_vars,
                "system_order":        system_order,
                "output_state_index":  output_state_index,
                "fit_params":          [],
                "n_train":             result.n_train,
                "train_rmse":          result.train_rmse,
                "uncertainty":         unc,
                "selector_rationale":  sel.rationale,
                **(result.extra_meta or {}),
            },
        )
        surrogate_obj_id = surrogate_artifact.id + "_obj"
        self._registry.store_object(surrogate_obj_id, result.predictor)

        surrogate_artifact = surrogate_artifact.model_copy(update={
            "metadata": {
                **surrogate_artifact.metadata,
                "surrogate_object_id": surrogate_obj_id,
            }
        })
        surrogate_id = self._registry.store_model(surrogate_artifact)

        # ── Step 10: report and dossier update ────────────────────────────────
        report = Report(
            agent="SurrogateSubOrchestrator",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=surrogate_id, type="model", store="registry")],
            summary=(
                f"Surrogate SO: paradigm={paradigm.value}, "
                f"model_class={sel.model_class.value}, "
                f"n_train={result.n_train}, "
                f"train_rmse={result.train_rmse:.4f}, "
                f"mean_std={unc.get('mean_std', float('nan')):.4f}."
            ),
            metadata={
                "model_id":            surrogate_id,
                "surrogate_paradigm":  paradigm.value,
                "model_class":         sel.model_class.value,
                "surrogate_object_id": surrogate_obj_id,
                "n_train":             result.n_train,
                "train_rmse":          result.train_rmse,
                "uncertainty":         unc,
                "run_ids":             new_ids,
            },
        )

        logger.info(
            "SurrogateSO done: rung→BLACK, model=%s, train_rmse=%.4f",
            surrogate_id, result.train_rmse,
        )
        return dossier.update(
            current_rung=Rung.BLACK,
            status=(
                f"surrogate_so done: "
                f"paradigm={paradigm.value}, model_class={sel.model_class.value}, "
                f"model={surrogate_id}"
            ),
            artifacts=dossier.artifacts.model_copy(update={
                "current_model_id": surrogate_id,
                "model_history":    dossier.artifacts.model_history + [surrogate_id],
                "dataset_ids":      dossier.artifacts.dataset_ids + new_ids,
            }),
            last_report=report,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model_meta(
        self,
        model_id: str,
    ) -> Tuple[List[str], List[str], List[str], int, int]:
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

    def _load_training_data(
        self,
        dataset_ids: List[str],
        contract_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        t_parts, u_parts, y_parts = [], [], []

        for run_id in dataset_ids:
            try:
                t_r, u_r, y_r = self._db.load_arrays(run_id)
                t_parts.append(t_r)
                u_parts.append(u_r[0])
                y_parts.append(y_r[0])
            except Exception as exc:
                logger.warning("SurrogateSO: skipping run_id=%s — %s", run_id, exc)

        if not t_parts:
            try:
                contract = self._load_contract(contract_id)
                from agents.experiment_design import ExperimentDesignAgent
                from core.schemas import SplitFlag
                designer = ExperimentDesignAgent()
                seq    = designer.design_for_identification(
                    contract, n_samples=self._n, seed=99
                )
                u_func = designer.make_u_func(seq["t"], seq["u"])
                result = self._api.apply_input(
                    u_func=u_func,
                    t_span=(float(seq["t"][0]), float(seq["t"][-1])),
                    dt=float(seq["t"][1] - seq["t"][0]),
                    purpose="identification",
                    input_type=seq["input_type"],
                    agent="surrogate_so",
                    split_flag=SplitFlag.TRAIN,
                )
                t_r, u_r, y_r = self._db.load_arrays(result["run_id"])
                t_parts.append(t_r)
                u_parts.append(u_r[0])
                y_parts.append(y_r[0])
            except Exception as exc:
                logger.error("SurrogateSO: failed to collect fallback data: %s", exc)
                return None, None, None

        return (
            np.concatenate(t_parts),
            np.concatenate(u_parts),
            np.concatenate(y_parts),
        )

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


# ── Module-level helpers ──────────────────────────────────────────────────────

def _fail(dossier: Dossier, msg: str) -> Dossier:
    logger.error("SurrogateSO _fail: %s", msg)
    report = Report(
        agent="SurrogateSubOrchestrator",
        status=AgentStatus.FAILED,
        summary=msg,
    )
    # Always mark rung as BLACK so the router routes to SHIP rather than
    # looping back to surrogate_so on the next validation failure.
    return dossier.update(
        current_rung=Rung.BLACK,
        status=f"surrogate_so failed: {msg}",
        last_report=report,
    )
