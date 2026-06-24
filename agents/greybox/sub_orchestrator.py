"""
Grey-box sub-orchestrator — generalized.

Receives a failed white-box model (gap_type=STRUCTURED_RESIDUAL), diagnoses the
acceleration-domain residuals with a general feature library, and produces a
corrected model via one of three deterministic strategies:

  COULOMB_TERM   — appends K_c·tanh(vel/ε) when sign(vel) dominates residual.
                   Re-estimates base params.  Result: WHITE_BOX, Rung.WHITE.
  SINDY          — appends a sparse symbolic expression from LASSO.
                   Re-estimates base params.  Result: GREY_BOX, Rung.GREY.
  GP_CORRECTION  — learns a GP(state, input) → correction.
                   Base params unchanged.    Result: GREY_BOX, Rung.GREY.

No LLM calls are made here.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import sympy as sp

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    Artifacts,
    Dossier,
    ModelArtifact,
    ModelType,
    Report,
    Rung,
    SplitFlag,
)
from agents.estimator import EstimatorAgent, _estimate_hidden_states, _multi_shoot, _sg_deriv
from agents.greybox.strategy_selector import Diagnosis, Strategy, StrategySelector
from agents.greybox.residual_trainer import ResidualTrainer, OutputCorrectionSpec
from agents.greybox.uncertainty_estimator import UncertaintyEstimator
from agents.surrogate.trainer import SurrogateTrainer
from agents.surrogate.model_class_selector import ModelClass, ModelClassSelector, Paradigm
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI
from tools.symbolic_math import _make_locals, make_ode_simulator

logger = logging.getLogger(__name__)


class GreyBoxSubOrchestrator:
    """
    Deterministic grey-box sub-orchestrator.

    Interface: __call__(dossier) → Dossier
    """

    def __init__(
        self,
        plant_api:  PlantAPI,
        registry:   ModelRegistry,
        db:         ExperimentDatabase,
        n_samples:  int = 600,
    ):
        self._api         = plant_api
        self._registry    = registry
        self._db          = db
        self._estimator   = EstimatorAgent(plant_api, registry, db, n_samples=n_samples)
        self._selector    = StrategySelector()
        self._trainer     = ResidualTrainer()
        self._unc_est     = UncertaintyEstimator()
        self._seq_trainer = SurrogateTrainer()
        self._class_sel   = ModelClassSelector()

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id    = dossier.artifacts.current_model_id
        contract_id = dossier.assets.plant_contract_id or ""

        if not model_id:
            return _fail(dossier, "GreyboxSO: no model_id in dossier")

        # ── Step 1: load white-box model ──────────────────────────────────────
        model = self._registry.load_model(model_id)
        meta  = model.metadata
        rhs   = meta["normalized_rhs"]

        # Sentinel RHS values ("RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED")
        # are not valid SymPy expressions — they mark data-driven or corrected
        # models.  On a retry the current_model_id can point to such a model.
        # Fall back to the underlying physics baseline so we get a real ODE RHS.
        _SENTINEL_RHS = {"RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"}
        if rhs in _SENTINEL_RHS:
            physics_id = meta.get("physics_model_id", "")
            if not physics_id:
                return _fail(
                    dossier,
                    f"GreyboxSO: current model has sentinel RHS '{rhs}' "
                    "but no physics_model_id — cannot re-run grey-box",
                )
            logger.info(
                "GreyboxSO: current model has sentinel RHS '%s' → "
                "loading physics baseline %s", rhs, physics_id,
            )
            model    = self._registry.load_model(physics_id)
            meta     = model.metadata
            rhs      = meta["normalized_rhs"]
            model_id = physics_id   # re-fit against the physics baseline

        fit_params   = meta["fit_params"]
        state_vars   = meta["state_vars"]
        input_vars   = meta["input_vars"]
        param_bounds = meta.get("param_bounds", {})
        system_order       = meta.get("system_order", len(state_vars))
        output_state_index = meta.get("output_state_index", 0)
        fitted_p = np.array([model.parameters.get(p, 1.0) for p in fit_params])

        # Velocity variable name for Coulomb/poly paths (first derivative of output)
        vel_idx      = min(1, system_order - 1)
        vel_var_name = state_vars[vel_idx] if state_vars else "x_dot"

        # ── Step 2: aggregate training data ──────────────────────────────────
        # On a retry (orchestrator sends us back after a grey-rung failure) the
        # rung is already GREY.  Collect a fresh batch to augment existing data.
        is_retry = dossier.current_rung == Rung.GREY
        t_all, u_all, y_all, collected_run_ids = self._load_training_data(
            dossier.artifacts.dataset_ids, contract_id, force_collect=is_retry
        )
        if t_all is None:
            return _fail(dossier, "GreyboxSO: no training data available")

        # Smooth any discontinuous sign() terms before residual computation or
        # NLS fitting — tanh(x/ε) converges better with stiff ODE solvers.
        rhs = _smooth_sign_terms(rhs)

        # ── Step 3: estimate all states via numerical differentiation ─────────
        states_est = _estimate_hidden_states(t_all, y_all, system_order)

        # ── Step 4: compute acceleration residuals (general: all states) ──────
        eps_ddot, valid_mask = self._compute_accel_residuals(
            rhs, fit_params, state_vars, input_vars,
            fitted_p, t_all, u_all, states_est,
        )

        t_v      = t_all[valid_mask]
        states_v = states_est[:, valid_mask]
        u_v      = u_all[valid_mask]
        eps_v    = eps_ddot[valid_mask]
        vel_v    = states_v[vel_idx]   # velocity row (for uncertainty estimator)

        # ── Step 4b: output residuals for residual sequence correction path ────
        e_output, t_seq, u_seq = _compute_output_residuals(
            rhs, fit_params, state_vars, input_vars, fitted_p,
            t_all, u_all, y_all, system_order, output_state_index, states_est,
        )

        # ── Step 5: general residual diagnosis ────────────────────────────────
        diag = self._selector.diagnose_general(
            states_v, u_v, eps_v, state_vars, input_vars
        )

        logger.info(
            "GreyboxSO: strategy=%s max_corr=%.3f top=%s",
            diag.strategy.value, diag.max_correlation,
            diag.top_features[:3],
        )

        # ── Step 6: build extended model spec ────────────────────────────────
        spec = self._trainer.fit(
            strategy=diag.strategy,
            base_rhs=rhs,
            base_params=fit_params,
            base_param_bounds=param_bounds,
            base_fitted_params={p: float(v) for p, v in zip(fit_params, fitted_p)},
            t=t_v,
            theta_dot=vel_v,
            eps_ddot=eps_v,
            vel_var_name=vel_var_name,
            states=states_v,
            state_names=state_vars,
            input_names=input_vars,
            u=u_v,
        )

        # ── Step 7: determine model type and rung ─────────────────────────────
        if diag.strategy == Strategy.COULOMB_TERM:
            model_type  = ModelType.WHITE_BOX
            new_rung    = Rung.WHITE
            call_estimator = True
        elif diag.strategy == Strategy.SINDY:
            model_type  = ModelType.GREY_BOX
            new_rung    = Rung.GREY
            call_estimator = True
        else:  # GP_CORRECTION
            model_type  = ModelType.GREY_BOX
            new_rung    = Rung.GREY
            call_estimator = False   # GP fitted separately; base params already good

        # ── Step 8: store extended model structure ────────────────────────────
        is_gp_path      = diag.strategy == Strategy.GP_CORRECTION
        gp_obj_id: Optional[str] = None
        # Smooth any sign() in the SINDY correction so ODE integrators don't
        # see a discontinuous RHS (sign is fine for correlation but bad for ODE).
        extended_rhs = (
            "GP_CORRECTED" if is_gp_path
            else _smooth_sign_terms(spec.normalized_rhs)
        )

        extended_artifact = ModelArtifact(
            model_type=model_type,
            structure_description=(
                f"Extended ({diag.strategy.value}): "
                + (spec.normalized_rhs if not is_gp_path else "GP correction + base ODE")
            ),
            parameters=(
                # For GP path, copy base fitted params — no re-estimation
                {p: float(v) for p, v in zip(fit_params, fitted_p)}
                if is_gp_path else spec.p0_override
            ),
            parent_id=model_id,
            metadata={
                "normalized_rhs":     extended_rhs,
                "base_rhs":           rhs,           # always store base for GP path
                "fit_params":         spec.fit_params,
                "param_bounds":       spec.param_bounds,
                "state_vars":         state_vars,
                "input_vars":         input_vars,
                "output_vars":        meta.get("output_vars", [state_vars[0]] if state_vars else ["y"]),
                "system_order":       system_order,
                "output_state_index": output_state_index,
                "improvable":         False,
                "greybox_strategy":   diag.strategy.value,
                "correction_coeffs":  spec.correction_coeffs,
                "p0_override":        spec.p0_override,
            },
        )
        extended_id = self._registry.store_model(extended_artifact)

        # For GP path: store correction callable and link it in artifact
        if is_gp_path and spec.correction_object is not None:
            gp_obj_id = extended_id + "_gp_corr"
            self._registry.store_object(gp_obj_id, spec.correction_object)
            # Update artifact metadata with the object ID
            updated_meta = {
                **extended_artifact.metadata,
                "correction_object_id": gp_obj_id,
            }
            extended_artifact = extended_artifact.model_copy(update={"metadata": updated_meta})
            self._registry.store_model(extended_artifact)   # overwrite with updated meta
            # Note: extended_id stays the same (same artifact.id)

        # ── Step 9: re-estimate all parameters (COULOMB and SINDY paths) ──────
        fitted_id  = extended_id
        cov_id_est = ""
        new_run_ids: List[str] = []

        if call_estimator:
            est_report = self._estimator.run(extended_id, contract_id)
            fitted_id  = est_report.metadata.get("model_id", extended_id)
            cov_id_est = est_report.metadata.get("covariance_id", "")
            new_run_ids = est_report.metadata.get("run_ids", [])

            if est_report.status == AgentStatus.FAILED:
                return _fail(
                    dossier,
                    f"GreyboxSO: EstimatorAgent failed — {est_report.summary}",
                    diag.strategy.value,
                )

        # ── Step 10: symbolic path training RMSE ─────────────────────────────
        sym_model    = self._registry.load_model(fitted_id)
        sym_fitted_p = np.array([sym_model.parameters.get(p, 1.0) for p in spec.fit_params])
        sym_corr_fn  = None
        if is_gp_path:
            corr_obj_id = extended_artifact.metadata.get("correction_object_id", "")
            if corr_obj_id:
                try:
                    sym_corr_fn = self._registry.load_object(corr_obj_id)
                except Exception:
                    pass
        sym_rhs_for_sim = rhs if is_gp_path else extended_rhs
        symbolic_train_rmse = _compute_training_rmse(
            sym_rhs_for_sim, spec.fit_params, state_vars, input_vars,
            sym_fitted_p, t_all, u_all, y_all, states_est,
            system_order, output_state_index,
            correction_fn=sym_corr_fn,
        )
        logger.info("GreyboxSO: symbolic path training RMSE=%.4f", symbolic_train_rmse)

        # ══════════════════════════════════════════════════════════════════════
        # PATH B — residual sequence correction (no double differentiation)
        # ══════════════════════════════════════════════════════════════════════
        seq_model_id   = None
        seq_train_rmse = np.inf
        seq_class_name = ""

        if len(e_output) >= 20:
            seq_res = self._fit_residual_sequence(e_output, u_seq)
            if seq_res is not None:
                seq_predictor, seq_train_rmse, seq_class_name = seq_res
                logger.info(
                    "GreyboxSO: sequence path train RMSE=%.4f (class=%s)",
                    seq_train_rmse, seq_class_name,
                )
                seq_obj_id = extended_id + "_res_seq"
                self._registry.store_object(seq_obj_id, seq_predictor)

                output_vars = meta.get("output_vars", [state_vars[0]] if state_vars else ["y"])
                seq_artifact = ModelArtifact(
                    model_type=ModelType.GREY_BOX,
                    structure_description=(
                        f"Residual surrogate ({seq_class_name}): "
                        f"physics baseline + {seq_class_name} correction on output residual"
                    ),
                    parameters={},
                    parent_id=model_id,
                    metadata={
                        "normalized_rhs":       "RESIDUAL_CORRECTED",
                        "physics_model_id":     model_id,
                        "residual_object_id":   seq_obj_id,
                        "residual_model_class": seq_class_name,
                        "fit_params":           fit_params,
                        "param_bounds":         param_bounds,
                        "state_vars":           state_vars,
                        "input_vars":           input_vars,
                        "output_vars":          output_vars,
                        "system_order":         system_order,
                        "output_state_index":   output_state_index,
                        "improvable":           False,
                        "train_rmse":           seq_train_rmse,
                        "greybox_strategy":     f"residual_sequence_{seq_class_name}",
                    },
                )
                seq_model_id = self._registry.store_model(seq_artifact)

        # ── Pick winner by training RMSE ──────────────────────────────────────
        use_sequence = seq_model_id is not None and seq_train_rmse < symbolic_train_rmse

        if use_sequence:
            final_model_id = seq_model_id
            final_rung     = Rung.GREY
            strategy_used  = f"residual_sequence_{seq_class_name}"
            winning_rmse   = seq_train_rmse
            logger.info(
                "GreyboxSO: SEQUENCE wins (seq=%.4f < sym=%.4f)",
                seq_train_rmse, symbolic_train_rmse,
            )
        else:
            final_model_id = fitted_id
            final_rung     = new_rung
            strategy_used  = diag.strategy.value
            winning_rmse   = symbolic_train_rmse
            logger.info("GreyboxSO: SYMBOLIC wins (rmse=%.4f)", symbolic_train_rmse)

        # ── Step 11: uncertainty ──────────────────────────────────────────────
        nls_cov: Optional[np.ndarray] = None
        if cov_id_est:
            try:
                nls_cov = self._registry.load_covariance(cov_id_est)
            except Exception:
                pass

        # When the sequence model wins, only the base physics params matter.
        # Use the base fit_params and extract the corresponding covariance block.
        unc_fit_params = fit_params if use_sequence else spec.fit_params
        if use_sequence and nls_cov is not None:
            n_base = len(fit_params)
            if nls_cov.shape[0] >= n_base:
                nls_cov = nls_cov[:n_base, :n_base]
            else:
                nls_cov = None

        cov_matrix, ci = self._unc_est.estimate(
            strategy=diag.strategy,
            fit_params=unc_fit_params,
            nls_covariance=nls_cov,
            t=t_v,
            theta_dot=vel_v,
            eps_ddot=eps_v,
        )
        gb_cov_id = final_model_id + "_gb_cov"
        self._registry.store_covariance(gb_cov_id, cov_matrix)

        # ── Step 12: build report and update dossier ──────────────────────────
        final_model = self._registry.load_model(final_model_id)
        param_str = (
            ", ".join(
                f"{p}={final_model.parameters.get(p, float('nan')):.3f}"
                for p in spec.fit_params
            ) if final_model.parameters else "(sequence model — no explicit params)"
        )

        report = Report(
            agent="GreyBoxSubOrchestrator",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=final_model_id, type="model", store="registry")],
            summary=(
                f"Grey-box SO: selected={strategy_used}, "
                f"sym_rmse={symbolic_train_rmse:.4f}, seq_rmse={seq_train_rmse:.4f}. "
                f"Symbolic: {diag.strategy.value} max_corr={diag.max_correlation:.3f} "
                f"top={diag.top_features[:3]}. Params: {param_str}."
            ),
            metadata={
                "model_id":          final_model_id,
                "strategy":          strategy_used,
                "symbolic_strategy": diag.strategy.value,
                "max_correlation":   diag.max_correlation,
                "top_features":      diag.top_features,
                "correction_coeffs": spec.correction_coeffs,
                "covariance_id":     gb_cov_id,
                "run_ids":           new_run_ids,
                "params":            dict(final_model.parameters),
                "sym_train_rmse":    symbolic_train_rmse,
                "seq_train_rmse":    seq_train_rmse,
                "used_sequence":     use_sequence,
            },
        )

        return dossier.update(
            current_rung=final_rung,
            status=f"greybox_so done: strategy={strategy_used}, model={final_model_id}",
            artifacts=dossier.artifacts.model_copy(update={
                "current_model_id": final_model_id,
                "model_history":    dossier.artifacts.model_history + [final_model_id],
                "dataset_ids":      dossier.artifacts.dataset_ids + collected_run_ids + new_run_ids,
            }),
            last_report=report,
        )

    # ── Public tool-callable interface (used by GreyBoxAgent) ────────────────

    def diagnose(
        self,
        physics_model_id: str,
        dataset_ids: List[str],
        contract_id: str,
    ) -> dict:
        """
        Load training data, compute residuals, run StrategySelector.
        Returns a plain dict safe to format into an LLM prompt.
        """
        model = self._registry.load_model(physics_model_id)
        meta  = model.metadata
        rhs   = meta.get("normalized_rhs", "")
        _SENTINEL = {"RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"}
        if rhs in _SENTINEL:
            physics_model_id = meta.get("physics_model_id", physics_model_id)
            model = self._registry.load_model(physics_model_id)
            meta  = model.metadata
            rhs   = meta["normalized_rhs"]

        fit_params   = meta["fit_params"]
        state_vars   = meta["state_vars"]
        input_vars   = meta["input_vars"]
        system_order = meta.get("system_order", len(state_vars))
        output_state_index = meta.get("output_state_index", 0)
        fitted_p = np.array([model.parameters.get(p, 1.0) for p in fit_params])

        t, u, y, _ = self._load_training_data(dataset_ids, contract_id)
        if t is None:
            return {"error": "no training data"}

        rhs_smooth   = _smooth_sign_terms(rhs)
        states_est   = _estimate_hidden_states(t, y, system_order)
        vel_idx      = min(1, system_order - 1)
        eps_ddot, vm = self._compute_accel_residuals(
            rhs_smooth, fit_params, state_vars, input_vars, fitted_p, t, u, states_est
        )
        diag = self._selector.diagnose_general(
            states_est[:, vm], u[vm], eps_ddot[vm], state_vars, input_vars
        )
        return {
            "recommended_strategy": diag.strategy.value,
            "max_correlation":      diag.max_correlation,
            "top_features":         diag.top_features[:5],
            "feature_correlations": dict(zip(diag.top_features, diag.correlations)),
            "n_train":              int(len(t)),
        }

    def run_re_estimation(
        self,
        new_rhs:          str,
        new_params:       List[str],
        physics_model_id: str,
        dataset_ids:      List[str],
        contract_id:      str,
        param_bounds:     Optional[Dict[str, List[float]]] = None,
    ) -> dict:
        """
        Re-parameterize the white-box ODE with a new RHS structure and re-fit via NLS.

        Creates a fresh ModelArtifact with the proposed structure, runs the NLS
        estimator on it, and returns the fitted model.  Returns rung=grey (Option B)
        so that, if the greybox agent posts this model, the router does not treat it
        as a fresh white-box entry and re-route to greybox again.
        """
        # ── Resolve physics baseline ──────────────────────────────────────────
        model = self._registry.load_model(physics_model_id)
        meta  = model.metadata
        _SENTINEL = {"RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"}
        if meta.get("normalized_rhs") in _SENTINEL:
            physics_model_id = meta.get("physics_model_id", physics_model_id)
            model = self._registry.load_model(physics_model_id)
            meta  = model.metadata

        state_vars         = meta["state_vars"]
        input_vars         = meta["input_vars"]
        system_order       = meta.get("system_order", len(state_vars))
        output_state_index = meta.get("output_state_index", 0)
        output_vars        = meta.get("output_vars", [state_vars[0]] if state_vars else ["y"])

        # Inherit existing param bounds; overlay caller-specified ones for new params
        merged_bounds = {**meta.get("param_bounds", {})}
        if param_bounds:
            merged_bounds.update(param_bounds)
        for p in new_params:
            if p not in merged_bounds:
                merged_bounds[p] = [1e-4, 1e3]

        # ── Store new model artifact with proposed structure ───────────────────
        from core.schemas import ModelArtifact, ModelType
        artifact = ModelArtifact(
            model_type=ModelType.WHITE_BOX,
            structure_description=f"Re-parameterized ODE: {new_rhs}",
            parameters={p: 1.0 for p in new_params},
            parent_id=physics_model_id,
            metadata={
                "normalized_rhs":     new_rhs,
                "fit_params":         new_params,
                "param_bounds":       merged_bounds,
                "state_vars":         state_vars,
                "input_vars":         input_vars,
                "output_vars":        output_vars,
                "system_order":       system_order,
                "output_state_index": output_state_index,
                "improvable":         True,
                "reestimated":        True,
            },
        )
        new_model_id = self._registry.store_model(artifact)

        # ── Run NLS estimator ─────────────────────────────────────────────────
        est_report   = self._estimator.run(new_model_id, contract_id)
        fitted_id    = est_report.metadata.get("model_id", new_model_id)
        new_run_ids  = est_report.metadata.get("run_ids", [])

        if est_report.status == AgentStatus.FAILED:
            return {"error": f"re-estimation failed: {est_report.summary}", "run_ids": new_run_ids}

        # ── Training RMSE via multi-shooting ──────────────────────────────────
        t, u, y, extra_run_ids = self._load_training_data(dataset_ids, contract_id)
        train_rmse = float("inf")
        if t is not None:
            fitted_model = self._registry.load_model(fitted_id)
            fitted_p     = np.array([fitted_model.parameters.get(p, 1.0) for p in new_params])
            states_est   = _estimate_hidden_states(t, y, system_order)
            train_rmse   = _compute_training_rmse(
                new_rhs, new_params, state_vars, input_vars,
                fitted_p, t, u, y, states_est, system_order, output_state_index,
            )

        fitted_model  = self._registry.load_model(fitted_id)
        fitted_params = {
            p: round(float(fitted_model.parameters.get(p, float("nan"))), 6)
            for p in new_params
        }

        return {
            "model_id":      fitted_id,
            "train_rmse":    float(train_rmse) if np.isfinite(train_rmse) else 9999.0,
            "rung":          Rung.GREY.value,
            "run_ids":       new_run_ids + extra_run_ids,
            "fitted_params": fitted_params,
        }

    def run_with_strategy(
        self,
        strategy: str,           # "coulomb" | "sindy" | "gp" | "sequence"
        physics_model_id: str,
        dataset_ids: List[str],
        contract_id: str,
        seq_model_class: str = "rnn",
        seq_epochs: int = 200,
        seq_len: int = 50,
        seq_hidden_size: int = 64,
        fitting_domain: str = "acceleration",  # "acceleration" | "output"
    ) -> dict:
        """
        Run one specific grey-box correction strategy.
        Returns {"model_id": ..., "train_rmse": ..., "run_ids": [...], "error": ...}
        """
        from agents.greybox.strategy_selector import Strategy as S

        strategy_map = {
            "coulomb":  S.COULOMB_TERM,
            "sindy":    S.SINDY,
            "gp":       S.GP_CORRECTION,
        }

        # ── Resolve physics base ──────────────────────────────────────────────
        model = self._registry.load_model(physics_model_id)
        meta  = model.metadata
        rhs   = meta.get("normalized_rhs", "")
        _SENTINEL = {"RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"}
        if rhs in _SENTINEL:
            physics_model_id = meta.get("physics_model_id", physics_model_id)
            model = self._registry.load_model(physics_model_id)
            meta  = model.metadata
            rhs   = meta["normalized_rhs"]

        fit_params         = meta["fit_params"]
        state_vars         = meta["state_vars"]
        input_vars         = meta["input_vars"]
        param_bounds       = meta.get("param_bounds", {})
        system_order       = meta.get("system_order", len(state_vars))
        output_state_index = meta.get("output_state_index", 0)
        fitted_p = np.array([model.parameters.get(p, 1.0) for p in fit_params])
        vel_idx  = min(1, system_order - 1)
        vel_var  = state_vars[vel_idx] if state_vars else "x_dot"
        output_vars = meta.get("output_vars", [state_vars[0]] if state_vars else ["y"])

        # ── Load training data ────────────────────────────────────────────────
        t, u, y, new_run_ids = self._load_training_data(dataset_ids, contract_id)
        if t is None:
            return {"error": "no training data", "run_ids": []}

        rhs_smooth = _smooth_sign_terms(rhs)
        states_est = _estimate_hidden_states(t, y, system_order)
        vel_v_all  = states_est[vel_idx]

        eps_ddot, vm = self._compute_accel_residuals(
            rhs_smooth, fit_params, state_vars, input_vars, fitted_p, t, u, states_est
        )
        t_v      = t[vm];  states_v = states_est[:, vm];  u_v = u[vm];  eps_v = eps_ddot[vm]
        vel_v    = states_v[vel_idx]

        # ── Output-domain SINDy: LASSO on y_meas − y_base, no differentiation ──
        if strategy == "sindy" and fitting_domain == "output":
            seg_len_sindy = 25
            try:
                base_sim = make_ode_simulator(
                    rhs_smooth, fit_params, state_vars, input_vars,
                    highest_deriv_var=state_vars[-1] + "_ddot" if state_vars else "x_ddot",
                    output_state_index=output_state_index,
                )
            except Exception as exc:
                return {"error": f"output SINDy: cannot build base simulator: {exc}", "run_ids": new_run_ids}

            e_parts, states_parts, u_parts = [], [], []
            N_t = len(t)
            for seg_i in range(0, N_t - seg_len_sindy, seg_len_sindy):
                t_s      = t[seg_i:seg_i + seg_len_sindy]
                u_s      = u[seg_i:seg_i + seg_len_sindy]
                y_s      = y[seg_i:seg_i + seg_len_sindy]
                x0_s     = states_est[:, seg_i]
                states_s = states_est[:, seg_i:seg_i + seg_len_sindy]
                try:
                    y_p = base_sim(fitted_p, t_s, u_s, x0=x0_s)
                    if not np.any(np.isnan(y_p)):
                        e_parts.append(y_s - y_p)
                        states_parts.append(states_s)
                        u_parts.append(u_s)
                except Exception:
                    pass

            if not e_parts:
                return {"error": "output SINDy: all segments failed during base ODE simulation", "run_ids": new_run_ids}

            e_out_sindy      = np.concatenate(e_parts)
            states_sindy     = np.concatenate(states_parts, axis=1)
            u_sindy          = np.concatenate(u_parts)

            out_spec = self._trainer._fit_sindy_output_domain(
                base_params=fit_params,
                base_param_bounds=param_bounds,
                states=states_sindy,
                u=u_sindy,
                e_out=e_out_sindy,
                state_names=state_vars,
                input_names=input_vars,
                base_rhs=rhs_smooth,
            )

            # Store as a new model with SINDY_OUTPUT_CORRECTED sentinel RHS.
            # The correction is stored directly in metadata as a coefficient dict —
            # no separate pickled object needed (it's just a small dict of floats).
            from core.schemas import ModelArtifact, ModelType, Rung
            artifact = ModelArtifact(
                model_type=ModelType.GREY_BOX,
                structure_description=(
                    f"Output-domain SINDy correction: "
                    f"y_pred = y_physics + {out_spec.correction_expr}"
                ),
                parameters={p: float(v) for p, v in zip(fit_params, fitted_p)},
                parent_id=physics_model_id,
                metadata={
                    "normalized_rhs":       "SINDY_OUTPUT_CORRECTED",
                    "physics_model_id":     physics_model_id,
                    "correction_coeffs":    out_spec.correction_coeffs,
                    "correction_expr":      out_spec.correction_expr,
                    "fit_params":           fit_params,
                    "param_bounds":         param_bounds,
                    "state_vars":           state_vars,
                    "input_vars":           input_vars,
                    "output_vars":          output_vars,
                    "system_order":         system_order,
                    "output_state_index":   output_state_index,
                    "fitting_domain":       "output",
                    "improvable":           False,
                    "model_class":          "sindy_output",
                },
            )
            model_id_out = self._registry.store_model(artifact)

            # Training RMSE: Theta @ c on the same segments used for fitting
            if out_spec.correction_coeffs:
                from tools.feature_library import FeatureLibrary
                _Theta, _names = FeatureLibrary().build(states_sindy, u_sindy, state_vars, input_vars)
                _c = np.array([out_spec.correction_coeffs.get(n, 0.0) for n in _names])
                train_rmse_out = float(np.sqrt(np.mean((e_out_sindy - _Theta @ _c) ** 2)))
            else:
                train_rmse_out = float(np.sqrt(np.mean(e_out_sindy ** 2)))

            return {
                "model_id":   model_id_out,
                "train_rmse": round(train_rmse_out, 6),
                "rung":       Rung.GREY.value,
                "run_ids":    new_run_ids,
                "fitting_domain": "output",
                "correction_expr": out_spec.correction_expr,
            }

        # ── Sequence strategy: operates on output residuals ───────────────────
        if strategy == "sequence":
            e_out, t_seq, u_seq = _compute_output_residuals(
                rhs_smooth, fit_params, state_vars, input_vars, fitted_p,
                t, u, y, system_order, output_state_index, states_est,
            )
            if len(e_out) < 20:
                return {"error": "too few output residual samples for sequence training", "run_ids": new_run_ids}

            from agents.surrogate.model_class_selector import ModelClass, Paradigm
            mc = ModelClass(seq_model_class) if seq_model_class in (mc.value for mc in ModelClass) else ModelClass.RNN
            try:
                if mc == ModelClass.NARX:
                    res = self._seq_trainer.fit_narx(e_out, u_seq)
                else:
                    res = self._seq_trainer.fit_rnn(
                        e_out, u_seq,
                        n_epochs=seq_epochs, seq_len=seq_len, hidden_size=seq_hidden_size,
                    )
            except Exception as exc:
                return {"error": f"sequence training failed: {exc}", "run_ids": new_run_ids}

            seq_obj_id = physics_model_id + f"_res_seq_{mc.value}"
            self._registry.store_object(seq_obj_id, res.predictor)
            from core.schemas import ModelArtifact, ModelType, Rung
            artifact = ModelArtifact(
                model_type=ModelType.GREY_BOX,
                structure_description=f"Residual surrogate ({mc.value}): physics + {mc.value} correction",
                parameters={},
                parent_id=physics_model_id,
                metadata={
                    "normalized_rhs":       "RESIDUAL_CORRECTED",
                    "physics_model_id":     physics_model_id,
                    "residual_object_id":   seq_obj_id,
                    "residual_model_class": mc.value,
                    "fit_params":           fit_params,
                    "param_bounds":         param_bounds,
                    "state_vars":           state_vars,
                    "input_vars":           input_vars,
                    "output_vars":          output_vars,
                    "system_order":         system_order,
                    "output_state_index":   output_state_index,
                    "improvable":           False,
                    "train_rmse":           res.train_rmse,
                    "model_class":          f"sequence_{mc.value}",
                    "seq_len":              seq_len,
                    "hidden_size":          seq_hidden_size,
                },
            )
            model_id = self._registry.store_model(artifact)
            return {
                "model_id":    model_id,
                "train_rmse":  res.train_rmse,
                "run_ids":     new_run_ids,
                "seq_len":     seq_len,
                "hidden_size": seq_hidden_size,
            }

        # ── Symbolic strategies: Coulomb / SINDY / GP ─────────────────────────
        if strategy not in strategy_map:
            return {"error": f"unknown strategy '{strategy}'", "run_ids": new_run_ids}
        chosen = strategy_map[strategy]

        diag = self._selector.diagnose_general(states_v, u_v, eps_v, state_vars, input_vars)

        spec = self._trainer.fit(
            strategy=chosen,
            base_rhs=rhs_smooth,
            base_params=fit_params,
            base_param_bounds=param_bounds,
            base_fitted_params={p: float(v) for p, v in zip(fit_params, fitted_p)},
            t=t_v, theta_dot=vel_v, eps_ddot=eps_v,
            vel_var_name=vel_var,
            states=states_v, state_names=state_vars,
            input_names=input_vars, u=u_v,
        )

        from core.schemas import ModelArtifact, ModelType, Rung
        if chosen == S.COULOMB_TERM:
            mtype, new_rung = ModelType.WHITE_BOX, Rung.WHITE
        else:
            mtype, new_rung = ModelType.GREY_BOX, Rung.GREY

        is_gp = (chosen == S.GP_CORRECTION)
        extended_rhs = "GP_CORRECTED" if is_gp else _smooth_sign_terms(spec.normalized_rhs)
        artifact = ModelArtifact(
            model_type=mtype,
            structure_description=f"Extended ({chosen.value}): {spec.normalized_rhs if not is_gp else 'GP correction'}",
            parameters={p: float(v) for p, v in zip(fit_params, fitted_p)} if is_gp else spec.p0_override,
            parent_id=physics_model_id,
            metadata={
                "normalized_rhs":     extended_rhs,
                "base_rhs":           rhs_smooth,
                "fit_params":         spec.fit_params,
                "param_bounds":       spec.param_bounds,
                "state_vars":         state_vars,
                "input_vars":         input_vars,
                "output_vars":        output_vars,
                "system_order":       system_order,
                "output_state_index": output_state_index,
                "improvable":         False,
                "greybox_strategy":   chosen.value,
                "correction_coeffs":  spec.correction_coeffs,
                "p0_override":        spec.p0_override,
                "model_class":        chosen.value,
            },
        )
        extended_id = self._registry.store_model(artifact)

        if is_gp and spec.correction_object is not None:
            gp_obj_id = extended_id + "_gp_corr"
            self._registry.store_object(gp_obj_id, spec.correction_object)
            artifact = artifact.model_copy(update={"metadata": {**artifact.metadata, "correction_object_id": gp_obj_id}})
            self._registry.store_model(artifact)

        # Re-estimate params for non-GP symbolic paths
        fitted_id  = extended_id
        new_run_ids_est: List[str] = []
        if not is_gp:
            est_report = self._estimator.run(extended_id, contract_id)
            fitted_id  = est_report.metadata.get("model_id", extended_id)
            new_run_ids_est = est_report.metadata.get("run_ids", [])

        # Compute training RMSE.
        # For the GP path use the analytically computed LOO RMSE stored by _fit_gp
        # (Rasmussen & Williams §5.4.2).  This avoids running ODE integration with a
        # GP correction at every step — which stalls RK45 when large corrections make
        # the trajectory stiff.
        fitted_model = self._registry.load_model(fitted_id)
        fp = np.array([fitted_model.parameters.get(p, 1.0) for p in spec.fit_params])
        if is_gp:
            gp_loo = spec.correction_coeffs.get("gp_loo_rmse")
            if gp_loo is not None:
                train_rmse = float(gp_loo)
                logger.debug("run_with_strategy GP: LOO train_rmse=%.4f (skipped ODE integration)", train_rmse)
            else:
                gp_corr_fn = None
                gp_obj_id  = extended_id + "_gp_corr"
                try:
                    gp_corr_fn = self._registry.load_object(gp_obj_id)
                except Exception as _exc:
                    logger.debug("run_with_strategy GP: could not load correction object %s — %s", gp_obj_id, _exc)
                train_rmse = _compute_training_rmse(
                    rhs_smooth, spec.fit_params, state_vars, input_vars,
                    fp, t, u, y, states_est, system_order, output_state_index,
                    correction_fn=gp_corr_fn,
                )
        else:
            train_rmse = _compute_training_rmse(
                extended_rhs, spec.fit_params, state_vars, input_vars,
                fp, t, u, y, states_est, system_order, output_state_index,
            )
        logger.debug("run_with_strategy(%s): train_rmse=%.4f", strategy, train_rmse)
        # Store train_rmse in metadata
        updated_meta = {**fitted_model.metadata, "train_rmse": train_rmse, "rung": new_rung.value}
        self._registry.store_model(fitted_model.model_copy(update={"metadata": updated_meta}))

        # Sanitize Infinity/NaN so the returned dict is always valid JSON
        train_rmse_json = float(train_rmse) if np.isfinite(train_rmse) else 9999.0

        return {
            "model_id":   fitted_id,
            "train_rmse": train_rmse_json,
            "rung":       new_rung.value,
            "run_ids":    new_run_ids + new_run_ids_est,
        }

    def collect_data(
        self,
        contract_id: str,
        n_samples: int = 600,
        strategy: str = "prbs",
        seed: int = 200,
        amplitude: Optional[float] = None,
        f_lo: Optional[float] = None,
        f_hi: Optional[float] = None,
    ) -> dict:
        """
        Run a new identification experiment.

        amplitude : peak amplitude in plant input units (e.g. 1.5 N·m).
                    Defaults to 70% of the contract limit when not specified.
        f_lo/f_hi : frequency band in Hz for chirp/multisine signals.
                    Defaults to [0.1, Nyquist/2] when not specified.

        Returns {"run_id": ..., "n_samples": ..., "signal_spec": {...}}
        """
        from agents.experiment_design import ExperimentDesignAgent
        designer = ExperimentDesignAgent()
        contract = self._load_contract(contract_id)

        input_name = contract.input_names[0]
        lo, hi     = contract.input_limits.get(input_name, (-1.0, 1.0))
        half_range = min(abs(lo), abs(hi))
        nyquist    = 0.5 / contract.sample_time

        # Convert absolute amplitude → fraction expected by the toolkit
        amp_fraction = float(np.clip(amplitude / half_range, 0.05, 1.0)) \
                       if amplitude is not None else 0.70

        # Build frequency range tuple if either bound was specified
        freq_range = None
        if f_lo is not None or f_hi is not None:
            freq_range = (
                float(np.clip(f_lo, 1e-3, nyquist * 0.99)) if f_lo is not None else 0.1,
                float(np.clip(f_hi, 1e-3, nyquist * 0.99)) if f_hi is not None else nyquist * 0.9,
            )
            if freq_range[0] >= freq_range[1]:
                return {"error": f"f_lo ({freq_range[0]:.3f}) must be less than f_hi ({freq_range[1]:.3f})"}

        try:
            if strategy == "compound":
                # PRBS + multisine broadband: amplitude/freq params not applicable
                seq = designer.design_broadband(contract, n_samples=n_samples, seed=seed)
                signal_spec = {
                    "strategy":  "compound",
                    "amplitude": None,
                    "f_lo_hz":   None,
                    "f_hi_hz":   None,
                    "n_samples": n_samples,
                    "seed":      seed,
                }
            else:
                method = strategy if strategy in ("prbs", "chirp", "multisine", "steps") else "prbs"
                seq = designer.design_for_identification(
                    contract,
                    n_samples=n_samples,
                    method=method,
                    seed=seed,
                    amplitude_fraction=amp_fraction,
                    frequency_range=freq_range,
                )
                signal_spec = {
                    "strategy":  method,
                    "amplitude": float(amp_fraction * half_range),
                    "f_lo_hz":   freq_range[0] if freq_range else None,
                    "f_hi_hz":   freq_range[1] if freq_range else None,
                    "n_samples": n_samples,
                    "seed":      seed,
                }

            u_func = designer.make_u_func(seq["t"], seq["u"])
            result = self._api.apply_input(
                u_func=u_func,
                t_span=(float(seq["t"][0]), float(seq["t"][-1])),
                dt=float(seq["t"][1] - seq["t"][0]),
                purpose="identification",
                input_type=seq["input_type"],
                agent="greybox_agent",
                split_flag=SplitFlag.TRAIN,
            )
            return {"run_id": result["run_id"], "n_samples": n_samples, "signal_spec": signal_spec}
        except Exception as exc:
            return {"error": str(exc)}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_training_data(
        self,
        dataset_ids: List[str],
        contract_id: str,
        force_collect: bool = False,
    ):
        """Returns (t, u, y, new_run_ids) where new_run_ids lists any freshly collected runs."""
        t_parts, u_parts, y_parts = [], [], []
        new_run_ids: List[str] = []

        for run_id in dataset_ids:
            try:
                t_r, u_r, y_r = self._db.load_arrays(run_id)
                t_parts.append(t_r)
                u_parts.append(u_r[0])
                y_parts.append(y_r[0])
            except Exception as exc:
                logger.warning("GreyboxSO: skipping run_id=%s — %s", run_id, exc)

        if not t_parts or force_collect:
            seed = 99 if not force_collect else 101  # different seed on retry for variety
            try:
                from agents.experiment_design import ExperimentDesignAgent
                from core.schemas import PlantContract
                designer = ExperimentDesignAgent()
                contract = self._load_contract(contract_id)
                seq = designer.design_for_identification(contract, n_samples=600, seed=seed)
                u_func = designer.make_u_func(seq["t"], seq["u"])
                result = self._api.apply_input(
                    u_func=u_func,
                    t_span=(float(seq["t"][0]), float(seq["t"][-1])),
                    dt=float(seq["t"][1] - seq["t"][0]),
                    purpose="identification",
                    input_type=seq["input_type"],
                    agent="greybox_so",
                    split_flag=SplitFlag.TRAIN,
                )
                new_run_ids.append(result["run_id"])
                t_r, u_r, y_r = self._db.load_arrays(result["run_id"])
                t_parts.append(t_r)
                u_parts.append(u_r[0])
                y_parts.append(y_r[0])
                logger.info(
                    "GreyboxSO: collected %d new samples (run_id=%s, force=%s)",
                    len(t_r), result["run_id"], force_collect,
                )
            except Exception as exc:
                logger.error("GreyboxSO: failed to collect new data: %s", exc)
                if not t_parts:
                    return None, None, None, []

        return (
            np.concatenate(t_parts),
            np.concatenate(u_parts),
            np.concatenate(y_parts),
            new_run_ids,
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

    @staticmethod
    def _compute_accel_residuals(
        rhs:        str,
        fit_params: List[str],
        state_vars: List[str],
        input_vars: List[str],
        fitted_p:   np.ndarray,
        t:          np.ndarray,
        u:          np.ndarray,
        states_est: np.ndarray,   # (system_order, N) — all state estimates
    ):
        """
        Compute acceleration residuals ε = highest_deriv_measured − highest_deriv_model.

        highest_deriv_model is evaluated analytically at each sample using all
        state estimates: f_rhs(fitted_params, x0[i], x1[i], ..., u[i]).

        highest_deriv_measured is np.gradient of the highest-order state estimate.
        This is correct for any system order: for order-2 that is d/dt(theta_dot),
        for order-1 it is d/dt(theta), for order-3 it is d/dt(theta_ddot), etc.
        """
        system_order = len(state_vars)
        locs     = _make_locals(fit_params, state_vars, input_vars)
        expr     = sp.sympify(rhs, locals=locs)
        all_syms = [locs[n] for n in fit_params + state_vars + input_vars]
        f_rhs    = sp.lambdify(all_syms, expr, modules=["numpy"])

        # Build list of all state arrays (one per state variable)
        state_args = [states_est[i] for i in range(system_order)]

        try:
            highest_model = f_rhs(*list(fitted_p), *state_args, u)
            highest_model = np.asarray(highest_model, dtype=float).ravel()
        except Exception:
            highest_model = np.zeros(states_est.shape[1])

        # Measured highest derivative: SG derivative of the highest-order state estimate
        highest_meas = _sg_deriv(states_est[-1], t, deriv=1)

        eps_ddot = highest_meas - highest_model
        eps_ddot = np.clip(eps_ddot, -100.0, 100.0)

        valid    = np.isfinite(eps_ddot)
        valid[:2]  = False
        valid[-2:] = False

        return eps_ddot, valid

    def _fit_residual_sequence(
        self,
        e_output:  np.ndarray,
        u_seq:     np.ndarray,
        n_epochs:  int = 200,
    ) -> "Optional[tuple]":
        """
        Train a sequence model on output residuals e = y_measured − y_physics.

        Model class selected by sample count (same thresholds as surrogate IO paradigm).
        Returns (predictor, train_rmse, class_name) or None on failure.
        """
        sel = self._class_sel.select(len(e_output), Paradigm.INPUT_OUTPUT)
        try:
            if sel.model_class == ModelClass.NARX:
                result = self._seq_trainer.fit_narx(e_output, u_seq)
            elif sel.model_class == ModelClass.RNN:
                result = self._seq_trainer.fit_rnn(e_output, u_seq, n_epochs=n_epochs)
            else:
                result = self._seq_trainer.fit_transformer(e_output, u_seq, n_epochs=n_epochs)
        except Exception as exc:
            logger.warning("GreyboxSO: residual sequence fit failed — %s", exc)
            return None
        return result.predictor, result.train_rmse, sel.model_class.value


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_output_residuals(
    rhs:               str,
    fit_params:        List[str],
    state_vars:        List[str],
    input_vars:        List[str],
    fitted_p:          np.ndarray,
    t:                 np.ndarray,
    u:                 np.ndarray,
    y_measured:        np.ndarray,
    system_order:      int,
    output_state_index: int,
    states_est:        np.ndarray,
    seg_len:           int = 25,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Compute output residuals e[k] = y_measured[k] − y_physics_simulated[k]
    using multi-shooting segments.  No numerical differentiation of the output.

    Returns (e_output, t_aligned, u_aligned) — concatenated arrays from all
    complete segments that didn't produce NaN.
    """
    try:
        simulator = make_ode_simulator(
            rhs, fit_params, state_vars, input_vars,
            highest_deriv_var=state_vars[-1] + "_ddot" if state_vars else "x_ddot",
            output_state_index=output_state_index,
        )
    except Exception:
        return np.array([]), np.array([]), np.array([])

    N = len(t)
    e_parts, t_parts, u_parts = [], [], []
    for i in range(0, N - seg_len, seg_len):
        t_s  = t[i:i + seg_len]
        u_s  = u[i:i + seg_len]
        y_s  = y_measured[i:i + seg_len]
        x0_s = states_est[:, i]
        try:
            y_p = simulator(fitted_p, t_s, u_s, x0=x0_s)
            if not np.any(np.isnan(y_p)):
                e_parts.append(y_s - y_p)
                t_parts.append(t_s)
                u_parts.append(u_s)
        except Exception:
            pass

    if not e_parts:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(e_parts),
        np.concatenate(t_parts),
        np.concatenate(u_parts),
    )


def _compute_training_rmse(
    rhs:               str,
    fit_params:        List[str],
    state_vars:        List[str],
    input_vars:        List[str],
    fitted_p:          np.ndarray,
    t:                 np.ndarray,
    u:                 np.ndarray,
    y_measured:        np.ndarray,
    states_est:        np.ndarray,
    system_order:      int,
    output_state_index: int,
    seg_len:           int = 25,
    correction_fn=None,
) -> float:
    """Output RMSE on training data via multi-shooting.  Returns inf on any error."""
    try:
        sim = make_ode_simulator(
            rhs, fit_params, state_vars, input_vars,
            highest_deriv_var=state_vars[-1] + "_ddot" if state_vars else "x_ddot",
            output_state_index=output_state_index,
            correction_fn=correction_fn,
        )
        residuals = _multi_shoot(sim, fitted_p, t, u, y_measured, states_est, seg_len)
        return float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else np.inf
    except Exception:
        return np.inf


def _smooth_sign_terms(rhs: str, eps: float = 0.01) -> str:
    """
    Replace sign(expr) with tanh((expr)/eps) in an ODE RHS string.

    Discontinuous sign() makes ODE integrators stiff and causes NLS to converge
    poorly.  tanh(x/ε) is a smooth approximation that recovers sign(x) as ε→0.
    Only handles non-nested arguments (the common case: sign(variable_name)).
    """
    import re
    def _replace(m: "re.Match") -> str:
        inner = m.group(1)
        return f"tanh(({inner})/{eps})"
    new_rhs = re.sub(r"sign\(([^()]*)\)", _replace, rhs)
    if new_rhs != rhs:
        logger.info("GreyboxSO: replaced sign() with tanh() in RHS for smooth fitting")
    return new_rhs


def _fail(dossier: Dossier, msg: str, strategy: str = "") -> Dossier:
    report = Report(
        agent="GreyBoxSubOrchestrator",
        status=AgentStatus.FAILED,
        summary=msg,
        metadata={"strategy": strategy},
    )
    return dossier.update(
        status=f"greybox_so failed: {msg}",
        last_report=report,
    )
