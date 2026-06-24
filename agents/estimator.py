"""
Estimator agent — fits model parameters via an inner excite↔estimate loop.

This agent is primarily deterministic (NLS fitting + covariance check).
The LLM is not needed here; the loop logic and stopping rules are explicit.

Inner loop:
    design input → apply to plant → fit parameters → assess covariance → repeat
Stops when: covariance acceptable | budget slice exhausted | no further improvement
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
    SplitFlag,
)
from agents.experiment_design import ExperimentDesignAgent
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI
from tools.solver_toolkit import nonlinear_least_squares
from tools.symbolic_math import make_ode_simulator

logger = logging.getLogger(__name__)

MAX_INNER_ITER   = 5      # maximum excite↔estimate repetitions
MIN_INNER_ITER   = 3      # minimum iterations before CV convergence is allowed to stop the loop
COV_TARGET_CV    = 0.001   # stop when all params have CV < 0.1%
ENABLE_AMS       = False   # set to False to skip AMS refinement (useful during testing)


class EstimatorAgent:
    """
    Fits the white-box model parameters.

    Given a model structure stored in the registry, this agent:
      1. Reads the reparameterized ODE and fit_params from model metadata
      2. Runs informative PRBS experiments on the plant
      3. Fits parameters via nonlinear least-squares (simulation-based)
      4. Iterates until covariance is acceptable or budget is exhausted
      5. Stores the fitted model + covariance and posts a Report
    """

    def __init__(
        self,
        plant_api:         PlantAPI,
        registry:          ModelRegistry,
        db:                ExperimentDatabase,
        n_samples:         int = 600,
        retrieval_service=None,
    ):
        self._api       = plant_api
        self._registry  = registry
        self._db        = db
        self._n         = n_samples
        self._designer  = ExperimentDesignAgent()
        self._retrieval = retrieval_service

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id = dossier.artifacts.current_model_id
        if not model_id:
            return _fail(dossier, "Estimator: no model_id in dossier")

        guidance = _extract_guidance(dossier)
        report = self.run(model_id, dossier.assets.plant_contract_id or "", guidance=guidance)

        meta = report.metadata
        fitted_id  = meta.get("model_id", model_id)
        cov_id     = meta.get("covariance_id", "")

        return dossier.update(
            status=f"estimator done: model={fitted_id}",
            re_estimate_count=dossier.re_estimate_count + 1,
            artifacts=dossier.artifacts.model_copy(update={
                "current_model_id": fitted_id,
                "model_history": dossier.artifacts.model_history + [fitted_id],
                "dataset_ids": dossier.artifacts.dataset_ids + meta.get("run_ids", []),
            }),
            last_report=report,
        )

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def run(self, model_id: str, contract_id: str = "", guidance: dict = None) -> Report:
        """
        Fit parameters for the model in the registry.
        Returns a Report with the fitted model_id and covariance_id.
        """
        guidance = guidance or {}
        re_est   = guidance.get("re_estimate_count", 0)

        # Use the experiment plan from ExperimentPlannerAgent when available.
        # Fall back to the original rule-based defaults so the estimator still
        # works correctly if called without a planner (tests, direct invocation).
        plan = guidance.get("experiment_plan")  # ExperimentPlan or None
        if plan is not None:
            base_amplitude          = float(plan.base_amplitude)
            _max_amplitude          = float(plan.max_amplitude)
            seg_len                 = int(plan.seg_len)
            _methods_cycle_override = list(plan.methods)
            _amplitude_schedule     = list(plan.amplitude_schedule) if plan.amplitude_schedule else None
            logger.info(
                "[EstimatorAgent] using ExperimentPlan: methods=%s amp=%.2f–%.2f seg_len=%d "
                "schedule=%s | %s",
                _methods_cycle_override, base_amplitude, _max_amplitude, seg_len,
                [round(a, 2) for a in _amplitude_schedule] if _amplitude_schedule else "auto",
                plan.reasoning,
            )
        else:
            # Rule-based fallback
            _methods_cycle_override = None
            _amplitude_schedule     = None
            _max_amplitude = 0.92
            if re_est == 0:
                base_amplitude = 0.65
            else:
                worst_amp      = guidance.get("worst_case_amplitude", 0.85)
                base_amplitude = max(0.80, min(float(worst_amp), 0.90))
            seg_len = 50 if re_est == 0 else min(50 + 70 * re_est, 200)
            logger.info(
                "[EstimatorAgent] re_estimate=%d | base_amplitude=%.2f | seg_len=%d (rule-based)",
                re_est, base_amplitude, seg_len,
            )

        if guidance.get("failure_hypothesis"):
            logger.info("[EstimatorAgent] guidance: %s", guidance["failure_hypothesis"][:200])

        # Load model structure
        model = self._registry.load_model(model_id)
        meta  = model.metadata
        rhs        = meta["normalized_rhs"]
        fit_params = meta["fit_params"]
        state_vars = meta["state_vars"]
        input_vars = meta["input_vars"]
        output_var = meta["output_vars"][0]

        # Load contract to get plant limits
        contract = self._load_contract(contract_id)

        # System order and output selection (stored by modeler; default: 2nd-order, output=state[0])
        system_order       = meta.get("system_order", len(state_vars))
        output_state_index = meta.get("output_state_index", 0)

        # Compile numerical ODE simulator (output-only) and full-state variant for AMS
        simulator = make_ode_simulator(
            rhs, fit_params, state_vars, input_vars,
            highest_deriv_var=state_vars[-1] + "_ddot",
            output_state_index=output_state_index,
        )
        simulator_full = make_ode_simulator(
            rhs, fit_params, state_vars, input_vars,
            highest_deriv_var=state_vars[-1] + "_ddot",
            output_state_index=output_state_index,
            return_full_state=True,
        )

        # Tighten param_bounds using prior runs from episodic memory
        param_bounds = meta.get("param_bounds", {})
        if self._retrieval is not None:
            query = (
                f"plant order={system_order} states={state_vars} inputs={input_vars} "
                f"ODE: {rhs[:120]}"
            )
            prior_runs = self._retrieval.query_runs_only(
                query, top_k=3, filter_system_order=system_order
            )
            if prior_runs:
                param_bounds = _tighten_bounds(fit_params, param_bounds, prior_runs)
                logger.debug("Estimator: tightened bounds from %d prior run(s)", len(prior_runs))

        logger.info("[EstimatorAgent] param_bounds going into NLS: %s",
                    {p: param_bounds.get(p, "unbounded") for p in fit_params})
        # p0_override: pre-specified warm-start values (e.g. K_c from grey-box SO).
        # On re-estimation, also warm-start from previously fitted parameters so
        # the NLS refines rather than restarts from the geometric-mean guess.
        p0_override = dict(meta.get("p0_override", {}))
        if re_est > 0 and model.parameters:
            for p, v in model.parameters.items():
                if p in fit_params and p not in p0_override:
                    p0_override[p] = v
            logger.info("[EstimatorAgent] warm-starting from previous fit: %s",
                        {p: f"{v:.4f}" for p, v in p0_override.items()})

        p0 = self._initial_guess(fit_params, param_bounds, p0_override)
        lo, hi = self._split_bounds(fit_params, param_bounds)

        best_params = p0.copy()
        best_cov    = np.full((len(p0), len(p0)), np.inf)
        run_ids: List[str] = []
        converged = False
        stall_reason = ""
        ols_done = False   # compute OLS initial guess on first dataset

        # Joint NLS: accumulate all datasets across iterations.
        # Each NLS call fits ALL collected data simultaneously so that
        # no single high-amplitude dataset can override parameters that
        # were well-identified from earlier low-amplitude or step data.
        all_datasets: List[tuple] = []   # (t, u, y, states_est)
        raw_datasets: List[tuple] = []   # (t, u, y) — raw, for UKF re-estimation
        amplitudes_used: List[float] = []
        methods_used_log: List[str]  = []

        # Method cycle: use the planner's choice when available; fall back to rule-based.
        if _methods_cycle_override is not None:
            _methods_cycle = _methods_cycle_override
        elif re_est >= 1:
            _methods_cycle = ["prbs", "multisine", "steps", "prbs", "multisine"]
        else:
            _methods_cycle = ["prbs", "steps", "prbs", "steps", "prbs"]

        # Build a stratified default schedule that always covers low, medium, and max tiers.
        # This ensures K_s (restoring-force term) stays identifiable even when the plan targets
        # a high worst-case amplitude — K_s can only be identified from oscillating (non-spinning)
        # data; spinning data (high amplitude) makes K_s estimation degrade.
        if _amplitude_schedule is None:
            _low = round(min(0.40, base_amplitude * 0.55), 2)
            _mid = round(min(0.65, base_amplitude * 0.85), 2)
            _amplitude_schedule = [_low, _mid, _low, base_amplitude, _max_amplitude]
            logger.info("[EstimatorAgent] stratified default schedule: %s", _amplitude_schedule)

        # Compile a fast batched RK4 step for UKF sigma-point propagation.
        # This replaces the Savitzky-Golay derivative from iteration 1 onward,
        # giving physics-informed θ̇ estimates at segment boundaries and
        # eliminating the SG bias that inflates K_d.
        _ukf_f_step = None
        try:
            from tools.symbolic_math import make_rk4_step
            _ukf_f_step = make_rk4_step(rhs, fit_params, state_vars, input_vars)
        except Exception as _e:
            logger.debug("[EstimatorAgent] UKF f_step compile failed (%s) — using SG", _e)

        # Process noise Q: tiny for position (well-tracked by measurements),
        # moderate for velocity (hidden, model mismatch allowed).
        # Measurement noise R = noise_std² ≈ 1e-6 for the pendulum.
        _ukf_Q = np.diag([1e-8] + [1e-2] * (system_order - 1))
        _ukf_R = float(meta.get("noise_std", 0.001)) ** 2

        for iteration in range(MAX_INNER_ITER):
            method = _methods_cycle[iteration % len(_methods_cycle)]
            # Use the explicit schedule (from planner or stratified default).
            amplitude = float(np.clip(
                _amplitude_schedule[iteration % len(_amplitude_schedule)], 0.05, 1.0
            ))
            amplitudes_used.append(round(amplitude, 3))
            methods_used_log.append(method)
            logger.info(
                "[EstimatorAgent] iter %d | method=%s | amplitude=%.2f | seg_len=%d",
                iteration, method, amplitude, seg_len,
            )
            seq = self._designer.design_for_identification(
                contract, n_samples=self._n, method=method, seed=iteration,
                amplitude_fraction=amplitude,
            )
            t_des, u_des = seq["t"], seq["u"]
            u_func = self._designer.make_u_func(t_des, u_des)

            # Apply to plant
            try:
                result = self._api.apply_input(
                    u_func=u_func,
                    t_span=(float(t_des[0]), float(t_des[-1])),
                    dt=float(t_des[1] - t_des[0]),
                    purpose="identification",
                    input_type=seq["input_type"],
                    agent="estimator",
                    split_flag=SplitFlag.TRAIN,
                )
            except Exception as exc:
                stall_reason = f"budget_or_safety: {exc}"
                break

            run_ids.append(result["run_id"])

            # Load data
            t_data, u_data, y_data = self._db.load_arrays(result["run_id"])
            y_true = y_data[0]
            u_flat = u_data[0]

            # Set geometric-mean initial guess on the first dataset.
            # We previously used OLS on double-differentiated theta here, but
            # d²θ/dt² noise = (noise_std / dt²) which at dt=0.02 amplifies
            # measurement noise ≈2500× — badly corrupting the warm start.
            # The geometric mean of the prior bounds is noiseless and lands
            # within 5× of any parameter in a decade-wide bound interval.
            if not ols_done:
                for i, p in enumerate(fit_params):
                    if p not in p0_override:
                        lo_i = max(lo[i], 1e-4)   # guard against zero lower bound
                        best_params[i] = float(np.sqrt(lo_i * hi[i]))
                ols_done = True

            # State estimation for multi-shooting segment boundaries.
            # Iteration 0: Savitzky-Golay (no physics params yet).
            # Iteration 1+: UKF/RTS smoother using current best_params.
            #   The UKF propagates sigma points through the compiled RK4 step,
            #   then the RTS backward pass refines every θ̇ estimate using
            #   future measurements — eliminating the SG boundary bias that
            #   drives K_d away from its true value.
            raw_datasets.append((t_data, u_flat, y_true))

            if _ukf_f_step is not None and iteration > 0:
                from tools.solver_toolkit import ukf_smooth
                # Only smooth the newly collected dataset; historical datasets keep
                # their SG/prior-UKF state estimates frozen.  Re-smoothing all datasets
                # with best_params makes the NLS cost flat at best_params (the UKF
                # produces ICs consistent with those params, so the gradient vanishes
                # and the optimizer stagnates regardless of what new data was collected).
                t_d, u_d, y_d = raw_datasets[-1]
                x0_ukf = np.zeros(system_order)
                x0_ukf[output_state_index] = float(y_d[0])
                if contract.x0 is not None:
                    for _ki, _vi in enumerate(contract.x0[:system_order]):
                        x0_ukf[_ki] = float(_vi)
                P0_ukf = np.eye(system_order)
                P0_ukf[output_state_index, output_state_index] = _ukf_R
                for _ki in range(system_order):
                    if _ki != output_state_index:
                        P0_ukf[_ki, _ki] = 10.0
                try:
                    s_est = ukf_smooth(
                        _ukf_f_step, best_params, t_d, u_d, y_d,
                        system_order, output_state_index,
                        _ukf_Q, _ukf_R, x0=x0_ukf, P0=P0_ukf,
                    )
                    s_est[output_state_index] = y_d   # pin to actual measurement
                    if not np.all(np.isfinite(s_est)):
                        raise ValueError("non-finite UKF output")
                except Exception as _ukf_err:
                    logger.debug(
                        "[EstimatorAgent] UKF fallback to SG (iter %d): %s",
                        iteration, _ukf_err,
                    )
                    s_est = _estimate_hidden_states(t_d, y_d, system_order)
                    if contract.x0 is not None:
                        for _ki, _vi in enumerate(contract.x0[:system_order]):
                            s_est[_ki, 0] = float(_vi)
                all_datasets.append((t_d, u_d, y_d, s_est))
                logger.info(
                    "[EstimatorAgent] iter %d | UKF smoother applied to new dataset (total=%d)",
                    iteration, len(all_datasets),
                )
            else:
                # Iteration 0 (or f_step unavailable): fall back to SG.
                states_est = _estimate_hidden_states(t_data, y_true, system_order)
                if contract.x0 is not None:
                    for _k, _v in enumerate(contract.x0[:system_order]):
                        states_est[_k, 0] = float(_v)
                all_datasets.append((t_data, u_flat, y_true, states_est))

            # Joint residuals: concatenation over ALL accumulated datasets.
            # This prevents a high-amplitude PRBS dataset from corrupting a
            # parameter that was well-identified by earlier step or low-amp data.
            _sl   = seg_len            # capture in closure
            _snap = list(all_datasets)  # snapshot for closure

            def residuals(params, _sl=_sl, _snap=_snap):
                parts = [
                    _multi_shoot(simulator, params, t_d, u_d, y_d, s_d, _sl)
                    for t_d, u_d, y_d, s_d in _snap
                ]
                return np.concatenate(parts) if parts else np.array([0.0])

            # Multi-start NLS: always run the warm-start plus three perturbed restarts
            # at every iteration.  Local minima are the norm here (iter-0 NLS already
            # minimised dataset-0, so subsequent iterations start at a flat point), and
            # the stagnation gate (rel_change < 1e-4) misses cases where the warm-start
            # accidentally lands in a worse basin.  Split the total nfev budget evenly
            # across all starts so total cost stays the same as before.
            best_params = np.clip(best_params, lo, hi)
            _n_starts   = 4 if iteration > 0 else 1   # iter 0 has no prior data to escape from
            _nfev_total = min(300 * len(all_datasets), 1500)
            _nfev_each  = max(100, _nfev_total // _n_starts)
            _rng        = np.random.default_rng(iteration * 17 + 42)

            fit = nonlinear_least_squares(
                residuals, best_params,
                bounds=(lo, hi),
                max_nfev=_nfev_each,
                gtol=0,
            )
            _best_fit  = fit
            _best_cost = fit["cost"]

            if iteration > 0:
                for _scale in [0.20, 0.50, 1.00]:
                    _pert  = _rng.uniform(1.0 - _scale, 1.0 + _scale, size=len(best_params))
                    _p_try = np.clip(best_params * _pert, lo, hi)
                    _fit_r = nonlinear_least_squares(
                        residuals, _p_try,
                        bounds=(lo, hi),
                        max_nfev=_nfev_each,
                    )
                    if (_fit_r["success"] or _fit_r["cost"] < 1e3) and _fit_r["cost"] < _best_cost:
                        _best_cost = _fit_r["cost"]
                        _best_fit  = _fit_r

                logger.info(
                    "[EstimatorAgent] iter %d | multi-start (4 starts, nfev=%d each):"
                    " best cost %.6f",
                    iteration, _nfev_each, _best_cost,
                )

            if _best_fit["success"] or _best_fit["cost"] < 1e3:
                best_params = _best_fit["params"]
                best_cov    = _best_fit["covariance"]
                fit         = _best_fit

            # Convergence check: coefficient of variation
            std_params = np.sqrt(np.diag(best_cov))
            cv = std_params / (np.abs(best_params) + 1e-12)
            _mse_joint = float(np.mean(fit["residuals"] ** 2))

            # Compute per-dataset MSE so logging is not misleading: the joint MSE
            # drops as easy-to-fit datasets (steps, low-amplitude) are added, making
            # it look like improvement even when parameters haven't changed.
            _per_ds_mse = []
            for t_d, u_d, y_d, s_d in all_datasets:
                _r = _multi_shoot(simulator, best_params, t_d, u_d, y_d, s_d, seg_len)
                _per_ds_mse.append(float(np.mean(_r ** 2)))
            _current_ds_mse = _per_ds_mse[-1]  # MSE on the dataset just collected

            logger.debug(
                "Iter %d: params=%s CV=%s joint_mse=%.6f solver=%s",
                iteration, best_params.round(3), cv.round(3), _mse_joint, fit["message"],
            )
            logger.info(
                "[EstimatorAgent] iter %d | mse_current=%.6f | mse_joint=%.6f"
                " | datasets=%d | %s",
                iteration,
                _current_ds_mse,
                _mse_joint,
                len(all_datasets),
                "  ".join(f"{p}={v:.4f}" for p, v in zip(fit_params, best_params)),
            )

            if plan is None and iteration >= MIN_INNER_ITER - 1 and np.all(cv < COV_TARGET_CV):
                converged = True
                break

        # Augmented Multiple-Shooting refinement: jointly optimise parameters and
        # hidden-state ICs at every segment boundary via λ-continuation, eliminating
        # the SG velocity-bias that the old state-refinement pass only partially corrected.
        # Skip when the main-loop parameters are already stable (< 1% relative change
        # from p0) — AMS only provides marginal refinement in that case and is expensive.
        _p0_rel_change = (
            float(np.max(np.abs(best_params - p0) / (np.abs(p0) + 1e-12)))
            if len(best_params) == len(p0) else 1.0
        )
        _ams_needed = ENABLE_AMS and (_p0_rel_change > 0.01 or not converged)
        if all_datasets and system_order >= 2 and not _ams_needed:
            logger.info(
                "[EstimatorAgent] AMS skipped — %s",
                "disabled (ENABLE_AMS=False)" if not ENABLE_AMS
                else f"params stable (rel_change={_p0_rel_change:.4f} < 1% and converged)",
            )
        if all_datasets and system_order >= 2 and _ams_needed:
            logger.info(
                "[EstimatorAgent] AMS refinement on %d dataset(s) | n_params=%d | seg_len=%d"
                " | p0_rel_change=%.3f",
                len(all_datasets), len(fit_params), seg_len, _p0_rel_change,
            )
            ams_params, ams_cov = _run_ams(
                simulator_full, best_params, all_datasets,
                seg_len, lo, hi,
                output_state_index=output_state_index,
                n_states=system_order,
            )
            if ams_params is not None and not np.any(np.isnan(ams_params)):
                best_params = np.clip(ams_params, lo, hi)
                if not np.all(np.isnan(ams_cov)):
                    best_cov = ams_cov
                logger.info(
                    "[EstimatorAgent] AMS final params | %s",
                    "  ".join(f"{p}={v:.5f}" for p, v in zip(fit_params, best_params)),
                )

        # Baseline trajectory quality on the lowest-amplitude dataset.
        # Stored so the validation agent can detect when a validation probe has entered
        # a qualitatively different dynamics regime (spinning, saturation, bifurcation)
        # where absolute RMSE loses meaning and NRMSE is the better quality metric.
        _baseline_trajectory_rmse = None
        _baseline_nrmse            = None
        _baseline_y_range          = None
        _baseline_amplitude        = None
        if all_datasets and amplitudes_used and simulator is not None:
            _min_idx          = int(np.argmin(amplitudes_used[:len(all_datasets)]))
            t_b, u_b, y_b, s_b = all_datasets[_min_idx]
            _baseline_amplitude = amplitudes_used[_min_idx]
            try:
                y_sim_b = simulator(best_params, t_b, u_b, x0=s_b[:, 0])
                if (y_sim_b is not None and len(y_sim_b) == len(y_b)
                        and not np.any(np.isnan(y_sim_b))):
                    err_b = y_b - y_sim_b
                    _baseline_trajectory_rmse = float(np.sqrt(np.mean(err_b ** 2)))
                    _baseline_y_range = float(np.ptp(y_b)) or 1.0
                    _baseline_nrmse   = _baseline_trajectory_rmse / _baseline_y_range
                    logger.debug(
                        "[EstimatorAgent] baseline (amp=%.2f): rmse=%.5f nrmse=%.5f y_range=%.4f",
                        _baseline_amplitude,
                        _baseline_trajectory_rmse, _baseline_nrmse, _baseline_y_range,
                    )
            except Exception as _exc:
                logger.debug("[EstimatorAgent] baseline computation failed: %s", _exc)

        # Store fitted model — preserve the parent model's type (e.g. GREY_BOX for poly fallback)
        fitted_artifact = ModelArtifact(
            model_type=model.model_type,
            structure_description=model.structure_description,
            parameters={p: float(v) for p, v in zip(fit_params, best_params)},
            parent_id=model_id,
            metadata={
                **meta,
                "run_ids":  run_ids,
                "n_train":  sum(len(d[0]) for d in all_datasets),
                "training_context": {
                    "n_iterations":    len(all_datasets),
                    "cv_converged":    converged,
                    "stall_reason":    stall_reason,
                    "seg_len":         seg_len,
                    "amplitudes_used": amplitudes_used,
                    "methods_used":    methods_used_log,
                    "re_estimate_count": re_est,
                    "baseline_trajectory_rmse": _baseline_trajectory_rmse,
                    "baseline_nrmse":           _baseline_nrmse,
                    "baseline_y_range":         _baseline_y_range,
                    "baseline_amplitude":       _baseline_amplitude,
                },
            },
        )
        fitted_id = self._registry.store_model(fitted_artifact)

        # Store covariance
        cov_id = fitted_id + "_cov"
        self._registry.store_covariance(cov_id, best_cov)
        fitted_artifact_updated = fitted_artifact.model_copy(
            update={"parameter_covariance_id": cov_id, "id": fitted_id}
        )
        self._registry.store_model(fitted_artifact_updated)

        param_str = ", ".join(f"{p}={v:.3f}" for p, v in zip(fit_params, best_params))
        logger.info("[EstimatorAgent] ── FINAL FIT (converged=%s) ──", converged)
        for p, v in zip(fit_params, best_params):
            logger.info("[EstimatorAgent]   %s = %.6f", p, v)
        logger.info("[EstimatorAgent] ── end of fit ──")

        return Report(
            agent="EstimatorAgent",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=fitted_id, type="model", store="registry")],
            summary=(
                f"Fitted {len(fit_params)} parameters: {param_str}. "
                f"Converged={converged}."
            ),
            metadata={
                "model_id":     fitted_id,
                "covariance_id": cov_id,
                "converged":    converged,
                "params":       {p: float(v) for p, v in zip(fit_params, best_params)},
                "run_ids":      run_ids,
                "stalled_reason": stall_reason,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_contract(self, contract_id: str):
        """Load plant contract or return a sensible default."""
        from core.schemas import PlantContract
        if contract_id:
            try:
                artifact = self._registry.load_model(contract_id)
                pc_data  = artifact.metadata.get("plant_contract", {})
                if pc_data:
                    # Reconstruct PlantContract from stored dict
                    pc_data["input_limits"] = {
                        k: tuple(v) for k, v in pc_data.get("input_limits", {}).items()
                    }
                    return PlantContract(**pc_data)
            except Exception:
                pass
        return PlantContract(
            name="unknown",
            input_names=["u"],
            output_names=["y"],
            input_limits={"u": (-2.0, 2.0)},
            sample_time=0.02,
        )

    @staticmethod
    def _initial_guess(
        params:      List[str],
        bounds:      dict,
        p0_override: dict = None,
    ) -> np.ndarray:
        """
        Initial parameter guess.
        Priority: p0_override > 10% of upper bound > 1.0.
        p0_override entries (e.g. K_c warm-start from the grey-box SO) are kept
        as-is and are NOT replaced by the later OLS step.
        """
        p0 = []
        for p in params:
            if p0_override and p in p0_override:
                p0.append(float(p0_override[p]))
            elif p in bounds:
                lo, hi = bounds[p]
                p0.append(max(lo + 0.01, hi * 0.10))
            else:
                p0.append(1.0)
        return np.array(p0, dtype=float)

    @staticmethod
    def _split_bounds(params: List[str], bounds: dict):
        lo = np.array([bounds.get(p, [-np.inf, np.inf])[0] for p in params])
        hi = np.array([bounds.get(p, [-np.inf, np.inf])[1] for p in params])
        return lo, hi

    @staticmethod
    def _ols_initial_guess(
        normalized_rhs: str,
        fit_params: List[str],
        state_vars: List[str],
        input_vars: List[str],
        t: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        system_order: int = 2,
    ) -> np.ndarray:
        """
        Physics-based initial guess via OLS on numerical derivatives.

        For a model linear in parameters:
            x^(n) = sum_j  p_j * (d rhs / d p_j)(x, x_dot, ..., u)

        Builds the regressor matrix from estimated state values and solves
        via OLS — no ODE integration needed.  Works for any system order.
        """
        from tools.symbolic_math import _make_locals

        # Estimate [y, dy/dt, ..., d^(n-1)y/dt^(n-1)] and the target d^n y/dt^n
        states_est = _estimate_hidden_states(t, y, system_order)
        target = _sg_deriv(y, t, deriv=system_order)  # n-th derivative directly from y

        N = len(t)
        locs = _make_locals(fit_params, state_vars, input_vars)
        expr = sp.sympify(normalized_rhs, locals=locs)

        all_names = fit_params + state_vars + input_vars
        all_syms  = [locs[n] for n in all_names]

        A = np.zeros((N, len(fit_params)))
        for j, p in enumerate(fit_params):
            df_dp   = sp.diff(expr, locs[p])
            df_dp_f = sp.lambdify(all_syms, df_dp, modules=["numpy"])
            p_ones  = [np.ones(N)] * len(fit_params)
            sv_arrs = [states_est[k] for k in range(system_order)]
            iv_arrs = [u]
            try:
                col = df_dp_f(*(p_ones + sv_arrs + iv_arrs))
                A[:, j] = np.asarray(col, dtype=float).ravel()[:N]
            except Exception:
                A[:, j] = 0.0

        try:
            p_ols, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            p_ols = np.clip(np.abs(p_ols), 1e-3, 1e4)
        except Exception:
            p_ols = np.ones(len(fit_params))

        return p_ols


# ── Private helpers ───────────────────────────────────────────────────────────

def _sg_deriv(y: np.ndarray, t: np.ndarray, deriv: int = 1) -> np.ndarray:
    """
    Compute the deriv-th derivative of y(t) using Savitzky-Golay's built-in
    polynomial differentiation (savgol_filter's `deriv` parameter).

    This is fundamentally different from the SG-smooth-then-np.gradient pattern:
    SG fits a degree-p polynomial to each window of W points and evaluates the
    analytical derivative of that polynomial at the centre — no finite differences.

    Noise in the result: O(σ · c(W,p) / Δt^deriv) where c(W=21, p=3) ≈ 0.18,
    versus c ≈ 2.0 for simple np.gradient chains — roughly 10× quieter for
    second derivatives at W=21.

    Falls back to np.gradient chains on very short signals or SG failure.
    """
    from scipy.signal import savgol_filter

    N = len(y)
    if N < 5:
        result = y.copy()
        for _ in range(deriv):
            result = np.gradient(result, t)
        return result

    dt = float(np.mean(np.diff(t)))

    # Window must be odd, >= 2*deriv+3 (so polyorder=3 >= deriv), at most 21.
    wl = max(2 * deriv + 3, min(21, (N // 10) * 2 + 1))
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, N if N % 2 == 1 else N - 1)
    polyorder = min(3, wl - 1)

    if polyorder < deriv:
        result = y.copy()
        for _ in range(deriv):
            result = np.gradient(result, t)
        return result

    try:
        return savgol_filter(y, window_length=wl, polyorder=polyorder,
                             deriv=deriv, delta=dt)
    except Exception:
        result = y.copy()
        for _ in range(deriv):
            result = np.gradient(result, t)
        return result


def _estimate_hidden_states(
    t: np.ndarray,
    y: np.ndarray,
    system_order: int,
) -> np.ndarray:
    """
    Estimate the full state vector at every time step via Savitzky-Golay
    polynomial differentiation of the measured output.

    Returns shape (system_order, N):
      row 0 — smoothed output (y)
      row 1 — dy/dt
      row 2 — d²y/dt²   (only for order >= 3)
      …

    All derivatives are computed directly from the original y using
    _sg_deriv(y, t, deriv=i), which differentiates the locally-fitted
    polynomial analytically.  This avoids compounding noise from chaining
    np.gradient calls: each row is independent and internally consistent.
    """
    N = len(y)
    states = np.zeros((system_order, N))
    states[0] = _sg_deriv(y, t, deriv=0)   # smoothed output (deriv=0 = smooth only)
    for i in range(1, system_order):
        states[i] = _sg_deriv(y, t, deriv=i)
    return states


def _refine_segment_states(
    simulator,
    params: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    y_true: np.ndarray,
    states_est: np.ndarray,
    seg_len: int,
    output_state_index: int = 0,
    n_iter: int = 2,
    damping: float = 0.6,
) -> np.ndarray:
    """
    Correct hidden-state estimates at multi-shooting segment boundaries using
    the current model parameters.

    The SG derivative of noisy measurements is biased: for a 2nd-order system
    where only the output state is measured, the estimated velocity (state[1])
    at each segment boundary can be systematically off by ~0.02–0.05 units/s at
    large amplitudes.  The NLS then compensates for this bias by converging to
    wrong parameters.

    This function uses the model to infer the velocity correction needed at each
    boundary.  For each segment starting at index i:

      1. Simulate a short forward horizon (min(10, seg_len//2) steps) from the
         current state estimate [output_measured[i], velocity_est[i]].
      2. Compute the true ODE velocity sensitivity ∂θ(j)/∂θ̇₀ by finite-
         differencing a second simulation perturbed by +1 rad/s at t=0.  This
         replaces the prior pure-integrator approximation (≈ j·Δt) that failed
         in the spinning regime (θ̇ >> ω_n).
      3. Solve the 1-D least-squares problem using the true sensitivity to find
         delta_velocity that minimises the short-horizon output error.
      4. Apply a damped correction: velocity_est[i] += damping * delta_velocity.

    Iterating n_iter times converges to a model-consistent velocity estimate
    without requiring a full Kalman filter.

    Only corrects hidden states (state index > output_state_index).
    For 1st-order systems where all states are observed, returns states_est
    unchanged.  For order >= 3, corrects all velocity-like hidden states using
    the same short-horizon regression.

    Parameters
    ----------
    damping : float
        Correction damping factor in (0, 1).  0.6 prevents overshoot while
        still converging in 2 iterations.
    """
    system_order = states_est.shape[0]
    # Nothing to do if all states are directly measured (1st-order systems)
    # or if the simulator is unavailable.
    hidden_indices = [i for i in range(system_order) if i != output_state_index]
    if not hidden_indices or params is None or len(params) == 0:
        return states_est

    N   = len(t)
    dt  = float(t[1] - t[0]) if len(t) > 1 else 0.02
    hor = max(2, min(10, seg_len // 2))   # short forward horizon in samples
    states = states_est.copy()

    for _ in range(n_iter):
        new_states = states.copy()
        for i in range(0, N - seg_len, seg_len):
            i_end = min(i + hor, N)
            if i_end - i < 2:
                continue
            t_h = t[i:i_end]
            u_h = u[i:i_end]
            y_h = y_true[i:i_end]
            x0  = states[:, i].copy()

            y_pred = simulator(params, t_h, u_h, x0=x0)
            if y_pred is None or np.any(np.isnan(y_pred)) or len(y_pred) < 2:
                continue

            # Short-horizon output prediction error
            n_h  = min(len(y_pred), len(y_h)) - 1
            if n_h < 1:
                continue
            err = y_h[1:n_h + 1] - y_pred[1:n_h + 1]

            # True ODE sensitivity via finite difference: ∂θ(j)/∂θ̇₀.
            # Replaces the pure-integrator approximation ≈ j·Δt, which breaks
            # down in the spinning regime (θ̇ >> ω_n) and for strongly nonlinear
            # trajectories.  fd_delta cancels in the LS ratio so its exact value
            # only affects numerical noise — 1 rad/s is well above machine epsilon.
            fd_delta = 1.0
            x0_pert = x0.copy()
            x0_pert[hidden_indices[0]] += fd_delta
            y_pert = simulator(params, t_h, u_h, x0=x0_pert)
            if y_pert is not None and not np.any(np.isnan(y_pert)) and len(y_pert) >= n_h + 1:
                sensitivity = (y_pert[1:n_h + 1] - y_pred[1:n_h + 1]) / fd_delta
                denom = float(np.dot(sensitivity, sensitivity))
                if denom < 1e-12:
                    continue   # model insensitive to velocity here — skip
                delta_v = float(np.dot(sensitivity, err)) / denom
            else:
                # Perturbed sim failed; fall back to integrator approximation
                times = np.arange(1, n_h + 1, dtype=float) * dt
                denom = float(np.dot(times, times))
                if denom < 1e-12:
                    continue
                delta_v = float(np.dot(times, err)) / denom

            # Clamp to a physically plausible range (≤ 10 units/s correction)
            delta_v = float(np.clip(delta_v, -10.0, 10.0))

            # Apply damped correction to all hidden velocity states.
            # For 2nd-order systems this is only state[1]; for 3rd-order
            # systems state[2] (acceleration) is corrected with half the gain
            # since it has a quadratic, not linear, effect on output.
            for k, idx in enumerate(hidden_indices):
                gain = damping / (2.0 ** k)   # halve gain for each higher derivative
                new_states[idx, i] += gain * delta_v

        states = new_states

    return states


def _run_ams(
    simulator_full,
    p0_params: np.ndarray,
    all_datasets: list,
    seg_len: int,
    lo: np.ndarray,
    hi: np.ndarray,
    output_state_index: int = 0,
    n_states: int = 2,
    lambda_schedule: Optional[list] = None,
) -> tuple:
    """
    Augmented Multiple-Shooting (AMS) refinement.

    Extends the variable vector to include hidden-state initial conditions at
    every segment boundary alongside the physics parameters.  A λ-continuation
    schedule progressively tightens the inter-segment continuity constraint,
    starting loose (λ=0.1, segments decouple → easy gradient landscape) and
    finishing tight (λ=10, strong continuity → equivalent to single-shooting).

    Variable vector layout
    ----------------------
    p_aug = [ params (n_p) | v̂_{k,d} for each dataset d, segment k ]

    Residual structure
    ------------------
    r = [ fit: θ_sim − θ_meas  (per segment, all datasets)
        | continuity: λ · (θ̇_sim_end_k − v̂_{k+1}) / v_scale  (between segments) ]

    Continuity residuals are normalised by v_scale (≈ peak angular velocity in
    the data) so they remain dimensionless and comparable to the angle residuals.

    Returns (best_params, param_covariance_block).
    Falls back to (p0_params, NaN) on failure or for 1st-order systems.
    """
    if lambda_schedule is None:
        lambda_schedule = [0.1, 1.0, 10.0]

    n_params       = len(p0_params)
    hidden_indices = [i for i in range(n_states) if i != output_state_index]
    n_hidden       = len(hidden_indices)

    if n_hidden == 0:
        return p0_params.copy(), np.full((n_params, n_params), np.nan)

    # ── Build segment starts and IC warm-starts from SG estimates ────────────
    all_seg_starts: list = []
    ic_warmstart:   list = []
    v_abs_max = 1.0

    for t_d, u_d, y_d, states_est_d in all_datasets:
        N_d          = len(t_d)
        seg_starts_d = np.arange(0, N_d - seg_len, seg_len, dtype=int)
        if len(seg_starts_d) == 0:
            seg_starts_d = np.array([0], dtype=int)
        all_seg_starts.append(seg_starts_d)
        for i0 in seg_starts_d:
            for idx in hidden_indices:
                v = float(states_est_d[idx, i0])
                ic_warmstart.append(v)
                v_abs_max = max(v_abs_max, abs(v))

    n_total_hidden = len(ic_warmstart)
    v_scale        = max(v_abs_max, 1.0)
    v_bound        = v_scale * 3.0

    lo_aug = np.concatenate([lo, np.full(n_total_hidden, -v_bound)])
    hi_aug = np.concatenate([hi, np.full(n_total_hidden,  v_bound)])

    # ── Residual closure ──────────────────────────────────────────────────────
    def make_residuals(lambda_c: float):
        def residuals(p_aug: np.ndarray) -> np.ndarray:
            params     = p_aug[:n_params]
            hidden_all = p_aug[n_params:]
            fit_res:  list = []
            cont_res: list = []
            offset = 0

            for d_idx, (t_d, u_d, y_d, _) in enumerate(all_datasets):
                seg_starts_d = all_seg_starts[d_idx]
                M_d  = len(seg_starts_d)
                n_ic = M_d * n_hidden
                h_d  = hidden_all[offset: offset + n_ic]
                offset += n_ic
                N_d  = len(t_d)

                for k, i0 in enumerate(seg_starts_d):
                    i0  = int(i0)
                    i1  = int(seg_starts_d[k + 1]) if k + 1 < M_d else N_d
                    t_s = t_d[i0:i1]
                    u_s = u_d[i0:i1]
                    y_s = y_d[i0:i1]
                    n_s = i1 - i0

                    x0 = np.zeros(n_states)
                    x0[output_state_index] = float(y_d[i0])
                    for j, idx in enumerate(hidden_indices):
                        x0[idx] = float(h_d[k * n_hidden + j])

                    Y = simulator_full(params, t_s, u_s, x0=x0)

                    if (Y is None or Y.ndim < 2
                            or Y.shape[1] < n_s or np.any(np.isnan(Y))):
                        fit_res.extend([1e3] * n_s)
                        if k + 1 < M_d:
                            cont_res.extend([1e3 * lambda_c] * n_hidden)
                        continue

                    fit_res.extend((Y[output_state_index, :n_s] - y_s).tolist())

                    if k + 1 < M_d:
                        for j, idx in enumerate(hidden_indices):
                            end_val  = float(Y[idx, -1])
                            next_ic  = float(h_d[(k + 1) * n_hidden + j])
                            cont_res.append(lambda_c * (end_val - next_ic) / v_scale)

            all_r = fit_res + cont_res
            return np.array(all_r) if all_r else np.array([0.0])
        return residuals

    # ── λ-continuation ────────────────────────────────────────────────────────
    p_aug = np.clip(
        np.concatenate([p0_params, np.array(ic_warmstart, dtype=float)]),
        lo_aug, hi_aug,
    )
    best_params = p0_params.copy()
    best_cov    = np.full((n_params, n_params), np.nan)
    best_cost   = np.inf
    n_aug       = len(p_aug)
    nfev        = int(np.clip(200 * n_aug, 2000, 8000))

    for lam in lambda_schedule:
        fit = nonlinear_least_squares(
            make_residuals(lam), p_aug, bounds=(lo_aug, hi_aug), max_nfev=nfev,
        )
        if fit["success"] or fit["cost"] < best_cost:
            p_aug     = fit["params"]
            best_cost = fit["cost"]
            best_params = p_aug[:n_params]
            full_cov = fit["covariance"]
            if full_cov is not None and full_cov.shape[0] >= n_params:
                block = full_cov[:n_params, :n_params]
                if not np.all(np.isnan(block)):
                    best_cov = block
            logger.info(
                "[AMS] λ=%.1f  cost=%.6f | %s",
                lam, best_cost,
                "  ".join(f"{v:.5f}" for v in best_params),
            )
        else:
            logger.info(
                "[AMS] λ=%.1f did not improve (cost=%.6f ≥ best=%.6f) — keeping previous",
                lam, fit["cost"], best_cost,
            )

    return best_params, best_cov


def _multi_shoot(
    simulator,
    params: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    y_true: np.ndarray,
    states_est: np.ndarray,
    seg_len: int = 25,
) -> np.ndarray:
    """
    Multi-shooting residuals: reinitialized at every segment boundary.

    Prevents trajectory divergence for misspecified or nonlinear models by
    integrating only over short windows.  Each segment is initialized from
    the estimated full state vector at that boundary.

    Parameters
    ----------
    states_est : ndarray, shape (system_order, N)
        Full state estimates from ``_estimate_hidden_states``.
    """
    all_res = []
    N = len(t)
    for i in range(0, N - seg_len, seg_len):
        t_s  = t[i:i + seg_len]
        u_s  = u[i:i + seg_len]
        y_s  = y_true[i:i + seg_len]
        x0_s = states_est[:, i]          # all state estimates at segment start
        y_p  = simulator(params, t_s, u_s, x0=x0_s)
        if np.any(np.isnan(y_p)):
            all_res.extend([1e6] * seg_len)
        else:
            all_res.extend((y_p - y_s).tolist())
    return np.array(all_res) if all_res else np.array([0.0])


# ── Private ───────────────────────────────────────────────────────────────────

def _tighten_bounds(fit_params: List[str], param_bounds: dict, prior_runs: list) -> dict:
    """
    Narrow param_bounds using fitted values from similar prior runs.

    For each parameter with prior data, set bounds to [mean/5, mean*5]
    intersected with the original bounds.  This is a 10x reduction in
    search space while remaining safe if the prior is off by a factor of 5.
    """
    from collections import defaultdict
    priors: dict = defaultdict(list)
    for run in prior_runs:
        for p, v in run.fitted_params.items():
            if p in fit_params and v > 0:
                priors[p].append(v)

    new_bounds = dict(param_bounds)
    for p in fit_params:
        if not priors[p]:
            continue
        mean_val = float(np.mean(priors[p]))
        orig_lo, orig_hi = param_bounds.get(p, (-np.inf, np.inf))
        tighter_lo = orig_lo                    # never raise the lower bound — priors may
        tighter_hi = min(orig_hi, mean_val * 5.0)  # be wrong (e.g. K_d=0 when b_v=0)
        if tighter_lo < tighter_hi:
            new_bounds[p] = [tighter_lo, tighter_hi]
            logger.debug("  %s: [%.3f, %.3f] → [%.3f, %.3f] (prior mean=%.3f)",
                         p, orig_lo, orig_hi, tighter_lo, tighter_hi, mean_val)
    return new_bounds


def _extract_guidance(dossier: Dossier) -> dict:
    """Pull improvement hints from the last validation verdict and experiment plan."""
    guidance: dict = {"re_estimate_count": dossier.re_estimate_count}

    # Pass the experiment plan through so run() can use it directly.
    if dossier.experiment_plan is not None:
        guidance["experiment_plan"] = dossier.experiment_plan

    if not dossier.last_verdict:
        return guidance
    v = dossier.last_verdict
    m = v.metrics
    wi = v.worst_case_inputs or {}
    guidance["worst_case_amplitude"] = (
        m.get("worst_case_amplitude_fraction")
        or wi.get("amplitude_fraction")
        or 0.85
    )
    guidance["worst_case_scenario"] = (
        m.get("worst_case_scenario")
        or wi.get("scenario_type")
        or "near_saturation"
    )
    guidance["failure_hypothesis"] = v.failure_hypothesis or ""
    guidance["max_feature_correlation"] = m.get("max_feature_correlation", 0.0)
    return guidance


def _fail(dossier: Dossier, msg: str) -> Dossier:
    return dossier.update(
        status=f"estimator failed: {msg}",
        last_report=Report(
            agent="EstimatorAgent",
            status=AgentStatus.FAILED,
            summary=msg,
        ),
    )
