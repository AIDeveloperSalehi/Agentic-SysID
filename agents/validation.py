"""
Validation agent — LLM-driven adversarial probing and verdict generation.

The agent runs an iterative probing loop:
  1. Inspect model metadata.
  2. Run broad adversarial scenarios (low_freq_sine, near_saturation, broadband_chirp).
  3. Inspect residual summaries — top correlated features, whiteness, RMSE.
  4. Design targeted follow-up probes to isolate the failure mode.
  5. Post a structured Verdict with a failure_hypothesis and worst_case_inputs.

Gap-type routing (enforced deterministically as fallback if LLM omits it):
  RMSE < tol AND residuals white                  → gap_type = NONE  (PASS)
  Non-white AND max feature correlation > 0.20    → gap_type = STRUCTURED_RESIDUAL
  Non-white AND correlation with input > 0.15     → gap_type = FIXABLE
  Otherwise                                       → gap_type = UNMODELABLE
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    AttemptEntry,
    Critique,
    Dossier,
    GapType,
    Report,
    SplitFlag,
    ValidityRegion,
    Verdict,
    VerdictResult,
)
from agents.base_agent import BaseAgent
from agents.experiment_design import ExperimentDesignAgent
from tools.model_registry import ModelRegistry
from tools.experiment_db import ExperimentDatabase
from tools.plant_api import PlantAPI
from tools.feature_library import FeatureLibrary
from tools.solver_toolkit import (
    compute_metrics,
    generate_chirp,
    generate_multisine,
    generate_prbs,
    generate_steps,
    residual_correlation_with_input,
    residual_whiteness_test,
)
from tools.symbolic_math import make_ode_simulator
from agents.estimator import _estimate_hidden_states, _multi_shoot

logger = logging.getLogger(__name__)

PROMPT_PATH    = Path(__file__).parent.parent / "prompts" / "validation.md"


def _top_feature_in_model(probe_results: List[dict], normalized_rhs: str) -> bool:
    """True if the top residual-correlated feature is already present in the model's RHS string."""
    if not probe_results or not normalized_rhs:
        return False
    worst = max(probe_results, key=lambda r: r.get("max_feature_correlation", 0.0))
    top   = worst.get("top_correlated_features", [])
    if not top:
        return False
    name = top[0]["feature"] if isinstance(top[0], dict) else str(top[0])
    return name in normalized_rhs


def _probe_y_range(probe: dict) -> float:
    """Peak-to-peak output range seen during this probe: y_range = rmse / nrmse."""
    rmse  = probe.get("rmse",  float("inf"))
    nrmse = probe.get("nrmse", float("inf"))
    return rmse / nrmse if nrmse > 1e-10 else float("inf")


def _compute_parameter_rmse_floor(
    simulator,
    fitted_p:  np.ndarray,
    param_cov: Optional[np.ndarray],
    t_data:    np.ndarray,
    u_flat:    np.ndarray,
    y_true:    np.ndarray,
    states_est: np.ndarray,
    seg_len:   int = 25,
    n_mc:      int = 10,
    seed:      int = 0,
) -> float:
    """
    Monte Carlo estimate of the RMSE floor attributable to parameter uncertainty.

    Draws n_mc parameter vectors from N(θ*, Σ_θ) and for each computes
    multi-shoot RMSE between the baseline model and the perturbed model on the
    same plant trajectory.  The 90th-percentile of those RMSE values is the floor:
    the prediction error any parametric model within the covariance ellipsoid would
    produce purely from parameter uncertainty — including trajectory-divergence
    effects that accumulate in sensitive operating regimes.

    This is fully system-agnostic: it uses only the model's own uncertainty
    (the NLS covariance) and requires no knowledge of the system physics.

    Returns 0.0 when covariance is unavailable or all perturbed simulations fail.
    """
    if param_cov is None or fitted_p is None or len(fitted_p) == 0:
        return 0.0
    if np.any(~np.isfinite(param_cov)):
        return 0.0

    # Use a short window to keep compute affordable (≤150 samples).
    n_use = min(len(t_data), 150)
    t_use, u_use, y_use = t_data[:n_use], u_flat[:n_use], y_true[:n_use]
    s_use = states_est[:, :n_use]

    # Baseline multi-shoot prediction on this window
    ms_base = _multi_shoot(simulator, fitted_p, t_use, u_use, y_use, s_use, seg_len)
    n_ms = len(ms_base)
    if n_ms == 0:
        return 0.0
    y_pred_base = y_use[:n_ms] - ms_base

    # Draw MC samples from N(θ*, Σ_θ)
    rng = np.random.default_rng(seed)
    try:
        diag_max = float(np.max(np.abs(np.diag(param_cov))))
        reg      = max(diag_max * 1e-8, 1e-14)
        L        = np.linalg.cholesky(param_cov + np.eye(len(fitted_p)) * reg)
        samples  = fitted_p + (L @ rng.standard_normal((len(fitted_p), n_mc))).T
    except np.linalg.LinAlgError:
        std     = np.sqrt(np.maximum(np.diag(param_cov), 0.0))
        samples = fitted_p + rng.standard_normal((n_mc, len(fitted_p))) * std

    rmse_vals = []
    for p_pert in samples:
        try:
            ms_pert = _multi_shoot(simulator, p_pert, t_use, u_use, y_use, s_use, seg_len)
            if len(ms_pert) < n_ms:
                continue
            y_pred_pert = y_use[:n_ms] - ms_pert[:n_ms]
            diff        = y_pred_base - y_pred_pert
            val         = float(np.sqrt(np.mean(diff ** 2)))
            if np.isfinite(val):
                rmse_vals.append(val)
        except Exception:
            continue

    return float(np.percentile(rmse_vals, 90)) if rmse_vals else 0.0


def compute_tool_reliability_floor(
    simulator,
    fitted_p:   np.ndarray,
    model_meta: dict,
    contract,
    amplitude_fraction: float,
    n_samples: int = 150,
    seg_len:   int = 25,
    seed:      int = 0,
) -> float:
    """
    Estimate the RMSE the multi-shoot tool would report for a PERFECT model at
    this amplitude, purely from hidden-state estimation errors.

    The multi-shoot pipeline estimates hidden states (e.g. θ̇) from the measured
    output (e.g. θ) using Savitzky-Golay differentiation, then uses those as
    initial conditions for each integration segment.  Any error in those estimates
    propagates through the ODE and inflates RMSE even when the model is correct.
    This function quantifies that inflation for the current model at a given amplitude.

    Method (no plant calls needed):
      1. Simulate the model at the given amplitude → clean trajectory y_clean.
      2. Run _estimate_hidden_states on y_clean (same pipeline as real validation).
         This introduces SG-differentiation errors in the hidden states.
      3. Run _multi_shoot against y_clean using those imperfect estimated states.
      4. The resulting RMSE has zero model error by construction — it comes solely
         from hidden-state estimation inaccuracy at segment boundaries.

    Works for any model whose simulator callable accepts (params, t, u, x0).
    Returns 0.0 for fully-observed systems (no hidden states) or when simulation
    fails.
    """
    system_order = model_meta.get("system_order", 2)
    n_hidden = system_order - 1  # states beyond the measured output
    if n_hidden <= 0:
        return 0.0  # all states observed — no estimation error

    input_name = contract.input_names[0]
    lo, hi = contract.input_limits.get(input_name, (-1.0, 1.0))
    half_range = min(abs(lo), abs(hi))
    amplitude = half_range * float(np.clip(amplitude_fraction, 0.05, 1.0))
    dt = contract.sample_time
    t  = np.arange(n_samples) * dt

    # Use a slow sine wave rather than PRBS so the floor reflects a signal that
    # SG differentiation can handle accurately.  PRBS has near-Nyquist content
    # that makes SG differentiation maximally inaccurate — it would give a floor
    # 6-10× higher than smooth probes actually experience.
    #
    # Frequency choice: one full cycle spanning the entire recording window.
    # freq = 1 / T_total = 1 / (n_samples * dt)
    # This is derived purely from the timing parameters (no system-specific Hz
    # values), gives SG the maximum number of samples per period, and works for
    # any model or plant.
    T_total  = n_samples * dt
    freq_hz  = 1.0 / T_total
    u        = amplitude * np.sin(2 * np.pi * freq_hz * t)

    x0 = None
    if contract.x0 is not None:
        x0 = np.array(contract.x0[:system_order], dtype=float)

    # fitted_p may be empty for corrected/surrogate models whose parameters are
    # embedded in the simulator closure — that is fine, we still need the simulator.
    p = fitted_p if (fitted_p is not None and len(fitted_p) > 0) else np.array([])

    try:
        y_clean = simulator(p, t, u, x0=x0)
    except Exception:
        return 0.0

    if y_clean is None or np.any(np.isnan(y_clean)) or len(y_clean) < seg_len * 2:
        return 0.0

    # Estimate hidden states from the clean trajectory — same pipeline as validation.
    # This captures the SG-differentiation approximation error in hidden states.
    states_est = _estimate_hidden_states(t, y_clean, system_order)
    if contract.x0 is not None:
        for k, v in enumerate(contract.x0[:system_order]):
            states_est[k, 0] = float(v)

    # Multi-shoot using imperfect estimated states against the clean trajectory.
    # Residuals are purely from state-estimation errors at segment boundaries.
    try:
        residuals = _multi_shoot(simulator, p, t, u, y_clean, states_est, seg_len)
    except Exception:
        return 0.0

    if len(residuals) == 0:
        return 0.0

    floor = float(np.sqrt(np.mean(residuals ** 2)))
    return floor if np.isfinite(floor) else 0.0


def compute_reliability_sweep(
    simulator,
    fitted_p:   np.ndarray,
    model_meta: dict,
    contract,
    rmse_tol:   float,
    amplitude_fractions: tuple = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_samples:  int = 150,
    seg_len:    int = 25,
) -> dict:
    """
    Compute the tool reliability floor at a range of amplitudes and determine
    the reliability ceiling — the highest amplitude where the multi-shoot tool
    can give meaningful RMSE quality measurements.

    The ceiling is defined as the highest amplitude where:
        tool_floor < rmse_tol * 0.5
    meaning at least half the tolerance budget remains for actual model error.

    Returns a dict with:
        sweep:             list of {amplitude_fraction, tool_floor, tool_reliable}
        reliability_ceiling: highest amplitude fraction where tool_reliable=True (0 if none)
        floor_at:          dict mapping amplitude_fraction → tool_floor (for fast lookup)
    """
    sweep = []
    ceiling = 0.0
    for amp in amplitude_fractions:
        floor = compute_tool_reliability_floor(
            simulator, fitted_p, model_meta, contract, amp,
            n_samples=n_samples, seg_len=seg_len,
        )
        reliable = floor < rmse_tol * 0.5
        sweep.append({
            "amplitude_fraction": round(float(amp), 2),
            "tool_floor":         round(floor, 5),
            "tool_reliable":      reliable,
        })
        if reliable:
            ceiling = float(amp)

    return {
        "sweep":               sweep,
        "reliability_ceiling": ceiling,
        "floor_at": {round(float(s["amplitude_fraction"]), 2): s["tool_floor"] for s in sweep},
    }


def _probe_passes_regime_aware(
    probe: dict,
    rmse_tol: float,
    baseline_y_range: Optional[float] = None,
    baseline_nrmse:   Optional[float] = None,
) -> bool:
    """
    True if this probe meets the quality threshold, using a hierarchy of criteria:

    0. Tool reliability gate (highest priority):
       If tool_reliable=False the multi-shoot tool itself produces errors exceeding
       rmse_tol*0.5 even for a perfect model.  RMSE cannot distinguish model error
       from tool artifact here — treat the probe as neutral (does not fail the model).

    1. Covariance-based excess RMSE (primary, system-agnostic):
       excess_rmse = rmse - rmse_floor, where rmse_floor is the MC estimate of
       RMSE attributable to parameter uncertainty alone.  If excess_rmse < tol,
       the observed error is fully explained by parameter uncertainty — pass.

    2. Raw RMSE fallback (when covariance floor was not computed):
       Standard rmse < tol criterion.

    3. Regime-aware NRMSE (insurance layer):
       When probe output range >> training baseline, the system is in a sensitive
       regime where trajectory divergence dominates even the floor estimate.
       NRMSE < max(3 × baseline_nrmse, 0.015) is used as the criterion.
    """
    # 0. Tool reliability gate: probe is above the reliability ceiling — RMSE is not
    # a meaningful quality signal here; treat as neutral so it doesn't fail the model.
    if probe.get("tool_reliable") is False:
        return True

    # 1. Covariance-based floor: primary criterion
    excess = probe.get("excess_rmse")
    if excess is not None:
        if excess < rmse_tol:
            return True
    else:
        # 2. Raw RMSE when no floor is available
        if probe.get("rmse", float("inf")) < rmse_tol:
            return True

    # 3. Regime-aware NRMSE insurance layer
    if baseline_y_range and baseline_nrmse:
        probe_yr = _probe_y_range(probe)
        if probe_yr > 4.0 * baseline_y_range:
            nrmse_tol = max(3.0 * baseline_nrmse, 0.015)
            return bool(probe.get("nrmse", float("inf")) < nrmse_tol)

    return False


def _regime_adjusted_worst_rmse(
    probe_results:    List[dict],
    rmse_tol:         float,
    baseline_y_range: Optional[float] = None,
    baseline_nrmse:   Optional[float] = None,
) -> float:
    """
    Worst quality metric across all probes, using regime-aware pass/fail.

    Probes that pass are capped at rmse_tol×0.99.
    For failing probes, uses excess_rmse when available (the parameter-error
    component of RMSE, stripped of trajectory-divergence noise); falls back
    to raw rmse otherwise.
    """
    if not probe_results:
        return float("inf")
    adjusted = []
    for r in probe_results:
        if _probe_passes_regime_aware(r, rmse_tol, baseline_y_range, baseline_nrmse):
            adjusted.append(rmse_tol * 0.99)
        else:
            err = r.get("excess_rmse") if r.get("excess_rmse") is not None else r.get("rmse", float("inf"))
            adjusted.append(float(err))
    return float(max(adjusted)) if adjusted else float("inf")


def _amplitude_gap_detected(
    probe_results:    List[dict],
    rmse_tol:         float,
    baseline_y_range: Optional[float] = None,
    baseline_nrmse:   Optional[float] = None,
) -> bool:
    """
    Return True when there is clear amplitude-dependent failure:
    at least one low-amplitude probe PASSES and at least one high-amplitude probe FAILS,
    with the lowest passing amplitude strictly below the lowest failing amplitude.

    This pattern proves the model structure is correct — a genuinely missing term
    would fail at all amplitudes, not just large ones.

    Uses regime-aware pass/fail when baseline_y_range / baseline_nrmse are provided:
    spinning-regime probes are evaluated on NRMSE so they don't mask the amplitude gap.
    """
    if not probe_results:
        return False
    passing = [r for r in probe_results
               if _probe_passes_regime_aware(r, rmse_tol, baseline_y_range, baseline_nrmse)]
    failing  = [r for r in probe_results
                if not _probe_passes_regime_aware(r, rmse_tol, baseline_y_range, baseline_nrmse)]
    if not passing or not failing:
        return False
    max_passing_amp = max(r.get("amplitude_fraction", 0.0) for r in passing)
    min_failing_amp = min(r.get("amplitude_fraction", 1.0) for r in failing)
    return max_passing_amp < min_failing_amp
RMSE_TOLERANCE = 0.05
FEAT_CORR_THR  = 0.20
INPUT_CORR_THR = 0.15

# ── Tool schemas for the LLM probe loop ──────────────────────────────────────

_RUN_SCENARIO_SCHEMA: dict = {
    "name": "run_scenario",
    "description": (
        "Run a single adversarial probe on the real plant and simulate the fitted model. "
        "Returns residual statistics and feature correlations — no raw arrays. "
        "Call this multiple times with different scenario types and amplitudes to isolate failures."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scenario_type": {
                "type": "string",
                "enum": [
                    "low_freq_sine",
                    "near_saturation",
                    "broadband_chirp",
                    "slow_sine",
                    "prbs",
                    "multisine",
                    "step_sequence",
                ],
                "description": (
                    "Input type to generate: "
                    "low_freq_sine=one slow cycle (exercises zero-crossings/Coulomb); "
                    "near_saturation=PRBS near input limit (large-signal nonlinear); "
                    "broadband_chirp=frequency sweep (dynamics coverage); "
                    "slow_sine=2 slow cycles (more zero-crossing stress); "
                    "prbs=general PRBS with controllable amplitude; "
                    "multisine=multi-frequency sine with controllable band; "
                    "step_sequence=staircase (static nonlinearity/hysteresis)."
                ),
            },
            "amplitude_fraction": {
                "type": "number",
                "description": (
                    "Fraction of plant input limit to use (0.1–1.0). "
                    "Default 0.7. Use >0.85 to stress large-signal regime, "
                    "<0.4 to probe small-signal / near-equilibrium."
                ),
            },
            "n_samples": {
                "type": "integer",
                "description": "Number of time samples (default 300 ≈ 6 s at dt=0.02).",
            },
            "seed": {
                "type": "integer",
                "description": "Random seed. Vary to get different realisations of stochastic inputs.",
            },
        },
        "required": ["scenario_type"],
    },
}

_GET_MODEL_METADATA_SCHEMA: dict = {
    "name": "get_model_metadata",
    "description": "Retrieve the fitted model structure, parameters, training RMSE, and RMSE tolerance.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

_POST_VERDICT_SCHEMA: dict = {
    "name": "post_verdict",
    "description": (
        "Record the final structured verdict. Call this BEFORE post_report "
        "once you have enough evidence from probe runs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": (
                    "pass if model meets quality threshold; fail otherwise. "
                    "Use excess_rmse (not raw rmse) for the decision when available: "
                    "excess_rmse < tolerance means the observed error is explained by "
                    "parameter uncertainty alone — the model is as good as it can be."
                ),
            },
            "gap_type": {
                "type": "string",
                "enum": ["none", "fixable", "structured_residual", "unmodelable"],
                "description": (
                    "none=pass; "
                    "fixable=dominant features ARE ALREADY IN THE MODEL (wrong coefficients, "
                    "re-estimate with better data — do NOT add new terms); "
                    "structured_residual=dominant features are NEW functions NOT in the model "
                    "(missing physics term, grey-box correction needed); "
                    "unmodelable=large unstructured error with no dominant feature."
                ),
            },
            "failure_hypothesis": {
                "type": "string",
                "description": (
                    "3-5 sentences: what conditions trigger failure, what residual pattern you observed, "
                    "what physical phenomenon is likely missing, and what the next agent should try. "
                    "Be specific — include feature names, amplitude ranges, scenario types."
                ),
            },
            "worst_case_scenario": {
                "type": "string",
                "description": "The scenario_type that produced the highest RMSE.",
            },
            "worst_case_amplitude_fraction": {
                "type": "number",
                "description": "The amplitude_fraction used in the worst-case probe.",
            },
            "amplitude_dependent_failure": {
                "type": "boolean",
                "description": (
                    "True when the failure is amplitude-dependent: at least one LOW-amplitude "
                    "probe PASSES but at least one HIGH-amplitude probe FAILS. "
                    "This means wrong coefficient (fixable), NOT missing structure. "
                    "False when the model fails equally at all amplitude levels."
                ),
            },
            "best_amplitude_tier": {
                "type": "number",
                "description": (
                    "The highest amplitude_fraction that PASSED (RMSE < tolerance). "
                    "Omit or set to 0 if every probe fails."
                ),
            },
            "regime_boundary_amplitude": {
                "type": "number",
                "description": (
                    "The lowest probe amplitude_fraction where the output range was >> "
                    "the training baseline (indicating the plant entered spinning/saturation "
                    "regime). Omit if not detected."
                ),
            },
            "excess_rmse": {
                "type": "number",
                "description": (
                    "The worst excess_rmse across all probes: max(0, rmse - rmse_floor). "
                    "This is the parameter-error component of RMSE, stripped of trajectory-"
                    "divergence noise from the model's own parameter uncertainty. "
                    "Use this for the pass/fail decision when it is available (non-null)."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of verdict and gap_type choice (1-2 sentences).",
            },
        },
        "required": ["verdict", "gap_type", "reasoning"],
    },
}


# ── Inner LLM probe loop ──────────────────────────────────────────────────────

class _LLMProbeLoop(BaseAgent):
    """
    LLM agent that iteratively designs and runs adversarial probes.
    Stores probe results and verdict data as instance state so the calling
    ValidationAgent can extract them after the loop completes.
    """

    MAX_PROBE_ITERATIONS = 12

    def __init__(
        self,
        simulator,
        fitted_p:  np.ndarray,
        plant_api: PlantAPI,
        db:        ExperimentDatabase,
        designer:  ExperimentDesignAgent,
        contract,
        rmse_tol:  float,
        model_meta: dict,
        model_id:  str,
        feat_lib:  FeatureLibrary,
        param_cov: Optional[np.ndarray] = None,
        tool_reliability: Optional[dict] = None,
        **base_kwargs,
    ):
        super().__init__(max_iterations=self.MAX_PROBE_ITERATIONS, **base_kwargs)
        self._simulator       = simulator
        self._fitted_p        = fitted_p
        self._api             = plant_api
        self._db              = db
        self._designer        = designer
        self._contract        = contract
        self._rmse_tol        = rmse_tol
        self._model_meta      = model_meta
        self._model_id        = model_id
        self._feat_lib        = feat_lib
        self._param_cov       = param_cov
        self._tool_reliability = tool_reliability or {}
        self.probe_results: List[dict] = []
        self.verdict_data:  dict       = {}

    def get_tools(self) -> List[dict]:
        return [_RUN_SCENARIO_SCHEMA, _GET_MODEL_METADATA_SCHEMA, _POST_VERDICT_SCHEMA]

    def call_tool(self, name: str, inputs: dict) -> str:
        if name == "run_scenario":
            return self._run_scenario(**inputs)
        if name == "get_model_metadata":
            return self._get_model_metadata()
        if name == "post_verdict":
            self.verdict_data = inputs
            return "Verdict recorded. Now call post_report to finish."
        return super().call_tool(name, inputs)

    def run_probe_loop(self, system_prompt: str, task_msg: str) -> Report:
        return self._run(system_prompt, task_msg)

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_model_metadata(self) -> str:
        meta = self._model_meta
        params = meta.get("fit_params", [])
        return json.dumps({
            "model_id":         self._model_id,
            "model_type":       meta.get("model_class", meta.get("normalized_rhs", "unknown")),
            "normalized_rhs":   meta.get("normalized_rhs", ""),
            "fit_params":       params,
            "state_vars":       meta.get("state_vars", []),
            "input_vars":       meta.get("input_vars", []),
            "train_rmse":       meta.get("train_rmse", None),
            "rmse_tolerance":   self._rmse_tol,
            "system_order":     meta.get("system_order", 2),
            "n_train":          meta.get("n_train", None),
            "training_context": meta.get("training_context", {}),
        })

    def _run_scenario(
        self,
        scenario_type: str,
        amplitude_fraction: float = 0.7,
        n_samples: int = 300,
        seed: int = 0,
    ) -> str:
        dt          = self._contract.sample_time
        input_name  = self._contract.input_names[0]
        lo, hi      = self._contract.input_limits.get(input_name, (-1.0, 1.0))
        half_range  = min(abs(lo), abs(hi))
        amplitude   = half_range * float(np.clip(amplitude_fraction, 0.05, 1.0))
        T_total     = n_samples * dt
        nyquist     = 0.5 / dt

        try:
            t, u = self._design_input(
                scenario_type, amplitude, n_samples, dt, T_total, nyquist, lo, hi, seed
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        u_func = self._designer.make_u_func(t, u)
        try:
            result = self._api.apply_input(
                u_func=u_func,
                t_span=(float(t[0]), float(t[-1])),
                dt=float(t[1] - t[0]),
                purpose="validation",
                input_type=scenario_type,
                agent="validation",
                split_flag=SplitFlag.VALIDATION,
            )
        except Exception as exc:
            return json.dumps({"error": f"Plant application failed: {exc}"})

        t_data, u_data, y_data = self._db.load_arrays(result["run_id"])
        y_true  = y_data[0]
        u_flat  = u_data[0]

        state_v      = self._model_meta.get("state_vars", [])
        input_v      = self._model_meta.get("input_vars", [])
        system_order = self._model_meta.get("system_order", 2)

        states_est  = _estimate_hidden_states(t_data, y_true, system_order)
        ms_residuals = _multi_shoot(
            self._simulator, self._fitted_p, t_data, u_flat, y_true, states_est, seg_len=25
        )
        n_ms   = len(ms_residuals)
        y_pred = y_true[:n_ms] - ms_residuals

        m  = compute_metrics(y_true[:n_ms], y_pred)
        wb = residual_whiteness_test(ms_residuals)
        xc = residual_correlation_with_input(ms_residuals, u_flat[:n_ms])

        feat_corr_map: Dict[str, float] = {}
        if state_v:
            try:
                _Theta, _names = self._feat_lib.build(
                    states_est[:, :n_ms], u_flat[:n_ms], state_v, input_v
                )
                feat_corr_map = self._feat_lib.feature_correlations(_Theta, _names, ms_residuals)
            except Exception:
                pass

        max_feat_corr = max(feat_corr_map.values(), default=0.0)
        top_features  = sorted(feat_corr_map.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # Covariance-based RMSE floor: the RMSE attributable to parameter uncertainty alone.
        # System-agnostic — uses only the NLS covariance from training, no physics knowledge.
        rmse_floor  = _compute_parameter_rmse_floor(
            self._simulator, self._fitted_p, self._param_cov,
            t_data, u_flat, y_true, states_est, seg_len=25, n_mc=10,
        )
        raw_rmse    = float(m["rmse"])
        excess_rmse = round(max(0.0, raw_rmse - rmse_floor), 5)

        # Tool reliability floor: RMSE the multi-shoot tool would report even for a
        # perfect model at this amplitude, from hidden-state estimation errors.
        # First check the pre-computed sweep for the nearest amplitude; if not
        # close enough, compute exactly for this probe's amplitude (no plant call).
        _floor_at   = self._tool_reliability.get("floor_at", {})
        _nearest    = min(_floor_at.keys(), key=lambda a: abs(a - amplitude_fraction), default=None)
        if _nearest is not None and abs(_nearest - amplitude_fraction) <= 0.05:
            tool_floor = _floor_at[_nearest]
        else:
            tool_floor = compute_tool_reliability_floor(
                self._simulator, self._fitted_p, self._model_meta,
                self._contract, amplitude_fraction, n_samples=150, seg_len=25,
            )
        tool_reliable = tool_floor < self._rmse_tol * 0.5

        summary = {
            "scenario_type":            scenario_type,
            "amplitude_fraction":       amplitude_fraction,
            "run_id":                   result["run_id"],
            "rmse":                     round(raw_rmse, 5),
            "nrmse":                    round(float(m.get("nrmse", float("nan"))), 5),
            "rmse_floor":               round(rmse_floor, 5),
            "excess_rmse":              excess_rmse,
            "tool_reliability_floor":   round(tool_floor, 5),
            "tool_reliable":            tool_reliable,
            "residual_whiteness_p":     round(float(wb["p_value"]), 4),
            "residuals_white":          bool(wb["p_value"] > 0.05),
            "max_input_correlation":    round(float(xc["max_cross_corr"]), 4),
            "max_feature_correlation":  round(float(max_feat_corr), 4),
            "top_correlated_features":  [
                {"feature": k, "correlation": round(float(v), 4)} for k, v in top_features
            ],
            "residual_mean": round(float(np.mean(ms_residuals)), 5),
            "residual_std":  round(float(np.std(ms_residuals)), 5),
            "passes_rmse":   bool(excess_rmse < self._rmse_tol),
        }

        self.probe_results.append(summary)
        logger.debug(
            "Probe %s amp=%.2f → RMSE=%.4f floor=%.4f excess=%.4f tool_floor=%.4f reliable=%s",
            scenario_type, amplitude_fraction, raw_rmse, rmse_floor, excess_rmse,
            tool_floor, tool_reliable,
        )
        return json.dumps(summary)

    @staticmethod
    def _design_input(
        scenario_type: str,
        amplitude: float,
        n_samples: int,
        dt: float,
        T_total: float,
        nyquist: float,
        lo: float,
        hi: float,
        seed: int,
    ):
        t = np.arange(n_samples) * dt

        if scenario_type == "low_freq_sine":
            f = max(1.0 / T_total, 0.01)
            u = amplitude * np.sin(2 * np.pi * f * t)

        elif scenario_type == "slow_sine":
            f = max(2.0 / T_total, 0.01)
            u = amplitude * np.sin(2 * np.pi * f * t)

        elif scenario_type == "near_saturation":
            _, u = generate_prbs(n_samples, dt, amplitude=amplitude, seed=seed)

        elif scenario_type == "broadband_chirp":
            f_lo = max(1.0 / T_total * 0.5, 0.01)
            f_hi = min(nyquist * 0.35, 10.0)
            _, u = generate_chirp(n_samples, dt, f0=f_lo, f1=f_hi, amplitude=amplitude)

        elif scenario_type == "prbs":
            _, u = generate_prbs(n_samples, dt, amplitude=amplitude, seed=seed)

        elif scenario_type == "multisine":
            f_lo = 0.1
            f_hi = min(nyquist * 0.4, 5.0)
            n_freqs = min(8, max(3, int(np.log2(n_samples))))
            freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freqs).tolist()
            _, u = generate_multisine(n_samples, dt, frequencies=freqs,
                                      amplitude=amplitude, seed=seed)

        elif scenario_type == "step_sequence":
            n_levels  = 5
            levels    = np.linspace(-amplitude, amplitude, n_levels).tolist()
            hold_time = T_total / n_levels
            t, u = generate_steps(levels, hold_time, dt)
            t = t[:n_samples]
            u = u[:n_samples]

        else:
            raise ValueError(
                f"Unknown scenario_type '{scenario_type}'. "
                "Choose: low_freq_sine, near_saturation, broadband_chirp, "
                "slow_sine, prbs, multisine, step_sequence."
            )

        u = np.clip(u, lo, hi)
        return t, u


# ── ValidationAgent (orchestrator node) ──────────────────────────────────────

class ValidationAgent:
    """
    Adversarial validation agent with LLM-driven probing loop.

    The LLM iteratively designs and runs probe scenarios, inspects residual
    summaries, and posts a structured Verdict with failure_hypothesis and
    worst_case_inputs.  If the LLM loop fails, a rule-based fallback uses
    whatever probes were completed.
    """

    def __init__(
        self,
        plant_api:  PlantAPI,
        registry:   ModelRegistry,
        db:         ExperimentDatabase,
        rmse_tol:   float = RMSE_TOLERANCE,
    ):
        self._api      = plant_api
        self._registry = registry
        self._db       = db
        self._rmse_tol = rmse_tol
        self._designer = ExperimentDesignAgent()

    # ── Orchestrator node interface ───────────────────────────────────────────

    def __call__(self, dossier: Dossier) -> Dossier:
        model_id    = dossier.artifacts.current_model_id
        contract_id = dossier.assets.plant_contract_id or ""

        if not model_id:
            return _fail_dossier(dossier, "Validation: no model_id in dossier")

        verdict, report = self.run(model_id, contract_id)

        # Critique carries the full LLM failure hypothesis so downstream agents
        # know exactly what to fix, not just "structured_residual".
        critiques = list(dossier.open_critiques)
        if verdict.verdict == VerdictResult.FAIL and verdict.critique_id:
            critique_desc = verdict.failure_hypothesis or report.summary
            critiques.append(Critique(
                id=verdict.critique_id,
                addressed_to=_critique_target(verdict.gap_type, dossier),
                ref=model_id,
                description=critique_desc,
            ))

        val_rmse   = verdict.metrics.get("rmse", float("nan"))
        model_meta = {}
        try:
            model_meta = self._registry.load_model(model_id).metadata
        except Exception:
            pass

        attempt = AttemptEntry(
            rung         = dossier.current_rung.value,
            agent        = (dossier.last_report.agent if dossier.last_report else "unknown"),
            model_class  = model_meta.get("model_class",
                           model_meta.get("greybox_strategy",
                           model_meta.get("normalized_rhs", "NLS"))),
            n_train      = model_meta.get("n_train", 0),
            epochs       = model_meta.get("n_epochs", None),
            train_rmse   = model_meta.get("train_rmse", float("nan")),
            val_rmse     = val_rmse,
            scenario_rmse= {
                t: verdict.metrics.get(f"rmse_{i}", float("nan"))
                for i, t in enumerate(
                    verdict.metrics.get("scenario_types", [f"scenario_{i}" for i in range(3)])
                )
            },
            gap_type        = verdict.gap_type.value,
            agent_reasoning = verdict.failure_hypothesis or "",
        )

        arts = dossier.artifacts
        if arts.best_val_rmse is None or val_rmse < arts.best_val_rmse:
            arts = arts.model_copy(update={
                "best_val_rmse": val_rmse,
                "best_model_id": model_id,
            })

        return dossier.update(
            status=f"validation: {verdict.verdict.value}/{verdict.gap_type.value}",
            last_verdict=verdict,
            open_critiques=critiques,
            attempt_log=dossier.attempt_log + [attempt],
            artifacts=arts.model_copy(update={
                "validation_report_ids": (
                    arts.validation_report_ids + [verdict.validity_region_id or ""]
                ),
            }),
            last_report=report,
        )

    # ── Main validation run ───────────────────────────────────────────────────

    def run(self, model_id: str, contract_id: str = "", deterministic: bool = False) -> tuple[Verdict, Report]:
        """
        Run adversarial validation.

        deterministic=True forces the 3-scenario rule-based path even when an
        API key is present.  Used by greybox agent's evaluate_model tool so it
        gets a fast quality check without spawning a nested LLM loop.
        """
        model   = self._registry.load_model(model_id)
        meta    = model.metadata
        rhs     = meta.get("normalized_rhs", "")
        params  = meta.get("fit_params", [])
        state_v = meta.get("state_vars", [])
        input_v = meta.get("input_vars", [])

        contract           = self._load_contract(contract_id)
        system_order       = meta.get("system_order", len(state_v) if state_v else 2)
        output_state_index = meta.get("output_state_index", 0)

        simulator, fitted_p = self._compile_simulator(
            rhs, params, state_v, input_v, system_order, output_state_index, meta, model
        )
        if simulator is None:
            return _budget_exhausted_verdict(model_id)

        # Load parameter covariance for the MC floor computation.
        # Falls back gracefully to None (floor = 0, excess_rmse = raw rmse).
        param_cov = None
        cov_id    = model.parameter_covariance_id
        if cov_id:
            try:
                param_cov = self._registry.load_covariance(cov_id)
                if param_cov is not None and not np.all(np.isfinite(param_cov)):
                    param_cov = None
            except Exception as _e:
                logger.debug("Could not load covariance %s: %s", cov_id, _e)

        feat_lib = FeatureLibrary()
        probe_results: List[dict] = []
        verdict_data:  dict       = {}

        # Compute tool reliability sweep before probing — no plant calls, fast.
        # Shows the LLM which amplitudes give trustworthy RMSE measurements.
        tool_reliability = compute_reliability_sweep(
            simulator, fitted_p, meta, contract, self._rmse_tol,
        )
        logger.info(
            "[ValidationAgent] tool reliability ceiling=%.2f  sweep=%s",
            tool_reliability["reliability_ceiling"],
            [(s["amplitude_fraction"], s["tool_floor"]) for s in tool_reliability["sweep"]],
        )

        if deterministic:
            probe_results = self._run_deterministic_probes(
                simulator, fitted_p, contract, meta, feat_lib, param_cov=param_cov,
                tool_reliability=tool_reliability,
            )
        else:
            try:
                probe_loop = _LLMProbeLoop(
                    simulator=simulator,
                    fitted_p=fitted_p,
                    plant_api=self._api,
                    db=self._db,
                    designer=self._designer,
                    contract=contract,
                    rmse_tol=self._rmse_tol,
                    model_meta=meta,
                    model_id=model_id,
                    feat_lib=feat_lib,
                    param_cov=param_cov,
                    tool_reliability=tool_reliability,
                )
                system_prompt = (
                    PROMPT_PATH.read_text() if PROMPT_PATH.exists()
                    else "You are the Validation agent. Probe the model adversarially and post_verdict."
                )
                task_msg = _build_validation_task_message(
                    meta, model_id, contract, self._rmse_tol,
                    tool_reliability=tool_reliability,
                )
                probe_loop.run_probe_loop(system_prompt, task_msg)
                probe_results = probe_loop.probe_results
                verdict_data  = probe_loop.verdict_data
            except RuntimeError as exc:
                if "ANTHROPIC_API_KEY" in str(exc):
                    logger.info("No API key — falling back to deterministic 3-scenario validation")
                    probe_results = self._run_deterministic_probes(
                        simulator, fitted_p, contract, meta, feat_lib, param_cov=param_cov,
                        tool_reliability=tool_reliability,
                    )
                else:
                    raise

        if not probe_results:
            return _budget_exhausted_verdict(model_id)

        # Aggregate metrics from all probes
        all_rmse     = [r["rmse"] for r in probe_results]
        worst_rmse   = float(np.max(all_rmse))
        metrics: Dict[str, Any] = {
            "rmse":                    worst_rmse,
            "scenario_types":          [r["scenario_type"] for r in probe_results],
            "residual_whiteness_p":    float(np.min([r["residual_whiteness_p"] for r in probe_results])),
            "max_input_correlation":   float(np.max([r["max_input_correlation"] for r in probe_results])),
            "max_feature_correlation": float(np.max([r["max_feature_correlation"] for r in probe_results])),
            "n_scenarios":             len(probe_results),
            "n_probes":                len(probe_results),
        }
        for i, r in enumerate(probe_results):
            metrics[f"rmse_{i}"] = r["rmse"]

        # Covariance-based excess RMSE: parameter-error component stripped of
        # trajectory-divergence noise.  Aggregated as worst across all probes.
        _excess_vals = [r["excess_rmse"] for r in probe_results if r.get("excess_rmse") is not None]
        _floor_vals  = [r["rmse_floor"]  for r in probe_results if r.get("rmse_floor")  is not None]
        if _excess_vals:
            metrics["excess_rmse"] = float(max(_excess_vals))
            metrics["rmse_floor"]  = float(max(_floor_vals)) if _floor_vals else 0.0
            logger.info(
                "[ValidationAgent] worst rmse=%.4f | floor=%.4f | excess=%.4f (tol=%.3f)",
                worst_rmse, metrics["rmse_floor"], metrics["excess_rmse"], self._rmse_tol,
            )

        # Extract training baseline for regime-adaptive quality assessment.
        # The estimator stores baseline_y_range (lowest-amp dataset peak-to-peak)
        # and baseline_nrmse so we can detect when a probe has crossed into the
        # spinning/saturation regime where absolute RMSE is no longer a fair metric.
        tc               = (meta or {}).get("training_context", {})
        baseline_y_range = tc.get("baseline_y_range")
        baseline_nrmse   = tc.get("baseline_nrmse")

        # Detect regime boundary: the lowest amplitude where probe y_range >> baseline.
        _regime_boundary = None
        if baseline_y_range:
            for r in probe_results:
                probe_yr = _probe_y_range(r)
                if probe_yr > 4.0 * baseline_y_range:
                    amp = r.get("amplitude_fraction", 0.0)
                    if _regime_boundary is None or amp < _regime_boundary:
                        _regime_boundary = amp
        if _regime_boundary is not None:
            metrics["regime_boundary_amplitude"] = float(_regime_boundary)

        # Regime-adjusted worst RMSE: spinning-regime probes that satisfy the NRMSE
        # criterion are capped at rmse_tol×0.99 so they don't block a PASS verdict.
        adj_worst = _regime_adjusted_worst_rmse(
            probe_results, self._rmse_tol, baseline_y_range, baseline_nrmse
        )
        metrics["worst_rmse_adjusted"] = adj_worst

        # Gap type: use LLM verdict if valid, else rule-based fallback
        res, gap, failure_hypothesis, worst_case_inputs = self._extract_verdict(
            verdict_data, metrics, probe_results, meta=meta
        )

        # Surface worst-case scenario info in metrics so router + estimator can read it
        if worst_case_inputs:
            metrics.setdefault("worst_case_scenario",
                               worst_case_inputs.get("scenario_type", ""))
            metrics.setdefault("worst_case_amplitude_fraction",
                               worst_case_inputs.get("amplitude_fraction", 0.7))

        # Surface amplitude-dependence analysis for the experiment planner.
        # Computed deterministically from probe results so the planner always has it.
        # Uses regime-aware pass/fail so spinning-regime probes don't mask the gap.
        amp_dep = _amplitude_gap_detected(
            probe_results, self._rmse_tol, baseline_y_range, baseline_nrmse
        )
        passing_amps = [
            r.get("amplitude_fraction", 0.0)
            for r in probe_results
            if _probe_passes_regime_aware(r, self._rmse_tol, baseline_y_range, baseline_nrmse)
        ]
        metrics["amplitude_dependent_failure"] = amp_dep
        metrics["best_amplitude_tier"] = float(max(passing_amps)) if passing_amps else 0.0

        # Identify dominant features from the worst probe (for router task message)
        dominant_features: List[str] = []
        if probe_results:
            worst_p = max(probe_results, key=lambda r: r.get("rmse", 0.0))
            dominant_features = [
                f"{f['feature']} (r={f['correlation']:.2f})"
                for f in worst_p.get("top_correlated_features", [])[:3]
            ]

        validity_id = self._store_validity(model_id, worst_rmse, metrics)
        critique_id = None
        if res == VerdictResult.FAIL:
            from core.schemas import _uid
            critique_id = _uid()

        verdict = Verdict(
            verdict=res,
            gap_type=gap,
            metrics=metrics,
            validity_region_id=validity_id,
            uncertainty_calibrated=False,
            critique_id=critique_id,
            failure_hypothesis=failure_hypothesis or None,
            worst_case_inputs=worst_case_inputs or None,
        )

        gap_desc = {
            GapType.NONE:                "clean fit",
            GapType.FIXABLE:             "parameter inaccuracy → re-estimate",
            GapType.STRUCTURED_RESIDUAL: "missing physics term → grey-box",
            GapType.UNMODELABLE:         "large unstructured residuals",
        }[gap]

        report = Report(
            agent="ValidationAgent",
            status=AgentStatus.DONE,
            produced=[ArtifactRef(id=validity_id, type="validity_region", store="results")],
            summary=(
                f"Validation {res.value}: gap={gap.value} ({gap_desc}). "
                f"worst RMSE={worst_rmse:.4f}, {len(probe_results)} probes."
            ),
            metadata={
                "model_id":           model_id,
                "verdict":            res.value,
                "gap_type":           gap.value,
                "metrics":            metrics,
                "validity_region_id": validity_id,
                "failure_hypothesis": failure_hypothesis,
                "dominant_features":  dominant_features,
            },
        )
        return verdict, report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compile_simulator(
        self, rhs, params, state_v, input_v,
        system_order, output_state_index, meta, model
    ):
        """Compile the right simulator for the model type. Returns (simulator, fitted_p)."""
        try:
            if rhs == "SURROGATE":
                surrogate_obj_id = meta.get("surrogate_object_id", "")
                predictor        = self._registry.load_object(surrogate_obj_id)
                paradigm_str     = meta.get("surrogate_paradigm", "ode")
                from agents.surrogate.model_class_selector import Paradigm
                if paradigm_str == Paradigm.INPUT_OUTPUT.value:
                    sim = _make_io_surrogate_simulator(predictor, output_state_index)
                else:
                    sim = _make_surrogate_simulator(
                        predictor, n_states=system_order,
                        output_state_index=output_state_index,
                    )
                return sim, np.array([])

            elif rhs == "RESIDUAL_CORRECTED":
                physics_model_id = meta.get("physics_model_id", "")
                residual_obj_id  = meta.get("residual_object_id", "")
                phys_model  = self._registry.load_model(physics_model_id)
                phys_meta   = phys_model.metadata
                phys_rhs    = phys_meta.get("normalized_rhs", "")
                phys_params = phys_meta.get("fit_params", [])
                phys_p      = np.array([phys_model.parameters.get(p, 1.0) for p in phys_params])
                try:
                    phys_sim = make_ode_simulator(
                        phys_rhs, phys_params, state_v, input_v,
                        highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
                        output_state_index=output_state_index,
                    )
                except Exception:
                    phys_sim = None
                residual_predictor = None
                if residual_obj_id:
                    try:
                        residual_predictor = self._registry.load_object(residual_obj_id)
                    except Exception:
                        pass
                if phys_sim is not None and residual_predictor is not None:
                    return _make_residual_sequence_simulator(phys_sim, phys_p, residual_predictor), np.array([])
                elif phys_sim is not None:
                    return phys_sim, np.array([])
                return None, None

            elif rhs == "GP_CORRECTED":
                base_rhs          = meta.get("base_rhs", "")
                correction_obj_id = meta.get("correction_object_id", "")
                correction_fn     = None
                if correction_obj_id:
                    try:
                        correction_fn = self._registry.load_object(correction_obj_id)
                    except Exception:
                        pass
                sim = make_ode_simulator(
                    base_rhs, params, state_v, input_v,
                    highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
                    output_state_index=output_state_index,
                    correction_fn=correction_fn,
                )
                return sim, np.array([model.parameters.get(p, 1.0) for p in params])

            elif rhs == "SINDY_OUTPUT_CORRECTED":
                # Physics baseline + additive output-domain symbolic correction.
                # y_pred(t) = y_phys(t) + Σ c_i · φ_i(x_phys(t), u(t))
                physics_model_id  = meta.get("physics_model_id", "")
                correction_coeffs = meta.get("correction_coeffs", {})
                phys_model  = self._registry.load_model(physics_model_id)
                phys_meta   = phys_model.metadata
                phys_rhs    = phys_meta.get("normalized_rhs", "")
                phys_params = phys_meta.get("fit_params", [])
                phys_p      = np.array([phys_model.parameters.get(p, 1.0) for p in phys_params])
                try:
                    phys_sim = make_ode_simulator(
                        phys_rhs, phys_params, state_v, input_v,
                        highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
                        output_state_index=output_state_index,
                    )
                except Exception:
                    phys_sim = None
                if phys_sim is None:
                    return None, None

                from tools.feature_library import FeatureLibrary
                from agents.estimator import _estimate_hidden_states

                def _sindy_output_sim(param_values, t_arr, u_arr, x0=None):
                    y_phys = phys_sim(phys_p, t_arr, u_arr, x0=x0)
                    if not correction_coeffs or np.any(np.isnan(y_phys)):
                        return y_phys
                    so = meta.get("system_order", system_order)
                    states_ph = _estimate_hidden_states(t_arr, y_phys, so)
                    Theta, feat_names = FeatureLibrary().build(
                        states_ph, u_arr, state_v, input_v
                    )
                    c = np.array([correction_coeffs.get(n, 0.0) for n in feat_names])
                    return y_phys + Theta @ c

                return _sindy_output_sim, np.array([])

            else:
                sim = make_ode_simulator(
                    rhs, params, state_v, input_v,
                    highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
                    output_state_index=output_state_index,
                )
                return sim, np.array([model.parameters.get(p, 1.0) for p in params])

        except Exception as exc:
            logger.error("Simulator compilation failed: %s", exc)
            return None, None

    def _extract_verdict(
        self,
        verdict_data: dict,
        metrics: Dict[str, Any],
        probe_results: List[dict],
        meta: dict = None,
    ) -> tuple[VerdictResult, GapType, str, dict]:
        """
        Extract verdict from LLM output; fall back to rule-based if missing or invalid.
        Returns (result, gap_type, failure_hypothesis, worst_case_inputs).
        """
        failure_hypothesis = ""
        worst_case_inputs  = {}

        # Extract training baseline so both the LLM-override path and rule-based
        # fallback can apply regime-aware pass/fail (spinning vs oscillating regime).
        tc               = (meta or {}).get("training_context", {})
        baseline_y_range = tc.get("baseline_y_range")
        baseline_nrmse   = tc.get("baseline_nrmse")

        if verdict_data:
            try:
                res = VerdictResult(verdict_data.get("verdict", "fail"))
                gap = GapType(verdict_data.get("gap_type", "unmodelable"))
                failure_hypothesis = verdict_data.get("failure_hypothesis", "")
                worst_case_inputs  = {
                    "scenario_type":      verdict_data.get("worst_case_scenario", ""),
                    "amplitude_fraction": verdict_data.get("worst_case_amplitude_fraction", 0.7),
                }
                # Override LLM gap_type if amplitude-dependence evidence is conclusive.
                # A model that passes at low amplitude but fails at high amplitude has
                # a PARAMETER problem (not a missing structural term).  No structural
                # term could be invisible at small excitation and dominant at large excitation
                # while the model already uses nonlinear functions like sin(theta).
                if res == VerdictResult.FAIL and gap == GapType.STRUCTURED_RESIDUAL:
                    amp_gap = _amplitude_gap_detected(
                        probe_results, self._rmse_tol, baseline_y_range, baseline_nrmse
                    )
                    if amp_gap:
                        logger.info(
                            "[ValidationAgent] Overriding LLM gap_type structured_residual→fixable: "
                            "amplitude-dependence detected (passes at low amp, fails at high amp)."
                        )
                        gap = GapType.FIXABLE
                        failure_hypothesis = (
                            "[Amplitude-dependence override] " + failure_hypothesis
                            if failure_hypothesis
                            else "Model passes at small amplitude but fails at large amplitude — "
                                 "this is parameter inaccuracy, not a missing structural term."
                        )
                return res, gap, failure_hypothesis, worst_case_inputs
            except (ValueError, KeyError):
                logger.warning("LLM verdict_data invalid — falling back to rule-based verdict")

        # Rule-based fallback
        min_whiteness_p = metrics["residual_whiteness_p"]
        max_input_corr  = metrics["max_input_correlation"]
        max_feat_corr   = metrics["max_feature_correlation"]

        rhs = (meta or {}).get("normalized_rhs", "")

        # Effective worst RMSE for pass/fail decisions.
        # Priority: (1) covariance-based excess_rmse — strips trajectory-divergence
        # noise and is fully system-agnostic; (2) regime-adjusted worst RMSE — uses
        # baseline_y_range heuristic; (3) raw worst RMSE.
        excess_worst   = metrics.get("excess_rmse")           # None when no covariance
        adj_worst_rmse = _regime_adjusted_worst_rmse(
            probe_results, self._rmse_tol, baseline_y_range, baseline_nrmse
        )
        effective_worst = excess_worst if excess_worst is not None else adj_worst_rmse

        if effective_worst < self._rmse_tol and max_feat_corr < FEAT_CORR_THR:
            # Non-white residuals are structurally expected for ODE multi-shoot models
            # (initial-state error propagates through dynamics, creating autocorrelation).
            # Use feature correlation as the structural-residual gate instead of whiteness.
            res, gap = VerdictResult.PASS, GapType.NONE
        elif _amplitude_gap_detected(probe_results, self._rmse_tol, baseline_y_range, baseline_nrmse):
            # Clear amplitude threshold: passes at low amplitude, fails at high.
            # A genuinely missing structural term would fail at ALL amplitudes.
            res, gap = VerdictResult.FAIL, GapType.FIXABLE
        elif max_feat_corr > FEAT_CORR_THR:
            # Check if top feature is already in the model — if so, it's parameter inaccuracy,
            # not a structural gap.
            if rhs and _top_feature_in_model(probe_results, rhs):
                res, gap = VerdictResult.FAIL, GapType.FIXABLE
            else:
                res, gap = VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL
        elif max_input_corr > INPUT_CORR_THR:
            res, gap = VerdictResult.FAIL, GapType.FIXABLE
        elif effective_worst >= self._rmse_tol:
            res, gap = VerdictResult.FAIL, GapType.UNMODELABLE
        else:
            res, gap = VerdictResult.FAIL, GapType.UNMODELABLE

        # Build a minimal hypothesis from the rule-based data
        if res == VerdictResult.FAIL:
            # Sort by excess_rmse if available, else raw rmse
            worst_idx = int(np.argmax(
                [r.get("excess_rmse", r["rmse"]) for r in probe_results]
            ))
            worst_sc  = probe_results[worst_idx]
            top_feats = worst_sc.get("top_correlated_features", [])
            feat_str  = (
                ", ".join(f["feature"] for f in top_feats[:3]) if top_feats else "none identified"
            )
            _floor_str = (
                f" floor={worst_sc.get('rmse_floor', 0):.4f}"
                if worst_sc.get("rmse_floor") is not None else ""
            )
            failure_hypothesis = (
                f"Rule-based diagnosis: gap={gap.value}. "
                f"Worst excess_rmse={effective_worst:.4f}{_floor_str} on scenario "
                f"'{worst_sc['scenario_type']}' (amplitude={worst_sc['amplitude_fraction']:.2f}). "
                f"Top correlated features: {feat_str}. "
                f"Residual whiteness p={min_whiteness_p:.3f}, "
                f"max_feat_corr={max_feat_corr:.3f}, max_input_corr={max_input_corr:.3f}."
            )
            worst_case_inputs = {
                "scenario_type":      worst_sc["scenario_type"],
                "amplitude_fraction": worst_sc["amplitude_fraction"],
            }

        return res, gap, failure_hypothesis, worst_case_inputs

    def _run_deterministic_probes(
        self, simulator, fitted_p, contract, meta: dict, feat_lib: FeatureLibrary,
        param_cov: Optional[np.ndarray] = None,
        tool_reliability: Optional[dict] = None,
    ) -> List[dict]:
        """
        Fallback: run the 3 standard adversarial scenarios deterministically
        (used when no ANTHROPIC_API_KEY is available).
        """
        tool_reliability = tool_reliability or {}
        scenarios = self._designer.design_for_validation(
            contract, n_samples_per_scenario=300, n_scenarios=3, seed=42
        )
        state_v      = meta.get("state_vars", [])
        input_v      = meta.get("input_vars", [])
        system_order = meta.get("system_order", 2)
        results: List[dict] = []

        for sc in scenarios:
            amp_frac = sc.get("amplitude_fraction", 0.7)
            t_sc, u_sc = sc["t"], sc["u"]
            u_func = self._designer.make_u_func(t_sc, u_sc)
            try:
                result = self._api.apply_input(
                    u_func=u_func,
                    t_span=(float(t_sc[0]), float(t_sc[-1])),
                    dt=float(t_sc[1] - t_sc[0]),
                    purpose="validation",
                    input_type=sc["scenario_type"],
                    agent="validation",
                    split_flag=SplitFlag.VALIDATION,
                )
            except Exception as exc:
                logger.warning("Deterministic probe %s skipped: %s", sc["scenario_type"], exc)
                continue

            t_data, u_data, y_data = self._db.load_arrays(result["run_id"])
            y_true = y_data[0]
            u_flat = u_data[0]

            states_est   = _estimate_hidden_states(t_data, y_true, system_order)
            ms_residuals = _multi_shoot(
                simulator, fitted_p, t_data, u_flat, y_true, states_est, seg_len=25
            )
            n_ms   = len(ms_residuals)
            y_pred = y_true[:n_ms] - ms_residuals

            m  = compute_metrics(y_true[:n_ms], y_pred)
            wb = residual_whiteness_test(ms_residuals)
            xc = residual_correlation_with_input(ms_residuals, u_flat[:n_ms])

            feat_corr_map: Dict[str, float] = {}
            if state_v:
                try:
                    _Theta, _names = feat_lib.build(
                        states_est[:, :n_ms], u_flat[:n_ms], state_v, input_v
                    )
                    feat_corr_map = feat_lib.feature_correlations(_Theta, _names, ms_residuals)
                except Exception:
                    pass

            max_feat_corr = max(feat_corr_map.values(), default=0.0)
            top_features  = sorted(feat_corr_map.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            rmse_floor  = _compute_parameter_rmse_floor(
                simulator, fitted_p, param_cov,
                t_data, u_flat, y_true, states_est, seg_len=25, n_mc=10,
            )
            raw_rmse    = float(m["rmse"])
            excess_rmse = max(0.0, raw_rmse - rmse_floor)

            # Tool reliability for this probe's amplitude
            _floor_at = tool_reliability.get("floor_at", {})
            _nearest  = min(_floor_at.keys(), key=lambda a: abs(a - amp_frac), default=None)
            tool_floor = (
                _floor_at[_nearest]
                if (_nearest is not None and abs(_nearest - amp_frac) <= 0.05)
                else compute_tool_reliability_floor(
                    simulator, fitted_p, meta, contract, amp_frac, n_samples=150, seg_len=25,
                )
            )
            tool_reliable = tool_floor < self._rmse_tol * 0.5

            results.append({
                "scenario_type":           sc["scenario_type"],
                "amplitude_fraction":      amp_frac,
                "run_id":                  result["run_id"],
                "rmse":                    raw_rmse,
                "nrmse":                   float(m.get("nrmse", float("nan"))),
                "rmse_floor":              rmse_floor,
                "excess_rmse":             excess_rmse,
                "tool_reliability_floor":  tool_floor,
                "tool_reliable":           tool_reliable,
                "residual_whiteness_p":    float(wb["p_value"]),
                "residuals_white":         bool(wb["p_value"] > 0.05),
                "max_input_correlation":   float(xc["max_cross_corr"]),
                "max_feature_correlation": float(max_feat_corr),
                "top_correlated_features": [
                    {"feature": k, "correlation": float(v)} for k, v in top_features
                ],
                "residual_mean": float(np.mean(ms_residuals)),
                "residual_std":  float(np.std(ms_residuals)),
                "passes_rmse":   bool(excess_rmse < self._rmse_tol),
            })

        return results

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

    def _store_validity(self, model_id: str, rmse: float, metrics: dict) -> str:
        region = ValidityRegion(
            model_id=model_id,
            bounds={"output": (-0.5, 0.5)},
            tolerance=self._rmse_tol,
            achieved_rmse=rmse,
            coverage_fraction=float(metrics.get("n_scenarios", 1) / 3),
            metadata=metrics,
        )
        return self._registry.store_validity(region)


# ── Task message builder ──────────────────────────────────────────────────────

def _build_validation_task_message(
    meta: dict, model_id: str, contract, rmse_tol: float,
    tool_reliability: Optional[dict] = None,
) -> str:
    input_name = contract.input_names[0]
    lo, hi     = contract.input_limits.get(input_name, (-1.0, 1.0))
    rhs        = meta.get("normalized_rhs", "unknown")
    fp         = meta.get("fit_params", [])
    sv         = meta.get("state_vars", [])
    iv         = meta.get("input_vars", [])
    tc         = meta.get("training_context", {})
    n_train    = meta.get("n_train", None)

    # ── Tool reliability section ──────────────────────────────────────────────
    tool_lines = ["", "## Tool reliability envelope (computed before any plant calls)"]
    tr = tool_reliability or {}
    ceiling = tr.get("reliability_ceiling", None)
    sweep   = tr.get("sweep", [])
    if sweep:
        tool_lines += [
            f"Reliability ceiling: {ceiling:.2f}  "
            f"(highest amplitude where tool_floor < {rmse_tol * 0.5:.3f} = tolerance/2)",
            "",
            "amp  | tool_floor | reliable?",
            "-----|------------|----------",
        ]
        for s in sweep:
            mark = "YES" if s["tool_reliable"] else "NO "
            tool_lines.append(
                f"{s['amplitude_fraction']:.2f} | {s['tool_floor']:.5f}   | {mark}"
            )
        tool_lines += [
            "",
            "TOOL RELIABILITY RULES (apply BEFORE choosing probe amplitudes):",
            "  • Each probe result now includes 'tool_reliability_floor' and 'tool_reliable'.",
            "  • When tool_reliable=False: RMSE at this amplitude cannot distinguish model",
            "    error from simulation tool artifact — do NOT use this probe for pass/fail.",
            "    The tool itself produces errors above tolerance/2 even for a perfect model,",
            "    because hidden-state estimation (e.g. velocity from position) degrades at",
            "    high amplitude and the ODE amplifies those errors across each segment.",
            "  • When tool_reliable=True: RMSE is a trustworthy model quality signal here.",
            f"  • DESIGN YOUR PROBES primarily at amplitudes ≤ {ceiling:.2f} where the tool",
            "    is reliable. You may run one probe above the ceiling to confirm the boundary,",
            "    but do not use it to fail the model.",
            "  • The ceiling is model-specific and recomputed fresh for every model (white-box,",
            "    grey-box, surrogate) — a better model may have a higher reliability ceiling.",
        ]
    else:
        tool_lines += [
            "Not computed (fully-observed system or simulator unavailable).",
            "Use standard RMSE criteria at all amplitudes.",
        ]

    # ── Training context section ──────────────────────────────────────────────
    tc_lines = []
    if tc:
        tc_lines += [
            "",
            "## Estimator training context — use this to interpret probe results",
            f"n_train samples:   {n_train if n_train is not None else 'unknown'}",
            f"CV converged:      {tc.get('cv_converged', 'unknown')}  "
            f"(False = parameters may still be inaccurate, not a structural gap)",
            f"Iterations run:    {tc.get('n_iterations', '?')} / 5",
            f"Segment length:    {tc.get('seg_len', '?')} samples  "
            f"(used in multi-shooting during training)",
            f"Amplitudes used:   {tc.get('amplitudes_used', 'unknown')}",
            f"Methods used:      {tc.get('methods_used', 'unknown')}",
            f"Stall reason:      {tc.get('stall_reason', 'none')}",
            f"Re-estimate #:     {tc.get('re_estimate_count', 0)}",
            "",
            "TRAINING-CONTEXT INTERPRETATION RULES (apply before classifying gap_type):",
            "  A) If cv_converged=False → parameters are unreliable; RMSE failures are likely",
            "     parameter inaccuracy, NOT missing structure. Prefer gap_type='fixable'.",
            "  B) If your failing probe amplitude > max(amplitudes_used) → the model was never",
            "     trained at that amplitude; failure = excitation gap, not structural gap → 'fixable'.",
            "  C) If training used short sequences (n_train small or n_iterations low) →",
            "     probe with n_samples=600 to check long-horizon prediction quality.",
            "  D) Only use gap_type='structured_residual' when rules A–C are ruled out AND",
            "     the top correlated feature is a NEW function not already in the ODE RHS.",
        ]
    else:
        tc_lines += [
            "",
            "## Estimator training context",
            "Not available — model may have been built without the estimator pipeline.",
        ]

    lines = [
        "## Task: Adversarial Model Validation",
        "",
        f"Model ID: {model_id}",
        f"RMSE tolerance: {rmse_tol}",
        "",
        "## Model structure — CRITICAL for gap_type classification",
        f"ODE right-hand side: {rhs}",
        f"Fitted parameters:   {fp}",
        f"State variables:     {sv}",
        f"Input variables:     {iv}",
        "",
        f"Plant: {contract.name}",
        f"Inputs: {contract.input_names}   Outputs: {contract.output_names}",
        f"Input limits [{input_name}]: {lo:.3f} to {hi:.3f}",
        f"Sample time: {contract.sample_time} s",
    ] + tool_lines + tc_lines + [
        "",
        "## Probing strategy",
        "1. Call get_model_metadata() to confirm model structure and training context.",
        f"2. Run probes WITHIN the reliability ceiling (≤ {ceiling:.2f}) first."
            if ceiling is not None else
        "2. Run probes across the full amplitude range.",
        "   Suggested standard sweep:",
        "   run_scenario('low_freq_sine',   amplitude_fraction=0.35)",
        "   run_scenario('broadband_chirp', amplitude_fraction=0.55)",
        f"   run_scenario('near_saturation', amplitude_fraction={min(0.9, ceiling):.2f})"
            if ceiling is not None else
        "   run_scenario('near_saturation', amplitude_fraction=0.90)",
        "3. Inspect each result: rmse, excess_rmse, tool_reliable, top_correlated_features.",
        "   tool_reliable=False → skip this probe for pass/fail; use nrmse as soft indicator only.",
        "4. Classify gap from RELIABLE probes only:",
        "   - Top feature already in ODE RHS → gap_type='fixable'",
        "   - cv_converged=False → gap_type='fixable'",
        "   - Top feature is a NEW function → gap_type='structured_residual'",
        "5. Run 2–5 targeted follow-up probes within the reliable amplitude range.",
        "6. Call post_verdict() + post_report() when done (max 8 probes total).",
        "",
        "Budget: max 8 probe calls.",
    ]
    return "\n".join(lines)


# ── Surrogate simulator wrappers (unchanged) ──────────────────────────────────

def _make_surrogate_simulator(predictor, n_states: int = 2, output_state_index: int = 0):
    from scipy.integrate import solve_ivp
    osi = output_state_index

    def simulator(params, t_seg, u_seg, x0=None):
        if x0 is None:
            x0 = np.zeros(n_states)
        x0 = np.asarray(x0, dtype=float)

        def rhs(t, x):
            idx = int(np.clip(np.searchsorted(t_seg, t, side="right") - 1, 0, len(u_seg) - 1))
            u_val        = float(u_seg[idx])
            xdot_highest = float(predictor.predict(*x, u_val))
            if not np.isfinite(xdot_highest):
                xdot_highest = 0.0
            return [*x[1:], xdot_highest]

        sol = solve_ivp(
            rhs, (float(t_seg[0]), float(t_seg[-1])), x0,
            t_eval=t_seg, method="RK45", rtol=1e-4, atol=1e-6,
        )
        if not sol.success or np.any(np.isnan(sol.y[osi])):
            return np.full(len(t_seg), np.nan)
        return sol.y[osi]

    return simulator


def _make_io_surrogate_simulator(predictor, output_state_index: int = 0):
    osi = output_state_index

    def simulator(params, t_seg, u_seg, x0=None):
        y_init = np.zeros(1)
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=float)
            if len(x0_arr) > osi:
                y_init = x0_arr[osi:osi + 1]
        try:
            y_pred = predictor.predict_sequence(y_init, np.asarray(u_seg, dtype=float))
        except Exception as _exc:
            import logging as _log
            _log.getLogger(__name__).warning(
                "IO surrogate predict_sequence failed (%s: %s) — segment filled with NaN",
                type(_exc).__name__, _exc,
            )
            return np.full(len(t_seg), np.nan)

        n = len(t_seg)
        if len(y_pred) >= n:
            return y_pred[:n]
        return np.concatenate([y_pred, np.full(n - len(y_pred), np.nan)])

    return simulator


def _make_residual_sequence_simulator(physics_sim, physics_p: np.ndarray, residual_predictor):
    lag_y = getattr(residual_predictor, "lag_y", 1)

    def simulator(params, t_seg, u_seg, x0=None):
        y_physics = physics_sim(physics_p, t_seg, u_seg, x0=x0)
        if np.any(np.isnan(y_physics)):
            return np.full(len(t_seg), np.nan)
        e_init = np.zeros(max(lag_y, 1))
        try:
            e_pred = residual_predictor.predict_sequence(e_init, np.asarray(u_seg, dtype=float))
        except Exception:
            e_pred = np.zeros(len(t_seg))
        n   = min(len(y_physics), len(e_pred))
        out = np.full(len(t_seg), np.nan)
        out[:n] = y_physics[:n] + e_pred[:n]
        return out

    return simulator


# ── Terminal / fallback helpers ───────────────────────────────────────────────

def _budget_exhausted_verdict(model_id: str) -> tuple:
    verdict = Verdict(
        verdict=VerdictResult.FAIL,
        gap_type=GapType.UNMODELABLE,
        metrics={"rmse": np.inf},
    )
    report = Report(
        agent="ValidationAgent",
        status=AgentStatus.FAILED,
        summary="Validation failed: no probe scenarios completed.",
        metadata={"model_id": model_id},
    )
    return verdict, report


def _critique_target(gap: GapType, dossier: Dossier) -> str:
    if gap == GapType.FIXABLE:
        return "estimator"  # parameter inaccuracy → re-estimate with better data
    return "greybox_so"


def _fail_dossier(dossier: Dossier, msg: str) -> Dossier:
    verdict = Verdict(
        verdict=VerdictResult.FAIL,
        gap_type=GapType.UNMODELABLE,
        metrics={"rmse": np.inf},
    )
    report = Report(
        agent="ValidationAgent",
        status=AgentStatus.FAILED,
        summary=msg,
    )
    return dossier.update(
        status=f"validation failed: {msg}",
        last_verdict=verdict,
        last_report=report,
    )
