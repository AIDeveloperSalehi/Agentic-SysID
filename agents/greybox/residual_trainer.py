"""
Residual correction model builder — generalized.

Given a chosen strategy and acceleration residuals, constructs an extended
ODE structure and provides warm-start parameter values for the NLS re-fit.

Strategies
----------
COULOMB_TERM   — appends K_c_fixed·tanh(vel_var/ε) to the ODE RHS.
                 K_c is fixed numerically (not a new fit parameter).
POLY_FALLBACK  — appends a1·vel_var + a3·vel_var³ (OLS pre-fit of coeffs).
SINDY          — appends a general sparse symbolic expression identified by
                 LASSO on a feature library of state/input functions.
GP_CORRECTION  — fits a GP on (states, u) → residual; stores as a correction
                 callable; the base ODE RHS is left unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agents.greybox.strategy_selector import Strategy

# Coulomb smoothing constant — matches the plant's ε in tanh(vel/ε)
COULOMB_EPS = 0.01   # rad/s (or compatible velocity unit)


@dataclass
class OutputCorrectionSpec:
    """
    Result of output-domain SINDy fitting.

    correction_expr : SymPy-parseable expression string, e.g. "0.023*theta**3"
    correction_coeffs : {feature_name: coeff} for active (non-zero) terms
    """
    correction_coeffs: Dict[str, float]
    correction_expr:   str = "0"


@dataclass
class ExtendedModelSpec:
    """
    Describes an extended ODE structure ready for re-fitting or evaluation.

    COULOMB_TERM    normalized_rhs = "<base> - K_c_fixed*tanh(vel/ε)"
                    fit_params unchanged; correction_coeffs={"K_c_fixed": ...}
    POLY_FALLBACK   normalized_rhs = "<base> + a1*vel + a3*vel**3"
                    fit_params += ["a1", "a3"]; correction_coeffs={"a1": ..., "a3": ...}
    SINDY           normalized_rhs = "<base> + <sparse symbolic expression>"
                    fit_params unchanged; correction_coeffs = {feature: coeff, ...}
    GP_CORRECTION   normalized_rhs unchanged; correction_object is a callable.
    """
    normalized_rhs:    str
    fit_params:        List[str]
    param_bounds:      Dict[str, List[float]]
    p0_override:       Dict[str, float]
    strategy:          Strategy
    correction_coeffs: Dict[str, float]          = field(default_factory=dict)
    correction_object: Optional[Any]             = None  # GP callable (GP path only)


class ResidualTrainer:
    """
    Constructs the extended model structure and provides warm-start estimates.
    """

    def fit(
        self,
        strategy:           Strategy,
        base_rhs:           str,
        base_params:        List[str],
        base_param_bounds:  Dict[str, List[float]],
        base_fitted_params: Dict[str, float],
        t:                  np.ndarray,
        theta_dot:          Optional[np.ndarray] = None,  # velocity array
        eps_ddot:           Optional[np.ndarray] = None,  # acceleration residuals
        K_c_estimate:       Optional[float]      = None,
        vel_var_name:       str                  = "theta_dot",  # actual variable name
        states:             Optional[np.ndarray] = None,          # (n, N) for SINDY/GP
        state_names:        Optional[List[str]]  = None,
        input_names:        Optional[List[str]]  = None,
        u:                  Optional[np.ndarray] = None,
    ) -> ExtendedModelSpec:
        """
        Build the ExtendedModelSpec for the chosen strategy.

        Parameters
        ----------
        vel_var_name : str
            Name of the velocity state variable in the ODE (e.g. "theta_dot",
            "omega", "v").  Defaults to "theta_dot" for backward compatibility.
        states : ndarray (system_order, N), optional
            Full state matrix — required for SINDY and GP_CORRECTION paths.
        """
        if strategy == Strategy.COULOMB_TERM:
            return self._fit_coulomb(
                base_rhs, base_params, base_param_bounds,
                K_c_estimate, vel_var_name,
                base_fitted_params,
            )
        elif strategy == Strategy.POLY_FALLBACK:
            return self._fit_poly(
                base_rhs, base_params, base_param_bounds,
                theta_dot if theta_dot is not None else np.array([]),
                eps_ddot  if eps_ddot  is not None else np.array([]),
                vel_var_name,
            )
        elif strategy == Strategy.SINDY:
            return self._fit_sindy(
                base_rhs, base_params, base_param_bounds,
                states, u, eps_ddot,
                state_names or [], input_names or [],
                base_fitted_params=base_fitted_params,
            )
        else:  # GP_CORRECTION
            return self._fit_gp(
                base_rhs, base_params, base_param_bounds,
                states, u, eps_ddot,
                state_names or [], input_names or [],
            )

    # ── Coulomb ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_coulomb(
        base_rhs:           str,
        base_params:        List[str],
        base_param_bounds:  Dict[str, List[float]],
        K_c_estimate:       Optional[float],
        vel_var_name:       str,
        base_fitted_params: Optional[Dict[str, float]] = None,
    ) -> ExtendedModelSpec:
        """
        Embed K_c as a numeric literal in the RHS — avoids K_c ↔ τ_d collinearity.
        Warm-starts NLS from the existing white-box fit values.

        If the base model already contains a tanh term on the velocity variable
        (i.e., a Coulomb-type term is already present as a fit parameter), skip
        appending a duplicate and just re-estimate the existing parameters.
        """
        import re
        already_present = bool(
            re.search(rf"tanh\s*\(\s*{re.escape(vel_var_name)}\s*/", base_rhs)
        )
        if already_present:
            # Coulomb term already in base model — re-estimate without duplicating.
            p0 = {p: v for p, v in (base_fitted_params or {}).items() if p in base_params}
            return ExtendedModelSpec(
                normalized_rhs=base_rhs,
                fit_params=base_params,
                param_bounds=base_param_bounds,
                p0_override=p0,
                strategy=Strategy.COULOMB_TERM,
                correction_coeffs={"already_present": True},
            )

        k_c_fixed = float(K_c_estimate) if K_c_estimate is not None else 1.0
        new_rhs = f"{base_rhs} - {k_c_fixed:.6f}*tanh({vel_var_name}/{COULOMB_EPS})"

        # Warm-start from previously fitted base params to avoid tau_d drift from
        # Coulomb/viscous collinearity at near-zero velocity.
        p0 = {p: v for p, v in (base_fitted_params or {}).items() if p in base_params}

        return ExtendedModelSpec(
            normalized_rhs=new_rhs,
            fit_params=base_params,
            param_bounds=base_param_bounds,
            p0_override=p0,
            strategy=Strategy.COULOMB_TERM,
            correction_coeffs={"K_c_fixed": k_c_fixed},
        )

    # ── Polynomial fallback ───────────────────────────────────────────────────

    @staticmethod
    def _fit_poly(
        base_rhs:          str,
        base_params:       List[str],
        base_param_bounds: Dict[str, List[float]],
        theta_dot:         np.ndarray,
        eps_ddot:          np.ndarray,
        vel_var_name:      str,
    ) -> ExtendedModelSpec:
        """OLS for antisymmetric polynomial: a1·vel + a3·vel³."""
        valid = (
            np.isfinite(eps_ddot) & np.isfinite(theta_dot)
            if len(eps_ddot) > 0 and len(theta_dot) > 0
            else np.array([], dtype=bool)
        )
        a1_init, a3_init = 0.0, 0.0

        if valid.sum() >= 4:
            Phi = np.column_stack([theta_dot[valid], theta_dot[valid] ** 3])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(Phi, eps_ddot[valid], rcond=None)
                a1_init = float(np.clip(coeffs[0], -20.0, 20.0))
                a3_init = float(np.clip(coeffs[1], -5.0,   5.0))
            except np.linalg.LinAlgError:
                pass

        new_rhs    = f"{base_rhs} + a1*{vel_var_name} + a3*{vel_var_name}**3"
        new_params = base_params + ["a1", "a3"]
        new_bounds = {
            **base_param_bounds,
            "a1": [-20.0, 20.0],
            "a3": [-5.0,   5.0],
        }

        return ExtendedModelSpec(
            normalized_rhs=new_rhs,
            fit_params=new_params,
            param_bounds=new_bounds,
            p0_override={"a1": a1_init, "a3": a3_init},
            strategy=Strategy.POLY_FALLBACK,
            correction_coeffs={"a1": a1_init, "a3": a3_init},
        )

    # ── SINDy sparse symbolic correction ─────────────────────────────────────

    @staticmethod
    def _fit_sindy(
        base_rhs:           str,
        base_params:        List[str],
        base_param_bounds:  Dict[str, List[float]],
        states:             Optional[np.ndarray],
        u:                  Optional[np.ndarray],
        eps_ddot:           Optional[np.ndarray],
        state_names:        List[str],
        input_names:        List[str],
        base_fitted_params: Optional[Dict[str, float]] = None,
    ) -> ExtendedModelSpec:
        """
        Sparse identification of the correction term.

        Uses LASSO on a feature library of state/input functions to find a
        sparse symbolic expression that best explains the residual.

        Each LASSO-selected coefficient is promoted to a named NLS parameter
        (K_sindy_N) so the estimator can jointly re-optimise the correction
        alongside the base model parameters.  LASSO values serve as warm-start
        initial guesses.

        For sin/cos features an additional frequency parameter omega_sindy_N is
        introduced; for tanh features an epsilon scale parameter eps_sindy_N is
        added.  Both are bounded and warm-started at physically sensible values.
        """
        from tools.feature_library import FeatureLibrary

        if states is None or u is None or eps_ddot is None:
            return ExtendedModelSpec(
                normalized_rhs=base_rhs,
                fit_params=base_params,
                param_bounds=base_param_bounds,
                p0_override=dict(base_fitted_params) if base_fitted_params else {},
                strategy=Strategy.SINDY,
                correction_coeffs={},
            )

        lib = FeatureLibrary()
        valid = np.isfinite(eps_ddot)
        Theta, names = lib.build(states[:, valid], u[valid], state_names, input_names)

        # Exclude features already captured by base model parameters — otherwise
        # LASSO may absorb model structure errors into the "correction", biasing
        # base-param re-estimation.
        exclude = _base_model_feature_indices(base_rhs, names, state_names, input_names)

        coeffs = lib.sindy_fit(Theta, eps_ddot[valid], exclude_cols=exclude)

        # ── Build parameterized correction ────────────────────────────────────
        correction_terms: List[str]          = []
        new_params:       List[str]          = list(base_params)
        new_bounds:       Dict[str, Any]     = dict(base_param_bounds)
        p0:               Dict[str, float]   = dict(base_fitted_params) if base_fitted_params else {}
        active_coeffs:    Dict[str, float]   = {}

        sindy_idx = 0
        for c, name in zip(coeffs, names):
            if abs(c) < 1e-4:
                continue
            term, pnames, pbounds, pp0 = _parameterize_sindy_feature(name, float(c), sindy_idx)
            correction_terms.append(term)
            new_params.extend(pnames)
            new_bounds.update(pbounds)
            p0.update(pp0)
            active_coeffs[name] = float(c)
            sindy_idx += 1

        if not correction_terms:
            new_rhs = base_rhs
        else:
            correction_expr = " + ".join(correction_terms).replace("+ -", "- ")
            new_rhs = f"{base_rhs} + {correction_expr}"

        return ExtendedModelSpec(
            normalized_rhs=new_rhs,
            fit_params=new_params,
            param_bounds=new_bounds,
            p0_override=p0,
            strategy=Strategy.SINDY,
            correction_coeffs=active_coeffs,
        )

    # ── SINDy in output domain (no differentiation) ───────────────────────────

    @staticmethod
    def _fit_sindy_output_domain(
        base_params:       List[str],
        base_param_bounds: Dict[str, List[float]],
        states:            Optional[np.ndarray],
        u:                 Optional[np.ndarray],
        e_out:             Optional[np.ndarray],
        state_names:       List[str],
        input_names:       List[str],
        base_rhs:          str = "",
    ) -> "OutputCorrectionSpec":
        """
        Sparse identification of an output-space correction.

        Instead of fitting against acceleration residuals (ε_ddot = θ̈_meas - f_base),
        this fits against the output residual:

            ε_out[t] = y_meas[t] - y_base[t]

        which has noise level O(σ) rather than O(σ/Δt²).  No differentiation of
        the measured signal is involved.

        The correction is symbolic: ε_out ≈ Σ cⱼ · φⱼ(state_base, u).
        It is applied as: y_pred[t] = y_base[t] + Θ[t,:] · c
        and stored in the RESIDUAL_CORRECTED format alongside the base ODE.

        The features are evaluated on the base-ODE state estimates, not the true
        states (which are unknown).  This introduces approximation error when the
        base trajectory diverges from truth, but that error is bounded by the base
        ODE's own prediction error — typically much smaller than the differentiation
        noise that corrupts acceleration-domain SINDy.
        """
        from tools.feature_library import FeatureLibrary

        if states is None or u is None or e_out is None or len(e_out) < 10:
            return OutputCorrectionSpec(correction_coeffs={}, correction_expr="0")

        lib = FeatureLibrary()
        valid = np.isfinite(e_out)
        if valid.sum() < 10:
            return OutputCorrectionSpec(correction_coeffs={}, correction_expr="0")

        Theta, names = lib.build(states[:, valid], u[valid], state_names, input_names)

        # Same exclusion logic: skip features already in the base model RHS to
        # avoid absorbing base-param estimation errors into the correction.
        exclude = _base_model_feature_indices(base_rhs, names, state_names, input_names)

        coeffs = lib.sindy_fit(Theta, e_out[valid], exclude_cols=exclude)
        correction_expr = lib.build_symbolic_expression(coeffs, names)

        active_coeffs = {
            name: float(c)
            for c, name in zip(coeffs, names)
            if abs(c) > 1e-4
        }

        return OutputCorrectionSpec(
            correction_coeffs=active_coeffs,
            correction_expr=correction_expr,
        )

    # ── GP non-parametric correction ──────────────────────────────────────────

    @staticmethod
    def _fit_gp(
        base_rhs:          str,
        base_params:       List[str],
        base_param_bounds: Dict[str, List[float]],
        states:            Optional[np.ndarray],
        u:                 Optional[np.ndarray],
        eps_ddot:          Optional[np.ndarray],
        state_names:       List[str],
        input_names:       List[str],
    ) -> ExtendedModelSpec:
        """
        Fit a GP on the feature library of (states, u) → residual.

        The resulting correction_object is a picklable callable
        ``(x: ndarray, u_scalar: float) → float`` that is added to the ODE
        RHS at each integration step (via make_ode_simulator's correction_fn).
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        from tools.feature_library import FeatureLibrary
        from agents.surrogate.trainer import _NumpyGP

        if states is None or u is None or eps_ddot is None:
            return ExtendedModelSpec(
                normalized_rhs=base_rhs,
                fit_params=base_params,
                param_bounds=base_param_bounds,
                p0_override={},
                strategy=Strategy.GP_CORRECTION,
                correction_coeffs={},
                correction_object=None,
            )

        lib = FeatureLibrary()
        valid = np.isfinite(eps_ddot)
        Theta, _ = lib.build(states[:, valid], u[valid], state_names, input_names)
        eps_v    = eps_ddot[valid]

        _log.debug("GP _fit_gp: n_valid=%d, eps_std=%.4f, eps_range=[%.4f, %.4f]",
                   len(eps_v), float(np.std(eps_v)), float(eps_v.min()), float(eps_v.max()))

        MAX_GP = 400
        rng = np.random.default_rng(0)
        N = len(Theta)
        if N > MAX_GP:
            idx    = rng.choice(N, MAX_GP, replace=False)
            Tf, ef = Theta[idx], eps_v[idx]
        else:
            Tf, ef = Theta, eps_v

        _log.debug("GP _fit_gp: fitting on N=%d points, feature_dim=%d", len(Tf), Theta.shape[1])

        T_mean = Tf.mean(axis=0)
        T_std  = Tf.std(axis=0) + 1e-8
        Tn     = (Tf - T_mean) / T_std

        stride = max(1, len(Tn) // 50)
        Xs = Tn[::stride]
        d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
        dists = np.sqrt(d2[d2 > 0])
        ls = float(np.median(dists)) if len(dists) > 0 else 1.0

        _log.debug("GP _fit_gp: length_scale=%.4f, sigma_f=%.4f", ls, float(np.std(ef) + 1e-8))

        try:
            gp = _NumpyGP(length_scale=ls, sigma_f=float(np.std(ef) + 1e-8), sigma_n=0.05)
            gp.fit(Tn, ef)
            mu_check = gp.predict(Tn[:5])
            _log.debug("GP _fit_gp: fit OK, sample predictions=%s", mu_check)
        except Exception as _exc:
            _log.warning("GP _fit_gp: fit FAILED — %s: %s", type(_exc).__name__, _exc)
            return ExtendedModelSpec(
                normalized_rhs=base_rhs,
                fit_params=base_params,
                param_bounds=base_param_bounds,
                p0_override={},
                strategy=Strategy.GP_CORRECTION,
                correction_coeffs={},
                correction_object=None,
            )

        # GP leave-one-out RMSE (Rasmussen & Williams §5.4.2):
        # e_i = alpha_i / [K^{-1}]_{ii}  — exact, O(N), no ODE integration needed.
        try:
            loo_errors = gp._alpha / np.diag(gp._K_inv)
            gp_loo_rmse = float(np.sqrt(np.mean(loo_errors ** 2)))
        except Exception:
            gp_loo_rmse = float(np.std(ef))
        _log.debug("GP _fit_gp: LOO RMSE=%.4f", gp_loo_rmse)

        correction = _GPCorrection(lib, state_names, input_names, T_mean, T_std, gp)

        return ExtendedModelSpec(
            normalized_rhs=base_rhs,       # base RHS unchanged; correction via callable
            fit_params=base_params,
            param_bounds=base_param_bounds,
            p0_override={},
            strategy=Strategy.GP_CORRECTION,
            correction_coeffs={"gp_loo_rmse": gp_loo_rmse},
            correction_object=correction,
        )


# ── SINDY helpers ────────────────────────────────────────────────────────────

def _parameterize_sindy_feature(
    feature_name: str,
    coeff: float,
    idx: int,
) -> tuple:
    """
    Build a parameterized RHS term for one LASSO-selected feature.

    Returns (rhs_term, param_names, param_bounds, p0_values):
      - rhs_term     : SymPy-parseable string for the ODE RHS contribution
      - param_names  : list of NLS parameter names introduced
      - param_bounds : {name: [lo, hi]} for each new parameter
      - p0_values    : {name: initial_value} warm-started from LASSO

    sin/cos → adds a frequency parameter omega_sindy_N so NLS can tune the harmonic.
    tanh    → adds an epsilon scale parameter eps_sindy_N so NLS can tune the width.
    All others → coefficient K_sindy_N only.
    """
    k_name  = f"K_sindy_{idx}"
    k_bound = max(abs(coeff) * 20.0, 1.0)
    k_bounds = [-k_bound, k_bound]

    if feature_name.startswith("tanh(") and feature_name.endswith(")"):
        inner    = feature_name[5:-1]
        eps_name = f"eps_sindy_{idx}"
        return (
            f"{k_name}*tanh({inner}/{eps_name})",
            [k_name, eps_name],
            {k_name: k_bounds, eps_name: [1e-3, 10.0]},
            {k_name: coeff,    eps_name: 1.0},
        )

    if feature_name.startswith("sin(") and feature_name.endswith(")"):
        inner      = feature_name[4:-1]
        omega_name = f"omega_sindy_{idx}"
        return (
            f"{k_name}*sin({omega_name}*{inner})",
            [k_name, omega_name],
            {k_name: k_bounds, omega_name: [0.1, 10.0]},
            {k_name: coeff,    omega_name: 1.0},
        )

    if feature_name.startswith("cos(") and feature_name.endswith(")"):
        inner      = feature_name[4:-1]
        omega_name = f"omega_sindy_{idx}"
        return (
            f"{k_name}*cos({omega_name}*{inner})",
            [k_name, omega_name],
            {k_name: k_bounds, omega_name: [0.1, 10.0]},
            {k_name: coeff,    omega_name: 1.0},
        )

    return (
        f"{k_name}*{feature_name}",
        [k_name],
        {k_name: k_bounds},
        {k_name: coeff},
    )


def _base_model_feature_indices(
    base_rhs:    str,
    names:       List[str],
    state_names: List[str],
    input_names: List[str],
) -> List[int]:
    """
    Return column indices of features to exclude from the SINDY correction.

    Excludes:
    - The constant "1" — a bias shift corrupts steady-state
    - Linear state and input terms — already covered by fitted base-model params
    - Features containing any input variable — the base model's input gain(s)
      account for linear and cross-product input terms; including them would
      absorb K_in estimation errors into the correction
    - Nonlinear sub-expressions already present verbatim in the base RHS

    Result: the remaining features are pure-state nonlinear terms (e.g.,
    sign(x_dot), tanh(x_dot), x**2) that represent unmodeled dynamics.
    """
    exclude: List[int] = []
    base_lower = base_rhs.lower().replace(" ", "")
    simple = set(state_names) | set(input_names)
    for j, name in enumerate(names):
        if name == "1":
            exclude.append(j)
        elif name in simple:
            exclude.append(j)
        elif any(uname in name for uname in input_names):
            # cross-product / product involving input: already captured by model
            exclude.append(j)
        else:
            # exclude if the expression already appears verbatim in the base RHS
            name_lower = name.lower().replace(" ", "")
            if name_lower in base_lower:
                exclude.append(j)
    return exclude


# ── GP correction callable (picklable) ───────────────────────────────────────

class _GPCorrection:
    """
    Picklable callable that wraps a GP to produce additive ODE corrections.
    Interface: ``(x: ndarray, u_scalar: float) → float``
    """

    def __init__(
        self,
        lib:         object,          # FeatureLibrary instance
        state_names: List[str],
        input_names: List[str],
        T_mean:      np.ndarray,
        T_std:       np.ndarray,
        gp:          object,          # _NumpyGP instance
    ):
        self._lib         = lib
        self._state_names = state_names
        self._input_names = input_names
        self._T_mean      = T_mean
        self._T_std       = T_std
        self._gp          = gp

    def __call__(self, x: np.ndarray, u_val: float) -> float:
        x = np.asarray(x, dtype=float)
        states_pt = x.reshape(-1, 1)      # (system_order, 1)
        u_pt      = np.array([float(u_val)])
        Theta_pt, _ = self._lib.build(
            states_pt, u_pt, self._state_names, self._input_names
        )
        Theta_n = (Theta_pt - self._T_mean) / (self._T_std + 1e-8)
        return float(self._gp.predict(Theta_n)[0])
