"""
Symbolic mathematics toolkit.

Structural identifiability analysis, reparameterization, and ODE manipulation.
All functions are deterministic; no LLM calls here.

Key entry point: check_structural_identifiability
  Given a normalized ODE and parameter names, returns which parameters are
  identifiable and suggests lumped reparameterizations.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp


# ── Namespace helpers ─────────────────────────────────────────────────────────

def _make_locals(
    params: List[str],
    state_vars: List[str],
    input_vars: List[str],
) -> Dict[str, sp.Basic]:
    """Build a SymPy parsing namespace from variable/parameter name lists."""
    locs: Dict[str, sp.Basic] = {}
    for name in params + state_vars + input_vars:
        locs[name] = sp.Symbol(name)
    locs.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh,
        "Abs": sp.Abs, "sign": sp.sign,
        "pi": sp.pi,
    })
    return locs


# ── Core analysis ─────────────────────────────────────────────────────────────

def find_lumped_parameters(
    rhs: str,
    params: List[str],
    state_vars: List[str],
    input_vars: List[str],
) -> Dict[str, sp.Expr]:
    """
    Extract identifiable parameter combinations from a normalized ODE RHS.

    The ODE is  ẋ_highest = rhs(states, inputs, params).
    Returns keys like "coeff_theta_dot", "coeff_sin_theta" mapped to the
    SymPy expression for each coefficient — these are the identifiable lumped
    parameters.

    Example
    -------
    rhs      = "tau/J - b_v/J*theta_dot - m*g*L/J*sin(theta)"
    params   = ["J", "b_v", "m", "g", "L"]
    → {"coeff_tau": 1/J,  "coeff_theta_dot": -b_v/J,  "coeff_sin_theta": -m*g*L/J}
    """
    locs = _make_locals(params, state_vars, input_vars)
    try:
        expr = sp.sympify(rhs, locals=locs)
    except Exception as exc:
        raise ValueError(f"Cannot parse ODE RHS '{rhs}': {exc}")

    lumped: Dict[str, sp.Expr] = {}

    # Linear terms (state vars and inputs)
    for vname in state_vars + input_vars:
        sym = locs[vname]
        c = expr.coeff(sym)
        if c != 0 and c is not sp.nan:
            lumped[f"coeff_{vname}"] = sp.simplify(c)

    # Nonlinear basis functions applied to state vars
    nonlinear = [
        (sp.sin,  "sin"),
        (sp.cos,  "cos"),
        (sp.exp,  "exp"),
        (sp.tanh, "tanh"),
    ]
    for vname in state_vars:
        sym = locs[vname]
        for func, fname in nonlinear:
            try:
                basis = func(sym)
                c = expr.coeff(basis)
                if c != 0 and c is not sp.nan:
                    lumped[f"coeff_{fname}_{vname}"] = sp.simplify(c)
            except Exception:
                pass

    return lumped


def check_structural_identifiability(
    normalized_ode_rhs: str,
    params: List[str],
    state_vars: List[str],
    input_vars: List[str],
    lumped_names: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Structural identifiability via ODE normalization.

    Determines which original parameters are individually identifiable from
    input-output data, and which only appear in inseparable combinations.

    Parameters
    ----------
    normalized_ode_rhs : str
        RHS of the highest-derivative equation.
    params : list[str]
        Unknown parameter names.
    state_vars : list[str]
        State variable names (e.g. ["theta", "theta_dot"]).
    input_vars : list[str]
        Input variable names (e.g. ["tau"]).
    lumped_names : dict, optional
        Human-readable overrides for lumped parameter keys.
        E.g. {"coeff_tau": "K_in", "coeff_theta_dot": "tau_d",
               "coeff_sin_theta": "K_g"}

    Returns
    -------
    dict
        identifiable:            "full" | "partial" | "none"
        non_identifiable_params: list[str]
        lumped_params:           dict {readable_name: str(expression)}
        recommendation:          str
    """
    lumped = find_lumped_parameters(normalized_ode_rhs, params, state_vars, input_vars)

    if not lumped:
        return {
            "identifiable": "none",
            "non_identifiable_params": list(params),
            "lumped_params": {},
            "recommendation": (
                "Could not extract parameter combinations. "
                "Verify that state/input variable names match the ODE."
            ),
        }

    locs = _make_locals(params, state_vars, input_vars)
    param_sym_set = {locs[p] for p in params}

    individually_identifiable: List[str] = []
    non_identifiable: List[str] = []

    for p in params:
        sym = locs[p]
        # Parameter is individually identifiable if it appears alone (no other
        # unknown parameter) in at least one lumped combination.
        solo = any(
            sym in expr.free_symbols
            and len(expr.free_symbols & param_sym_set) == 1
            for expr in lumped.values()
        )
        if solo:
            individually_identifiable.append(p)
        else:
            non_identifiable.append(p)

    # Build human-readable lumped param mapping
    lumped_readable: Dict[str, str] = {}
    for k, v in lumped.items():
        name = (lumped_names or {}).get(k, k)
        lumped_readable[name] = str(v)

    if not non_identifiable:
        id_status = "full"
        rec = "All parameters are individually identifiable from this output."
    elif len(non_identifiable) == len(params):
        id_status = "none"
        rec = (
            f"No individual parameter is identifiable. "
            f"Reparameterize with: {list(lumped_readable.keys())}"
        )
    else:
        id_status = "partial"
        rec = (
            f"Parameters {non_identifiable} are not individually identifiable. "
            f"Reparameterize using: {list(lumped_readable.keys())}"
        )

    return {
        "identifiable": id_status,
        "non_identifiable_params": non_identifiable,
        "lumped_params": lumped_readable,
        "recommendation": rec,
    }


def reparameterize_ode(
    normalized_ode_rhs: str,
    params: List[str],
    state_vars: List[str],
    input_vars: List[str],
    substitutions: Dict[str, str],
) -> str:
    """
    Rewrite the ODE RHS using new (lumped) parameter symbols.

    Parameters
    ----------
    substitutions : dict
        {new_symbol_name: expression_in_original_params}
        E.g. {"K_g": "m*g*L/J", "tau_d": "b_v/J", "K_in": "1/J"}

    Returns
    -------
    Simplified RHS string in terms of the new symbols.
    """
    locs = _make_locals(params, state_vars, input_vars)
    expr = sp.sympify(normalized_ode_rhs, locals=locs)

    # Apply substitutions from longest to shortest expression to avoid
    # partial-pattern collisions.
    for new_name, old_str in sorted(substitutions.items(), key=lambda x: -len(x[1])):
        old_expr = sp.sympify(old_str, locals=locs)
        new_sym = sp.Symbol(new_name)
        expr = expr.subs(old_expr, new_sym)

    return str(sp.simplify(expr))


# ── Numerical ODE factory ─────────────────────────────────────────────────────

def make_ode_simulator(
    normalized_ode_rhs: str,
    fit_params: List[str],
    state_vars: List[str],
    input_vars: List[str],
    highest_deriv_var: str,
    output_state_index: int = 0,
    correction_fn: Optional[Callable] = None,
    return_full_state: bool = False,
) -> Callable:
    """
    Compile the symbolic ODE RHS into a numerical simulator function.

    Supports scalar ODEs of any order n in companion form.  The system order
    is inferred from ``len(state_vars)``.  An optional ``correction_fn``
    (callable ``(x: ndarray, u: float) → float``) is added to the highest
    derivative at each RHS evaluation — used for GP grey-box corrections.

      order 1  state = [q]                  dynamics: [f_rhs]
      order 2  state = [q, q_dot]           dynamics: [q_dot, f_rhs]
      order 3  state = [q, q_dot, q_ddot]   dynamics: [q_dot, q_ddot, f_rhs]

    Parameters
    ----------
    state_vars : list[str]
        State variable names in ascending derivative order.
        ``len(state_vars)`` sets the system order.
    output_state_index : int
        Index of the state variable that is the measured output (default 0).
        E.g. set to 1 when only velocity is observed for a 2nd-order system.
    input_vars : list[str]
        Input variable names (single-input only).

    Returns
    -------
    simulator(param_values, t, u, x0=None) → y_pred  shape (N,)
    """
    _SENTINEL_RHS = {"RESIDUAL_CORRECTED", "SURROGATE", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"}
    if normalized_ode_rhs in _SENTINEL_RHS:
        raise ValueError(
            f"make_ode_simulator received sentinel RHS '{normalized_ode_rhs}'. "
            "Sentinel models cannot be compiled as symbolic ODEs."
        )

    locs = _make_locals(fit_params, state_vars, input_vars)
    expr = sp.sympify(normalized_ode_rhs, locals=locs)

    all_syms = [locs[n] for n in fit_params + state_vars + input_vars]
    # Detect free symbols in the parsed expression that are not in our variable list.
    # If any exist, lambdify will return a SymPy expression instead of a float,
    # producing a cryptic TypeError later inside solve_ivp.
    known = {s.name for s in all_syms}
    unknown = {str(s) for s in expr.free_symbols} - known
    if unknown:
        raise ValueError(
            f"make_ode_simulator: RHS contains unknown symbol(s) {unknown}. "
            f"Known: {sorted(known)}. RHS: {normalized_ode_rhs[:200]}"
        )

    f_rhs    = sp.lambdify(all_syms, expr, modules=["numpy"])
    n_states = len(state_vars)

    def simulator(
        param_values: np.ndarray,
        t: np.ndarray,
        u: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from scipy.integrate import solve_ivp

        if x0 is None:
            x0 = np.zeros(n_states)

        def rhs(ti, x):
            idx = int(np.clip(np.searchsorted(t, ti, side="right") - 1, 0, len(u) - 1))
            ui  = float(u[idx])
            args = list(param_values) + list(x) + [ui]
            xdot_highest = float(f_rhs(*args))
            if correction_fn is not None:
                xdot_highest += float(correction_fn(np.array(x), ui))
            # Companion form: [x[1], x[2], ..., x[n-1], f_rhs]
            # order-1 → [f_rhs], order-2 → [x[1], f_rhs], order-3 → [x[1], x[2], f_rhs]
            return [*x[1:], xdot_highest]

        # When a GP correction is active, cap max_step to the sample interval
        # and relax tolerances.  The GP itself is an approximation, so tight
        # ODE tolerances waste thousands of correction-function evaluations
        # without improving model quality.
        _dt = float(t[1] - t[0]) if len(t) > 1 else np.inf
        _rtol, _atol, _max_step = (
            (1e-4, 1e-6, _dt) if correction_fn is not None
            else (1e-8, 1e-10, np.inf)
        )
        sol = solve_ivp(
            rhs, (t[0], t[-1]), x0, t_eval=t,
            method="RK45", rtol=_rtol, atol=_atol,
            max_step=_max_step, dense_output=False,
        )
        if not sol.success:
            return (np.full((n_states, len(t)), np.nan)
                    if return_full_state else np.full(len(t), np.nan))
        return sol.y if return_full_state else sol.y[output_state_index]

    return simulator


def make_rk4_step(
    normalized_ode_rhs: str,
    fit_params: List[str],
    state_vars: List[str],
    input_vars: List[str],
) -> Callable:
    """
    Compile the symbolic ODE RHS into a fast batched RK4 one-step integrator.

    No scipy overhead — pure numpy.  Supports vectorized propagation of
    multiple state vectors simultaneously, which is used by the UKF to
    propagate all sigma points in a single call.

    Returns
    -------
    f_step(param_values, x, u, dt) → x_next
        param_values : (n_params,) current parameter vector
        x            : (n_states,) single state  OR  (n_states, M) batch
        u            : scalar input value at this step
        dt           : time step size
    """
    locs = _make_locals(fit_params, state_vars, input_vars)
    expr = sp.sympify(normalized_ode_rhs, locals=locs)
    all_syms = [locs[n] for n in fit_params + state_vars + input_vars]
    f_rhs = sp.lambdify(all_syms, expr, modules=["numpy"])
    n_states = len(state_vars)

    def _companion(pv: np.ndarray, x: np.ndarray, u: float) -> np.ndarray:
        # x[i] is either a scalar (single) or a 1-D array (batch column i)
        args = list(pv) + [x[i] for i in range(n_states)] + [float(u)]
        xd = np.asarray(f_rhs(*args), dtype=float)
        if n_states == 1:
            return xd.reshape(x.shape)
        # Companion form: ẋ = [x[1], ..., x[n-1], f_rhs]
        leading = x[1:]
        if x.ndim == 1:
            return np.concatenate([leading, xd.ravel()[:1]])
        return np.vstack([leading, xd.reshape(1, -1)])

    def f_step(pv: np.ndarray, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        k1 = _companion(pv, x, u)
        k2 = _companion(pv, x + dt / 2 * k1, u)
        k3 = _companion(pv, x + dt / 2 * k2, u)
        k4 = _companion(pv, x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return f_step


def evaluate_sensitivities(
    normalized_ode_rhs: str,
    params: List[str],
    state_vars: List[str],
    input_vars: List[str],
) -> Dict[str, str]:
    """
    Symbolic partial derivatives ∂(rhs)/∂pᵢ for each parameter.
    Useful for FIM construction in practical identifiability assessment.
    """
    locs = _make_locals(params, state_vars, input_vars)
    expr = sp.sympify(normalized_ode_rhs, locals=locs)

    return {
        p: str(sp.simplify(sp.diff(expr, locs[p])))
        for p in params
    }
