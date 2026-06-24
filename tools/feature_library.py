"""
General feature library for grey-box residual identification (SINDy-style).

Builds a dictionary of candidate basis functions evaluated on state/input data,
computes correlations with a residual signal, and finds a sparse linear
combination (LASSO) that best explains the residual.

No physics assumptions are embedded here — the data decides which basis
functions matter.  The resulting symbolic expression uses the actual state and
input variable names so it can be appended directly to any ODE RHS string.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

MAX_CANDIDATE_FEATURES = 25   # cap library size for computational reasons


class FeatureLibrary:
    """
    Builds and evaluates a dictionary of candidate basis functions.

    Features include: constant, linear, quadratic, cross-products, state×input,
    and nonlinear functions (sin, cos, Abs, sign, tanh) of each state variable.
    """

    def build(
        self,
        states:      np.ndarray,   # (system_order, N) — rows are state variables
        u:           np.ndarray,   # (N,)
        state_names: List[str],    # variable names for rows of states
        input_names: List[str],    # variable names for u
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build (Theta, feature_names) where Theta is (N, n_features).

        Variable names in feature_names match the original ODE variable names so
        that the resulting symbolic expression can be appended to the ODE RHS
        and parsed by SymPy without additional renaming.
        """
        N = states.shape[1] if states.ndim == 2 else len(states)
        u_arr  = np.asarray(u, dtype=float).ravel()
        u_name = input_names[0] if input_names else "u"

        cols: List[np.ndarray] = []
        names: List[str]       = []

        def _add(col: np.ndarray, name: str) -> None:
            col = np.asarray(col, dtype=float).ravel()
            cols.append(np.where(np.isfinite(col), col, 0.0))
            names.append(name)

        # ── Constant ─────────────────────────────────────────────────────────
        _add(np.ones(N), "1")

        # ── Linear state terms ────────────────────────────────────────────────
        for i, vname in enumerate(state_names):
            _add(states[i], vname)

        # ── Input ─────────────────────────────────────────────────────────────
        _add(u_arr, u_name)

        # ── Quadratic state terms ─────────────────────────────────────────────
        for i, vname in enumerate(state_names):
            _add(states[i] ** 2, f"{vname}**2")

        # ── State cross products ──────────────────────────────────────────────
        n_sv = len(state_names)
        for i in range(n_sv):
            for j in range(i + 1, n_sv):
                _add(states[i] * states[j],
                     f"{state_names[i]}*{state_names[j]}")

        # ── State × input ─────────────────────────────────────────────────────
        for i, vname in enumerate(state_names):
            _add(states[i] * u_arr, f"{vname}*{u_name}")

        # ── Nonlinear functions of each state ─────────────────────────────────
        # Names use SymPy-parseable forms: Abs (not abs), sign, tanh, sin, cos
        for i, vname in enumerate(state_names):
            xi = states[i]
            _add(np.sin(xi),  f"sin({vname})")
            _add(np.cos(xi),  f"cos({vname})")
            _add(np.abs(xi),  f"Abs({vname})")
            _add(np.sign(xi), f"sign({vname})")
            _add(np.tanh(xi), f"tanh({vname})")

        # Cap library size
        cols  = cols[:MAX_CANDIDATE_FEATURES]
        names = names[:MAX_CANDIDATE_FEATURES]

        Theta = np.column_stack(cols)
        return Theta, names

    # ── Correlation diagnosis ─────────────────────────────────────────────────

    def feature_correlations(
        self,
        Theta:    np.ndarray,    # (N, n_features)
        names:    List[str],
        residual: np.ndarray,    # (N,)
    ) -> Dict[str, float]:
        """Return |Pearson correlation| of each feature column with the residual."""
        r     = residual - residual.mean()
        r_std = np.std(r) + 1e-12
        corrs: Dict[str, float] = {}
        for j, name in enumerate(names):
            f     = Theta[:, j] - Theta[:, j].mean()
            f_std = np.std(f) + 1e-12
            corrs[name] = float(np.abs(np.mean(f * r)) / (f_std * r_std))
        return corrs

    # ── Sparse regression (SINDy-style LASSO) ────────────────────────────────

    def sindy_fit(
        self,
        Theta:         np.ndarray,         # (N, n_features)
        residual:      np.ndarray,         # (N,)
        alpha:         float = 5e-4,
        threshold:     float = 1e-2,
        exclude_cols:  "Optional[List[int]]" = None,
    ) -> np.ndarray:
        """
        Sparse regression: c such that Theta @ c ≈ residual.

        Feature columns are normalized to unit std before LASSO so that the
        sparsity penalty is applied equally regardless of feature scale.
        Coefficients are denormalized before returning.
        Falls back to OLS if sklearn is unavailable.

        exclude_cols: column indices to force to zero (e.g. features already in
                      the base model, to avoid double-counting).
        """
        n_feat = Theta.shape[1]
        full_coeffs = np.zeros(n_feat)

        # Mask: work only on non-excluded columns
        keep = np.ones(n_feat, dtype=bool)
        if exclude_cols:
            for j in exclude_cols:
                if 0 <= j < n_feat:
                    keep[j] = False

        if keep.sum() == 0:
            return full_coeffs

        Th = Theta[:, keep]

        # Normalize features column-wise for stable LASSO regularization
        col_std = np.std(Th, axis=0)
        col_std = np.where(col_std < 1e-8, 1.0, col_std)
        Th_n = Th / col_std

        res_std = float(np.std(residual)) or 1.0
        residual_n = residual / res_std

        try:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-5)
            model.fit(Th_n, residual_n)
            coeffs_n = model.coef_.copy()
        except ImportError:
            coeffs_n, _, _, _ = np.linalg.lstsq(Th_n, residual_n, rcond=None)

        # Denormalize: c_original = c_normalized * res_std / col_std
        coeffs = coeffs_n * res_std / col_std
        coeffs[np.abs(coeffs) < threshold] = 0.0

        full_coeffs[keep] = coeffs
        return full_coeffs

    def build_symbolic_expression(
        self,
        coeffs:        np.ndarray,
        feature_names: List[str],
        tol:           float = 1e-4,
    ) -> str:
        """
        Build a SymPy-parseable expression string from non-zero LASSO coefficients.

        Returns "0" if all coefficients are zero.
        """
        terms: List[str] = []
        for c, name in zip(coeffs, feature_names):
            if abs(c) < tol:
                continue
            terms.append(f"{c:.6f}" if name == "1" else f"{c:.6f}*{name}")

        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr
