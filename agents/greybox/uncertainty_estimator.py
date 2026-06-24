"""
Uncertainty calibration for the grey-box corrected model.

COULOMB_TERM  — reuses the NLS Jacobian covariance from EstimatorAgent.
SINDY         — same as COULOMB_TERM (base params re-estimated via NLS).
GP_CORRECTION — same as COULOMB_TERM (base params unchanged, GP has
                its own internal uncertainty).
POLY_FALLBACK — bootstraps 200 resamples of (θ̇, ε_ddot) to get 95 % CIs
                on the polynomial coefficients; block-diagonal covariance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.greybox.strategy_selector import Strategy


class UncertaintyEstimator:
    """
    Produces a covariance matrix and 95 % confidence intervals for the corrected model.
    """

    def estimate(
        self,
        strategy:       Strategy,
        fit_params:     List[str],
        nls_covariance: Optional[np.ndarray],     # (n_params × n_params) from EstimatorAgent
        t:              np.ndarray,
        theta_dot:      np.ndarray,
        eps_ddot:       np.ndarray,
        n_bootstrap:    int = 200,
        seed:           int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Returns
        -------
        covariance_matrix      : (n_params × n_params) ndarray
        confidence_intervals   : {param_name: (lo_95, hi_95)}
        """
        n = len(fit_params)

        if strategy == Strategy.POLY_FALLBACK:
            return self._from_bootstrap(fit_params, nls_covariance,
                                        theta_dot, eps_ddot,
                                        n_bootstrap, seed)
        else:
            # COULOMB_TERM, SINDY, GP_CORRECTION: use NLS covariance for base params
            return self._from_nls_cov(fit_params, nls_covariance)

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _from_nls_cov(
        fit_params:     List[str],
        nls_covariance: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        n = len(fit_params)

        if nls_covariance is not None and nls_covariance.shape == (n, n):
            cov = nls_covariance.copy()
        else:
            cov = np.diag([999.0] * n)

        diag = np.diag(cov)
        ci: Dict[str, Tuple[float, float]] = {}
        for i, p in enumerate(fit_params):
            std = float(np.sqrt(max(diag[i], 0.0)))
            ci[p] = (-1.96 * std, 1.96 * std)   # relative to fitted value

        return cov, ci

    @staticmethod
    def _from_bootstrap(
        fit_params:     List[str],
        nls_covariance: Optional[np.ndarray],
        theta_dot:      np.ndarray,
        eps_ddot:       np.ndarray,
        n_bootstrap:    int,
        seed:           int,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        rng   = np.random.default_rng(seed)
        valid = np.isfinite(eps_ddot) & np.isfinite(theta_dot)
        td_v  = theta_dot[valid]
        ep_v  = eps_ddot[valid]
        M     = len(td_v)

        a1_samples: list[float] = []
        a3_samples: list[float] = []

        if M >= 4:
            Phi = np.column_stack([td_v, td_v ** 3])
            for _ in range(n_bootstrap):
                idx = rng.integers(0, M, M)
                try:
                    c, _, _, _ = np.linalg.lstsq(Phi[idx], ep_v[idx], rcond=None)
                    a1_samples.append(float(c[0]))
                    a3_samples.append(float(c[1]))
                except np.linalg.LinAlgError:
                    pass

        n = len(fit_params)

        # Build block-diagonal: base params from NLS cov, poly params from bootstrap
        n_base = n - 2   # last two are a1, a3
        base_cov = (
            nls_covariance[:n_base, :n_base]
            if nls_covariance is not None and nls_covariance.shape[0] >= n_base
            else np.diag([999.0] * n_base)
        )
        poly_var = np.zeros((2, 2))
        if a1_samples:
            poly_var[0, 0] = float(np.var(a1_samples, ddof=1))
            poly_var[1, 1] = float(np.var(a3_samples, ddof=1))
            poly_var[0, 1] = poly_var[1, 0] = float(
                np.cov(a1_samples, a3_samples)[0, 1]
            )

        cov = np.zeros((n, n))
        cov[:n_base, :n_base] = base_cov
        cov[n_base:, n_base:] = poly_var

        ci: Dict[str, Tuple[float, float]] = {}
        for i, p in enumerate(fit_params):
            std = float(np.sqrt(max(cov[i, i], 0.0)))
            ci[p] = (-1.96 * std, 1.96 * std)

        if a1_samples:
            ci["a1"] = (
                float(np.percentile(a1_samples, 2.5)),
                float(np.percentile(a1_samples, 97.5)),
            )
        if a3_samples:
            ci["a3"] = (
                float(np.percentile(a3_samples, 2.5)),
                float(np.percentile(a3_samples, 97.5)),
            )

        return cov, ci
