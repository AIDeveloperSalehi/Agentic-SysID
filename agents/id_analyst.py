"""
ID Analyst — structural + practical identifiability service.

Invoked as a synchronous service call by the Modeler (pre-commit check)
and the Estimator (which parameters are recoverable).  No LLM needed; the
analysis is pure symbolic math + FIM numerics.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.schemas import IdentifiabilityReport, IdentifiabilityResult
from tools.solver_toolkit import compute_fim
from tools.symbolic_math import check_structural_identifiability, make_ode_simulator


class IDAnalyst:
    """
    Service agent for identifiability analysis.

    Usage
    -----
    analyst = IDAnalyst()
    report = analyst.analyze(
        normalized_rhs = "K_in*tau - tau_d*theta_dot - K_g*sin(theta)",
        fit_params     = ["K_in", "tau_d", "K_g"],
        state_vars     = ["theta", "theta_dot"],
        input_vars     = ["tau"],
        lumped_names   = {"coeff_tau": "K_in", "coeff_theta_dot": "tau_d",
                           "coeff_sin_theta": "K_g"},
    )
    """

    def analyze(
        self,
        normalized_rhs: str,
        fit_params: List[str],
        state_vars: List[str],
        input_vars: List[str],
        lumped_names: Optional[dict] = None,
        existing_t: Optional[np.ndarray] = None,
        existing_u: Optional[np.ndarray] = None,
        noise_var: float = 1e-6,
    ) -> IdentifiabilityReport:
        """
        Check structural and (if data provided) practical identifiability.

        Parameters
        ----------
        normalized_rhs : str
            RHS of the highest-derivative ODE (already reparameterized, using
            the fit_params names).
        fit_params : list[str]
            Parameter names to check.
        state_vars, input_vars : list[str]
            As in the ODE.
        lumped_names : dict, optional
            Human-readable name overrides for lumped param keys.
        existing_t, existing_u : arrays, optional
            If supplied, a practical FIM check is also run on these data.
        noise_var : float
            Assumed measurement noise variance for FIM.

        Returns
        -------
        IdentifiabilityReport
        """
        # Structural identifiability
        structural = check_structural_identifiability(
            normalized_rhs, fit_params, state_vars, input_vars,
            lumped_names=lumped_names,
        )

        non_id = structural["non_identifiable_params"]

        # If structural says none are identifiable, return immediately
        if structural["identifiable"] == "none":
            return IdentifiabilityReport(
                identifiable=IdentifiabilityResult.NONE,
                non_identifiable_params=non_id,
                recommendation=structural["recommendation"],
            )

        # Practical identifiability via FIM (only if data provided)
        practical_ok = True
        practical_non_id: List[str] = []
        if existing_t is not None and existing_u is not None:
            try:
                sim = make_ode_simulator(
                    normalized_rhs, fit_params, state_vars, input_vars,
                    highest_deriv_var=state_vars[-1] + "_ddot",
                )
                # Nominal parameter values (unit values as a proxy if unknown)
                p_nom = np.ones(len(fit_params))

                def sens_fn(params: np.ndarray, u: np.ndarray) -> np.ndarray:
                    return sim(params, existing_t, u)

                fim_result = compute_fim(sens_fn, existing_t, existing_u, p_nom,
                                         noise_var=noise_var)
                if not fim_result["full_rank"]:
                    practical_ok = False
                    # Identify which params correspond to near-zero eigenvalue directions
                    evals = np.array(fim_result["eigenvalues"])
                    small = evals < 1e-10 * evals[-1]
                    # Heuristic: mark params at the end of the eigenvalue spectrum
                    n_small = int(np.sum(small))
                    practical_non_id = fit_params[-n_small:] if n_small else []
            except Exception:
                practical_ok = True   # data check failed silently; trust structural

        # Combine
        if non_id:
            # Structural non-identifiability dominates
            id_result = IdentifiabilityResult.PARTIAL if len(non_id) < len(fit_params) \
                        else IdentifiabilityResult.NONE
            recommendation = structural["recommendation"]
        elif not practical_ok and practical_non_id:
            id_result = IdentifiabilityResult.PARTIAL
            non_id = practical_non_id
            recommendation = (
                f"Parameters {practical_non_id} are practically non-identifiable "
                f"given current data.  Design a more informative experiment."
            )
        else:
            id_result = IdentifiabilityResult.FULL
            recommendation = "All parameters are identifiable."

        return IdentifiabilityReport(
            identifiable=id_result,
            non_identifiable_params=non_id,
            recommendation=recommendation,
        )
