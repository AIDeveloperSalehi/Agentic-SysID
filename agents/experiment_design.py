"""
Experiment-Design service agent.

Wraps the experiment_design_toolkit with a clean interface:
  design_for_identification() — informative excitation
  design_for_validation()     — adversarial probing

Called synchronously by Estimator, Validation, and the sub-orchestrators.
Does NOT touch the plant — the calling agent applies the returned sequence.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.schemas import PlantContract
from tools.experiment_design_toolkit import (
    design_adversarial_inputs,
    design_compound_identification_input,
    design_identification_input,
)


class ExperimentDesignAgent:
    """
    Service agent for experiment design (no LLM, pure numerical).

    Usage
    -----
    designer = ExperimentDesignAgent()
    seq = designer.design_for_identification(contract, n_samples=500)
    # seq["u"] is a 1-D numpy array
    """

    def design_for_identification(
        self,
        contract: PlantContract,
        n_samples: int = 500,
        method: str = "prbs",
        seed: int = 0,
        amplitude_fraction: float = 0.70,
        frequency_range: Optional[tuple] = None,
    ) -> dict:
        """
        Create an informative input sequence for system identification.

        Returns dict with keys: t, u, input_type, description
        """
        return design_identification_input(
            contract,
            n_samples=n_samples,
            method=method,
            seed=seed,
            amplitude_fraction=amplitude_fraction,
            frequency_range=frequency_range,
        )

    def design_broadband(
        self,
        contract: PlantContract,
        n_samples: int = 800,
        seed: int = 0,
    ) -> dict:
        """Compound PRBS+multisine for maximum frequency coverage."""
        return design_compound_identification_input(contract, n_samples=n_samples, seed=seed)

    def design_for_validation(
        self,
        contract: PlantContract,
        n_samples_per_scenario: int = 300,
        n_scenarios: int = 3,
        seed: int = 42,
    ) -> List[dict]:
        """
        Create adversarial validation scenarios.

        Returns list of dicts {t, u, description, scenario_type}.
        """
        return design_adversarial_inputs(
            contract,
            n_samples_per_scenario=n_samples_per_scenario,
            n_scenarios=n_scenarios,
            seed=seed,
        )

    def make_u_func(self, t: np.ndarray, u: np.ndarray):
        """
        Return a callable u_func(ti) → np.ndarray suitable for PlantAPI.

        Uses linear interpolation and clamps to the array bounds.
        """
        def u_func(ti: float) -> np.ndarray:
            val = float(np.interp(ti, t, u,
                                  left=float(u[0]), right=float(u[-1])))
            return np.array([val])
        return u_func
