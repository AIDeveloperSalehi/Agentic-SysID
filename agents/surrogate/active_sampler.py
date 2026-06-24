"""
Active sampler for the surrogate sub-orchestrator.

When the available training set is smaller than MIN_SAMPLES_FOR_SURROGATE,
runs additional space-filling PRBS experiments at different seeds to broaden
state-space coverage before fitting the surrogate.
"""
from __future__ import annotations

import logging
from typing import List

from core.schemas import PlantContract, SplitFlag
from agents.experiment_design import ExperimentDesignAgent
from tools.plant_api import PlantAPI
from tools.experiment_db import ExperimentDatabase

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_SURROGATE = 300
MAX_ACTIVE_EXPERIMENTS    = 3


class ActiveSampler:
    """
    Collects additional PRBS data when coverage is insufficient for surrogate fitting.
    """

    def __init__(self, plant_api: PlantAPI, db: ExperimentDatabase):
        self._api      = plant_api
        self._db       = db
        self._designer = ExperimentDesignAgent()

    def maybe_collect(
        self,
        n_available:      int,
        contract:         PlantContract,
        existing_run_ids: List[str],
        seed_offset:      int = 200,
        force_collect:    bool = False,
    ) -> List[str]:
        """
        Returns a list of new run_ids (may be empty if data was already sufficient).

        Set force_collect=True to always collect new data regardless of n_available
        (used on retry when previous surrogate did not improve).
        """
        if not force_collect and n_available >= MIN_SAMPLES_FOR_SURROGATE:
            return []

        new_ids: List[str] = []
        n_needed = min(
            MAX_ACTIVE_EXPERIMENTS,
            max(1, MAX_ACTIVE_EXPERIMENTS - len(existing_run_ids)),
        )

        for i in range(n_needed):
            try:
                seq    = self._designer.design_for_identification(
                    contract, n_samples=300, method="prbs", seed=seed_offset + i
                )
                u_func = self._designer.make_u_func(seq["t"], seq["u"])
                result = self._api.apply_input(
                    u_func=u_func,
                    t_span=(float(seq["t"][0]), float(seq["t"][-1])),
                    dt=float(seq["t"][1] - seq["t"][0]),
                    purpose="identification",
                    input_type=seq["input_type"],
                    agent="surrogate_active_sampler",
                    split_flag=SplitFlag.TRAIN,
                )
                new_ids.append(result["run_id"])
                logger.info("ActiveSampler: collected run_id=%s", result["run_id"])
            except Exception as exc:
                logger.warning("ActiveSampler: experiment %d failed: %s", i, exc)

        return new_ids
