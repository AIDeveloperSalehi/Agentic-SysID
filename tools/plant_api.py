"""
Guarded Plant API — the sole path to the physical plant.

Every input applied to the plant goes through here.  No agent can bypass it.

Responsibilities:
  1. Safety gate  — clip inputs to contract limits, enforce rate limits
  2. Budget debit — refuses to run if budget is exhausted
  3. DB write     — every run is appended to the experiment database
  4. Summary      — returns a compact summary suitable for an agent context window
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from core.schemas import (
    ExperimentRun,
    PlantContract,
    SafetyStatus,
    SplitFlag,
)
from plants.base_plant import BasePlant
from tools.budget_manager import BudgetExhaustedError, BudgetManager
from tools.experiment_db import ExperimentDatabase


class PlantAPI:
    """
    Wraps a BasePlant with safety, budget, and logging.

    Usage
    -----
    api = PlantAPI(plant, contract, budget_manager, db)
    result = api.apply_input(
        u_func       = lambda t: np.array([0.5]),
        t_span       = (0.0, 5.0),
        dt           = 0.02,
        purpose      = "identification",
        input_type   = "prbs",
        agent        = "estimator",
    )
    """

    def __init__(
        self,
        plant:          BasePlant,
        contract:       PlantContract,
        budget_manager: BudgetManager,
        db:             ExperimentDatabase,
        experiment_cost: float = 1.0,
    ):
        self._plant   = plant
        self._contract = contract
        self._budget  = budget_manager
        self._db      = db
        self._cost    = experiment_cost

    # ── Public interface ──────────────────────────────────────────────────────

    def apply_input(
        self,
        u_func:      Callable[[float], np.ndarray],
        t_span:      tuple[float, float],
        dt:          float,
        purpose:     str,
        input_type:  str,
        agent:       str,
        x0:          Optional[np.ndarray] = None,
        split_flag:  SplitFlag = SplitFlag.TRAIN,
        run_cost:    Optional[float] = None,
    ) -> dict:
        """
        Apply an input sequence to the plant with full safety and logging.

        Returns a compact result dict (not raw arrays) suitable for agent context:
            run_id, n_samples, safety_status, summary stats, cost_spent
        """
        cost = run_cost or self._cost

        # ── Budget check ──────────────────────────────────────────────────────
        if self._budget.check_stop(required=cost):
            raise BudgetExhaustedError(
                f"Cannot run experiment: only {self._budget.remaining:.2f} units left."
            )

        # ── Safety-gated input wrapper ────────────────────────────────────────
        safety_status = SafetyStatus.OK
        u_limits = self._contract.input_limits
        u_rate   = self._contract.input_rate_limits
        dt_ctrl  = dt

        _prev_u   = [None]   # mutable closure for rate limiting

        def _safe_u(t: float) -> np.ndarray:
            nonlocal safety_status
            raw = u_func(t)
            clipped = np.empty_like(raw)

            for i, name in enumerate(self._contract.input_names):
                lo, hi = u_limits.get(name, (-np.inf, np.inf))
                val = float(raw[i])

                # Rate limit
                if _prev_u[0] is not None and name in u_rate:
                    max_delta = u_rate[name] * dt_ctrl
                    val = float(np.clip(val, _prev_u[0][i] - max_delta,
                                              _prev_u[0][i] + max_delta))

                # Amplitude limit
                clipped_val = float(np.clip(val, lo, hi))
                if abs(clipped_val - float(raw[i])) > 1e-9:
                    safety_status = SafetyStatus.CLIPPED
                clipped[i] = clipped_val

            _prev_u[0] = clipped.copy()
            return clipped

        # ── Run simulation ────────────────────────────────────────────────────
        t, u, y = self._plant.apply_input(_safe_u, t_span, dt, x0=x0)

        # ── Debit budget ──────────────────────────────────────────────────────
        self._budget.debit(cost, description=f"{agent}/{purpose}")

        # ── Store run ─────────────────────────────────────────────────────────
        run = ExperimentRun(
            input_type=input_type,
            purpose=purpose,
            originating_agent=agent,
            cost=cost,
            safety_status=safety_status,
            split_flag=split_flag,
            n_samples=len(t),
        )
        self._db.store_run(run, t, u, y)

        # ── Compact summary ───────────────────────────────────────────────────
        return {
            "run_id":        run.id,
            "n_samples":     len(t),
            "t_span":        (float(t[0]), float(t[-1])),
            "safety_status": safety_status.value,
            "cost_spent":    cost,
            "budget_remaining": self._budget.remaining,
            "y_mean":        float(np.mean(y)),
            "y_std":         float(np.std(y)),
            "u_mean":        float(np.mean(u)),
            "u_std":         float(np.std(u)),
            "summary": (
                f"Run {run.id}: {purpose}/{input_type}, "
                f"{len(t)} samples over {t[-1]-t[0]:.1f}s, "
                f"safety={safety_status.value}, cost={cost:.1f}"
            ),
        }

    # ── Convenience: multi-segment experiment ─────────────────────────────────

    def apply_sequence(
        self,
        segments: list[dict],
        agent:    str,
        purpose:  str = "identification",
    ) -> list[dict]:
        """
        Apply multiple input segments in order, returning one result per segment.

        Each segment is a dict: {u_func, t_span, dt, input_type, ...}.
        Useful for step-response experiments with multiple operating points.
        """
        results = []
        for seg in segments:
            result = self.apply_input(
                u_func=seg["u_func"],
                t_span=seg["t_span"],
                dt=seg.get("dt", 0.02),
                purpose=purpose,
                input_type=seg.get("input_type", "designed"),
                agent=agent,
                split_flag=seg.get("split_flag", SplitFlag.TRAIN),
                run_cost=seg.get("cost", None),
            )
            results.append(result)
            if self._budget.exhausted:
                break
        return results
