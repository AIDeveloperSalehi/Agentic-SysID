"""
Data Steward — coverage and excitation quality service.

Invoked by Intake, Experiment-Design, Estimator, and the sub-orchestrators
before any new experiment is requested.  No LLM needed — pure DB queries
and spectral analysis.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np

from core.schemas import DataCoverageReport, ExcitationQuality, SplitFlag
from tools.experiment_db import ExperimentDatabase


class DataSteward:
    """
    Service agent for dataset coverage and quality assessment.

    Usage
    -----
    steward = DataSteward(db)
    report  = steward.assess_coverage(purpose="identification")
    """

    def __init__(self, db: ExperimentDatabase):
        self._db = db

    # ── Public interface ──────────────────────────────────────────────────────

    def assess_coverage(
        self,
        purpose: Optional[str] = None,
        split: Optional[SplitFlag] = None,
    ) -> DataCoverageReport:
        """
        Summarise what data already exists in the database.

        Parameters
        ----------
        purpose : str, optional
            Filter by experiment purpose ("identification", "validation", …).
        split : SplitFlag, optional
            Filter by train/validation split.

        Returns
        -------
        DataCoverageReport
        """
        runs = self._db.query_runs(purpose=purpose, split=split)

        if not runs:
            return DataCoverageReport(
                run_ids=[],
                coverage={},
                excitation=ExcitationQuality.INSUFFICIENT,
                quality=0.0,
                usable_for=[],
                gaps=["No experiments in database."],
            )

        run_ids = [r["id"] for r in runs]

        # Load all arrays and compute coverage from output ranges and input energy
        all_y: List[np.ndarray] = []
        all_u: List[np.ndarray] = []
        total_samples = 0

        for r in runs:
            try:
                t, u, y = self._db.load_arrays(r["id"])
                all_y.append(y[0])
                all_u.append(u[0])
                total_samples += len(t)
            except Exception:
                pass

        if not all_y:
            return DataCoverageReport(
                run_ids=run_ids,
                coverage={},
                excitation=ExcitationQuality.INSUFFICIENT,
                quality=0.0,
                usable_for=[],
                gaps=["Could not load array data."],
            )

        y_cat = np.concatenate(all_y)
        u_cat = np.concatenate(all_u)

        coverage = {
            "output": (float(y_cat.min()), float(y_cat.max())),
            "input":  (float(u_cat.min()), float(u_cat.max())),
        }

        # Excitation quality: check input spectral content
        excitation, quality = self._assess_excitation(u_cat)

        # What it's usable for
        usable_for: List[Literal["identify", "validate", "train"]] = []
        if excitation in (ExcitationQuality.SUFFICIENT,):
            usable_for.append("identify")
        if total_samples >= 100:
            usable_for.extend(["validate", "train"])

        # Gaps
        gaps: List[str] = []
        if excitation == ExcitationQuality.INSUFFICIENT:
            gaps.append("Insufficient input excitation for identification.")
        if total_samples < 200:
            gaps.append(f"Only {total_samples} samples; consider collecting more data.")

        return DataCoverageReport(
            run_ids=run_ids,
            coverage=coverage,
            excitation=excitation,
            quality=quality,
            usable_for=usable_for,
            gaps=gaps,
        )

    def has_identification_data(self) -> bool:
        """Return True if sufficient identification data already exists."""
        report = self.assess_coverage(purpose="identification")
        return "identify" in report.usable_for

    def list_run_ids(self, purpose: Optional[str] = None) -> List[str]:
        return self._db.all_run_ids(purpose=purpose)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _assess_excitation(u: np.ndarray) -> tuple:
        """
        Heuristic excitation quality check via input energy and variance.
        Returns (ExcitationQuality, quality_score [0-1]).
        """
        if len(u) < 20:
            return ExcitationQuality.INSUFFICIENT, 0.0

        # Spectral energy: fraction of energy in frequencies > 0 (non-DC)
        U = np.fft.rfft(u - u.mean())
        energy = float(np.sum(np.abs(U[1:]) ** 2))
        total  = float(np.sum(np.abs(np.fft.rfft(u)) ** 2)) + 1e-14
        spectral_fraction = energy / total

        # RMS relative to max absolute value
        rms = float(np.sqrt(np.mean(u ** 2)))
        max_val = float(np.max(np.abs(u))) + 1e-14
        amplitude_ratio = rms / max_val

        quality = float(np.clip(spectral_fraction * amplitude_ratio * 2.0, 0.0, 1.0))

        if spectral_fraction > 0.5 and rms > 1e-6:
            return ExcitationQuality.SUFFICIENT, quality
        elif rms > 1e-6:
            return ExcitationQuality.VALIDATION_ONLY, quality * 0.5
        else:
            return ExcitationQuality.INSUFFICIENT, 0.0
