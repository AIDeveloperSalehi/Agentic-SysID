"""
Append-only experiment database.

Metadata lives in SQLite (fast queries, easy inspection).
Raw time-series arrays live in data/runs/{run_id}.npz (keeps blobs out of SQLite).

Discipline: every agent calls query() before requesting a new experiment.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sqlalchemy as sa
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.schemas import ExperimentRun, SplitFlag


# ── ORM ───────────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _RunRow(_Base):
    __tablename__ = "runs"

    id               = Column(String, primary_key=True)
    input_type       = Column(String, nullable=False)
    purpose          = Column(String, nullable=False)
    originating_agent = Column(String, nullable=False)
    cost             = Column(Float, default=1.0)
    safety_status    = Column(String, default="ok")
    split_flag       = Column(String, default="train")
    provenance       = Column(String, default="system")
    n_samples        = Column(Integer, default=0)
    npz_path         = Column(String)
    metadata_json    = Column(Text, default="{}")
    created_at       = Column(DateTime, default=datetime.utcnow)


# ── Database ──────────────────────────────────────────────────────────────────

class ExperimentDatabase:
    """
    Append-only store for all experiment runs.

    store_run   — write metadata + arrays to disk
    load_arrays — retrieve (t, u, y) for a run
    query_runs  — filter runs by purpose, split, agent, date range
    """

    def __init__(self, db_path: str = "data/experiment.db", data_dir: str = "data/runs"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        engine = sa.create_engine(f"sqlite:///{db_path}", echo=False)
        _Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_run(
        self,
        run: ExperimentRun,
        t: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> str:
        """
        Persist a run.  Returns run_id.

        Parameters
        ----------
        run : ExperimentRun metadata
        t   : (N,)           time vector
        u   : (n_inputs, N)  applied inputs
        y   : (n_outputs, N) measured outputs
        """
        npz_path = str(self._data_dir / f"{run.id}.npz")
        np.savez_compressed(npz_path, t=t, u=u, y=y)

        import json
        row = _RunRow(
            id=run.id,
            input_type=run.input_type,
            purpose=run.purpose,
            originating_agent=run.originating_agent,
            cost=run.cost,
            safety_status=run.safety_status.value,
            split_flag=run.split_flag.value,
            provenance=run.provenance,
            n_samples=len(t),
            npz_path=npz_path,
            metadata_json=json.dumps(run.metadata),
            created_at=run.created_at,
        )
        with self._Session() as session:
            session.add(row)
            session.commit()

        return run.id

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_arrays(self, run_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (t, u, y) for a stored run."""
        npz_path = self._data_dir / f"{run_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No arrays for run {run_id}")
        data = np.load(str(npz_path))
        return data["t"], data["u"], data["y"]

    def query_runs(
        self,
        purpose:   Optional[str] = None,
        split:     Optional[SplitFlag] = None,
        agent:     Optional[str] = None,
        since:     Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return list of run metadata dicts matching the filters."""
        import json
        with self._Session() as session:
            q = session.query(_RunRow)
            if purpose:
                q = q.filter(_RunRow.purpose == purpose)
            if split:
                q = q.filter(_RunRow.split_flag == split.value)
            if agent:
                q = q.filter(_RunRow.originating_agent == agent)
            if since:
                q = q.filter(_RunRow.created_at >= since)
            rows = q.order_by(_RunRow.created_at).all()

        return [
            {
                "id":                row.id,
                "input_type":        row.input_type,
                "purpose":           row.purpose,
                "originating_agent": row.originating_agent,
                "cost":              row.cost,
                "safety_status":     row.safety_status,
                "split_flag":        row.split_flag,
                "n_samples":         row.n_samples,
                "created_at":        row.created_at,
                "metadata":          json.loads(row.metadata_json or "{}"),
            }
            for row in rows
        ]

    def all_run_ids(self, purpose: Optional[str] = None) -> List[str]:
        with self._Session() as session:
            q = session.query(_RunRow.id)
            if purpose:
                q = q.filter(_RunRow.purpose == purpose)
            return [r[0] for r in q.all()]

    def total_cost(self) -> float:
        with self._Session() as session:
            result = session.query(sa.func.sum(_RunRow.cost)).scalar()
            return float(result or 0.0)

    def __len__(self) -> int:
        with self._Session() as session:
            return session.query(_RunRow).count()
