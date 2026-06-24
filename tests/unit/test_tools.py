"""Tests for tools/budget_manager.py, experiment_db.py, model_registry.py."""
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.schemas import (
    ExperimentRun,
    ModelArtifact,
    ModelType,
    SafetyStatus,
    SplitFlag,
    ValidityRegion,
)
from tools.budget_manager import BudgetExhaustedError, BudgetManager
from tools.experiment_db import ExperimentDatabase
from tools.model_registry import ModelRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def budget():
    return BudgetManager(total=100.0)


@pytest.fixture
def db(tmp_path):
    return ExperimentDatabase(
        db_path=str(tmp_path / "test.db"),
        data_dir=str(tmp_path / "runs"),
    )


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(base_dir=str(tmp_path / "models"))


def _make_run(**kwargs) -> ExperimentRun:
    defaults = dict(input_type="prbs", purpose="identification", originating_agent="estimator")
    defaults.update(kwargs)
    return ExperimentRun(**defaults)


def _make_arrays(N=50):
    t = np.linspace(0, 1, N)
    u = np.random.randn(1, N)
    y = np.random.randn(1, N)
    return t, u, y


# ── BudgetManager ─────────────────────────────────────────────────────────────

class TestBudgetManager:
    def test_initial_remaining(self, budget):
        assert budget.remaining == pytest.approx(100.0)

    def test_debit(self, budget):
        budget.debit(30.0)
        assert budget.remaining == pytest.approx(70.0)

    def test_debit_accumulates(self, budget):
        budget.debit(10.0)
        budget.debit(20.0)
        assert budget.spent == pytest.approx(30.0)

    def test_raises_on_overdraft(self, budget):
        budget.debit(99.0)
        with pytest.raises(BudgetExhaustedError):
            budget.debit(2.0)

    def test_try_debit_returns_none_on_overdraft(self, budget):
        budget.debit(100.0)
        result = budget.try_debit(1.0)
        assert result is None

    def test_check_stop(self, budget):
        assert not budget.check_stop()
        budget.debit(99.5)
        assert budget.check_stop()

    def test_allocate_slice(self, budget):
        budget.allocate_slice("greybox", 40.0)
        assert budget.budget.slice_allocations["greybox"] == 40.0

    def test_summary_string(self, budget):
        budget.debit(25.0)
        s = budget.summary()
        assert "25.0" in s
        assert "100.0" in s

    def test_exhausted_flag(self, budget):
        assert not budget.exhausted
        budget.debit(100.0)
        assert budget.exhausted


# ── ExperimentDatabase ────────────────────────────────────────────────────────

class TestExperimentDatabase:
    def test_store_and_load(self, db):
        run = _make_run()
        t, u, y = _make_arrays()
        db.store_run(run, t, u, y)
        t2, u2, y2 = db.load_arrays(run.id)
        np.testing.assert_allclose(t, t2)
        np.testing.assert_allclose(u, u2)
        np.testing.assert_allclose(y, y2)

    def test_len(self, db):
        assert len(db) == 0
        db.store_run(_make_run(), *_make_arrays())
        assert len(db) == 1
        db.store_run(_make_run(), *_make_arrays())
        assert len(db) == 2

    def test_query_by_purpose(self, db):
        db.store_run(_make_run(purpose="identification"), *_make_arrays())
        db.store_run(_make_run(purpose="validation"),    *_make_arrays())
        db.store_run(_make_run(purpose="validation"),    *_make_arrays())

        id_runs  = db.query_runs(purpose="identification")
        val_runs = db.query_runs(purpose="validation")
        assert len(id_runs)  == 1
        assert len(val_runs) == 2

    def test_query_by_split(self, db):
        db.store_run(_make_run(split_flag=SplitFlag.TRAIN),      *_make_arrays())
        db.store_run(_make_run(split_flag=SplitFlag.VALIDATION), *_make_arrays())
        train = db.query_runs(split=SplitFlag.TRAIN)
        val   = db.query_runs(split=SplitFlag.VALIDATION)
        assert len(train) == 1
        assert len(val)   == 1

    def test_all_run_ids(self, db):
        for _ in range(3):
            db.store_run(_make_run(), *_make_arrays())
        ids = db.all_run_ids()
        assert len(ids) == 3

    def test_total_cost(self, db):
        db.store_run(_make_run(cost=2.5), *_make_arrays())
        db.store_run(_make_run(cost=1.5), *_make_arrays())
        assert db.total_cost() == pytest.approx(4.0)

    def test_missing_run_raises(self, db):
        with pytest.raises(FileNotFoundError):
            db.load_arrays("nonexistent_id")

    def test_n_samples_stored(self, db):
        run = _make_run()
        t, u, y = _make_arrays(N=80)
        db.store_run(run, t, u, y)
        rows = db.query_runs()
        assert rows[0]["n_samples"] == 80


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    def _make_model(self, model_type=ModelType.WHITE_BOX) -> ModelArtifact:
        return ModelArtifact(
            model_type=model_type,
            structure_description="test structure",
            parameters={"K_g": 29.4, "tau_d": 0.4},
        )

    def test_store_and_load_model(self, registry):
        m = self._make_model()
        mid = registry.store_model(m)
        m2  = registry.load_model(mid)
        assert m2.id == m.id
        assert m2.parameters["K_g"] == pytest.approx(29.4)

    def test_load_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load_model("does_not_exist")

    def test_list_models_empty(self, registry):
        assert registry.list_models() == []

    def test_list_models_filtered(self, registry):
        registry.store_model(self._make_model(ModelType.WHITE_BOX))
        registry.store_model(self._make_model(ModelType.GREY_BOX))
        registry.store_model(self._make_model(ModelType.WHITE_BOX))
        wb = registry.list_models(model_type=ModelType.WHITE_BOX)
        gb = registry.list_models(model_type=ModelType.GREY_BOX)
        assert len(wb) == 2
        assert len(gb) == 1

    def test_latest_model(self, registry):
        m1 = self._make_model(); registry.store_model(m1)
        m2 = self._make_model(); registry.store_model(m2)
        latest = registry.latest_model()
        assert latest.id == m2.id   # m2 created after m1

    def test_store_and_load_covariance(self, registry):
        cov = np.array([[1.0, 0.2], [0.2, 0.5]])
        registry.store_covariance("cov_01", cov)
        cov2 = registry.load_covariance("cov_01")
        np.testing.assert_allclose(cov, cov2)

    def test_store_and_load_object(self, registry):
        obj = {"key": "value", "arr": [1, 2, 3]}
        registry.store_object("obj_01", obj)
        obj2 = registry.load_object("obj_01")
        assert obj2 == obj

    def test_store_and_load_validity(self, registry):
        vr = ValidityRegion(
            model_id="m1",
            bounds={"theta": (-1.5, 1.5)},
            tolerance=0.05,
            achieved_rmse=0.03,
            coverage_fraction=0.95,
        )
        registry.store_validity(vr)
        vr2 = registry.load_validity(vr.id)
        assert vr2.model_id == "m1"
        assert vr2.achieved_rmse == pytest.approx(0.03)

    def test_summary(self, registry):
        registry.store_model(self._make_model())
        s = registry.summary()
        assert s["models"] == 1
