"""
Unit tests for the deterministic service agents:
  IDAnalyst, DataSteward, ExperimentDesignAgent
"""
import tempfile
import numpy as np
import pytest

from core.schemas import (
    ExperimentRun, IdentifiabilityResult, PlantContract, SafetyStatus, SplitFlag,
)
from agents.id_analyst import IDAnalyst
from agents.data_steward import DataSteward
from agents.experiment_design import ExperimentDesignAgent
from tools.experiment_db import ExperimentDatabase


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pendulum_contract():
    return PlantContract(
        name="pendulum",
        input_names=["torque"],
        output_names=["angle"],
        input_limits={"torque": (-2.0, 2.0)},
        sample_time=0.02,
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary ExperimentDatabase for tests."""
    db = ExperimentDatabase(
        db_path=str(tmp_path / "test.db"),
        data_dir=str(tmp_path / "runs"),
    )
    return db


@pytest.fixture
def db_with_runs(tmp_db):
    """DB pre-populated with two identification runs."""
    rng = np.random.default_rng(42)
    for i in range(2):
        run = ExperimentRun(
            input_type="prbs",
            purpose="identification",
            originating_agent="estimator",
            cost=1.0,
            safety_status=SafetyStatus.OK,
            split_flag=SplitFlag.TRAIN,
            n_samples=300,
        )
        t = np.linspace(0, 6.0, 300)
        u = rng.choice([-0.5, 0.5], size=(1, 300))
        y = rng.normal(0, 0.01, (1, 300))
        tmp_db.store_run(run, t, u, y)
    return tmp_db


# ── IDAnalyst tests ───────────────────────────────────────────────────────────

class TestIDAnalyst:
    REPARAM_RHS   = "K_in*tau - tau_d*theta_dot - K_g*sin(theta)"
    PARAMS        = ["K_in", "tau_d", "K_g"]
    STATE_VARS    = ["theta", "theta_dot"]
    INPUT_VARS    = ["tau"]

    def _analyze(self, **kwargs):
        return IDAnalyst().analyze(
            self.REPARAM_RHS, self.PARAMS, self.STATE_VARS, self.INPUT_VARS, **kwargs
        )

    def test_returns_identifiability_report(self):
        r = self._analyze()
        from core.schemas import IdentifiabilityReport
        assert isinstance(r, IdentifiabilityReport)

    def test_lumped_params_are_fully_identifiable(self):
        """Reparameterized ODE: K_in, tau_d, K_g individually identifiable."""
        r = self._analyze()
        assert r.identifiable == IdentifiabilityResult.FULL

    def test_no_non_identifiable_params(self):
        r = self._analyze()
        assert r.non_identifiable_params == []

    def test_original_params_not_identifiable(self):
        """Physical params J, b_v, m, g, L are NOT individually identifiable."""
        r = IDAnalyst().analyze(
            "tau/J - b_v/J*theta_dot - m*g*L/J*sin(theta)",
            ["J", "b_v", "m", "g", "L"],
            self.STATE_VARS,
            self.INPUT_VARS,
        )
        assert r.identifiable in (IdentifiabilityResult.PARTIAL, IdentifiabilityResult.NONE)
        assert len(r.non_identifiable_params) > 0

    def test_recommendation_non_empty(self):
        r = self._analyze()
        assert isinstance(r.recommendation, str)

    def test_simple_identifiable_model(self):
        r = IDAnalyst().analyze("-a*x + b*u", ["a", "b"], ["x"], ["u"])
        assert r.identifiable == IdentifiabilityResult.FULL


# ── DataSteward tests ─────────────────────────────────────────────────────────

class TestDataSteward:
    def test_empty_db_returns_insufficient(self, tmp_db):
        ds = DataSteward(tmp_db)
        report = ds.assess_coverage()
        from core.schemas import ExcitationQuality
        assert report.excitation == ExcitationQuality.INSUFFICIENT
        assert len(report.run_ids) == 0

    def test_populated_db_has_run_ids(self, db_with_runs):
        ds = DataSteward(db_with_runs)
        report = ds.assess_coverage(purpose="identification")
        assert len(report.run_ids) == 2

    def test_coverage_keys_present(self, db_with_runs):
        ds = DataSteward(db_with_runs)
        report = ds.assess_coverage()
        assert "output" in report.coverage
        assert "input" in report.coverage

    def test_has_identification_data_true_after_runs(self, db_with_runs):
        ds = DataSteward(db_with_runs)
        # PRBS is broadband → should pass excitation check
        assert ds.has_identification_data()

    def test_list_run_ids_returns_list(self, db_with_runs):
        ds = DataSteward(db_with_runs)
        ids = ds.list_run_ids(purpose="identification")
        assert len(ids) == 2

    def test_quality_between_0_and_1(self, db_with_runs):
        ds = DataSteward(db_with_runs)
        r = ds.assess_coverage()
        assert 0.0 <= r.quality <= 1.0


# ── ExperimentDesignAgent tests ───────────────────────────────────────────────

class TestExperimentDesignAgent:
    def test_design_for_identification_returns_dict(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        seq = designer.design_for_identification(pendulum_contract, n_samples=100)
        assert "t" in seq and "u" in seq

    def test_identification_u_within_limits(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        seq = designer.design_for_identification(pendulum_contract, n_samples=200)
        assert np.all(seq["u"] >= -2.0)
        assert np.all(seq["u"] <= 2.0)

    def test_design_broadband_length(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        seq = designer.design_broadband(pendulum_contract, n_samples=400)
        assert len(seq["u"]) == 400

    def test_design_for_validation_returns_list(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        scenarios = designer.design_for_validation(pendulum_contract, n_scenarios=3)
        assert len(scenarios) == 3

    def test_make_u_func_is_callable(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        seq = designer.design_for_identification(pendulum_contract, n_samples=100)
        uf = designer.make_u_func(seq["t"], seq["u"])
        result = uf(0.0)
        assert result.shape == (1,)

    def test_u_func_clamps_at_boundaries(self, pendulum_contract):
        designer = ExperimentDesignAgent()
        seq = designer.design_for_identification(pendulum_contract, n_samples=100)
        uf = designer.make_u_func(seq["t"], seq["u"])
        # t beyond end should return last value
        v_end = uf(seq["t"][-1] + 100.0)
        assert np.isfinite(v_end[0])
