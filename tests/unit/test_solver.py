"""
Tests for tools/solver_toolkit.py and tools/plant_api.py.
"""
import numpy as np
import pytest

from tools.solver_toolkit import (
    compute_fim,
    compute_metrics,
    fit_arx,
    generate_chirp,
    generate_multisine,
    generate_prbs,
    generate_steps,
    nonlinear_least_squares,
    residual_whiteness_test,
    simulate_ode,
)


# ── Input generators ──────────────────────────────────────────────────────────

class TestInputGenerators:
    def test_prbs_shape(self):
        t, u = generate_prbs(200, dt=0.02)
        assert len(t) == 200
        assert len(u) == 200

    def test_prbs_binary(self):
        _, u = generate_prbs(500, dt=0.02, amplitude=1.5)
        unique = set(np.round(u, 6))
        assert unique == {-1.5, 1.5}

    def test_multisine_shape(self):
        t, u = generate_multisine(200, dt=0.02, frequencies=[1.0, 2.0, 3.0])
        assert len(t) == 200

    def test_multisine_amplitude(self):
        _, u = generate_multisine(1000, dt=0.01, frequencies=[0.5, 1.0], amplitude=2.0)
        assert abs(np.max(np.abs(u)) - 2.0) < 0.01

    def test_chirp_shape(self):
        t, u = generate_chirp(500, dt=0.02, f0=0.1, f1=5.0)
        assert len(t) == 500

    def test_steps_shape(self):
        t, u = generate_steps([0.0, 0.5, 1.0, -0.5], hold_time=2.0, dt=0.02)
        expected = 4 * int(2.0 / 0.02)
        assert len(u) == expected


# ── ODE simulation ────────────────────────────────────────────────────────────

class TestSimulateODE:
    def test_double_integrator(self):
        """ẋ₁=x₂, ẋ₂=u → position should increase under positive constant input."""
        def f(t, x, u):
            return np.array([x[1], u[0]])
        t  = np.linspace(0, 2, 100)
        u  = np.ones((1, 100)) * 0.5
        x, y = simulate_ode(f, t, u, x0=np.array([0.0, 0.0]))
        assert x[0, -1] > 0.5    # position grew
        assert x[1, -1] > 0.0    # velocity grew

    def test_stable_first_order(self):
        """ẋ = -x + u, u=0 → exponential decay from x0=1."""
        def f(t, x, u): return np.array([-x[0] + u[0]])
        t = np.linspace(0, 5, 200)
        u = np.zeros((1, 200))
        x, _ = simulate_ode(f, t, u, x0=np.array([1.0]))
        assert x[0, -1] < 0.05    # decayed to near zero

    def test_output_matrix(self):
        """C selects only the second state."""
        def f(t, x, u): return np.array([x[1], -x[0]])
        t  = np.linspace(0, 1, 50)
        u  = np.zeros((1, 50))
        C  = np.array([[0.0, 1.0]])
        x, y = simulate_ode(f, t, u, x0=np.array([1.0, 0.0]), C=C)
        assert y.shape == (1, 50)


# ── Nonlinear least squares ───────────────────────────────────────────────────

class TestNLS:
    def test_fit_line(self):
        """y = a*x + b — trivial regression to check interface."""
        t    = np.linspace(0, 10, 100)
        a, b = 2.5, -1.0
        y    = a * t + b + np.random.default_rng(0).normal(0, 0.01, len(t))

        def res(p): return p[0] * t + p[1] - y

        result = nonlinear_least_squares(res, p0=np.array([1.0, 0.0]))
        assert result["success"]
        assert abs(result["params"][0] - a) < 0.1
        assert abs(result["params"][1] - b) < 0.1

    def test_covariance_shape(self):
        def res(p): return p[0] * np.ones(20) - 1.0
        result = nonlinear_least_squares(res, p0=np.array([0.5]))
        assert result["covariance"].shape == (1, 1)


# ── ARX ───────────────────────────────────────────────────────────────────────

class TestARX:
    def test_fit_first_order(self):
        """AR(1) model: y_k = 0.9*y_{k-1} + 0.5*u_k."""
        rng  = np.random.default_rng(1)
        N    = 500
        a, b = 0.9, 0.5
        u    = rng.uniform(-1, 1, N)
        y    = np.zeros(N)
        for k in range(1, N):
            y[k] = a * y[k-1] + b * u[k] + rng.normal(0, 0.01)

        result = fit_arx(y, u, na=1, nb=1, nk=0)
        params = result["params"]
        assert abs(params[0] - (-a)) < 0.05   # ARX sign convention (negated a)
        assert abs(params[1] - b) < 0.05

    def test_returns_covariance(self):
        y = np.random.randn(200)
        u = np.random.randn(200)
        r = fit_arx(y, u, na=2, nb=2)
        assert r["covariance"].shape == (4, 4)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["rmse"]  == pytest.approx(0.0, abs=1e-10)
        assert m["r2"]    == pytest.approx(1.0, abs=1e-10)
        assert m["nrmse"] == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_error(self):
        y_true = np.zeros(4)
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(1.0)
        assert m["mae"]  == pytest.approx(1.0)

    def test_r2_range(self):
        rng = np.random.default_rng(0)
        y_t = rng.standard_normal(100)
        y_p = y_t + rng.normal(0, 0.3, 100)
        m   = compute_metrics(y_t, y_p)
        assert -1 < m["r2"] <= 1.0


# ── Whiteness test ────────────────────────────────────────────────────────────

class TestWhitenessTest:
    def test_white_noise_passes(self):
        r = np.random.default_rng(0).standard_normal(500)
        result = residual_whiteness_test(r)
        assert result["is_white"]   # white noise should pass

    def test_autocorrelated_fails(self):
        r = np.zeros(500)
        for i in range(1, 500):
            r[i] = 0.95 * r[i-1] + np.random.randn() * 0.1
        result = residual_whiteness_test(r)
        assert not result["is_white"]


# ── FIM ───────────────────────────────────────────────────────────────────────

class TestFIM:
    def test_two_param_system(self):
        """Simple y = a*sin(t) + b*cos(t) — both params should be identifiable."""
        t = np.linspace(0, 4*np.pi, 200)
        u = np.ones(len(t))   # not used

        def sensitivity_fn(p, u_):
            return p[0] * np.sin(t) + p[1] * np.cos(t)

        result = compute_fim(sensitivity_fn, t, u, params=np.array([1.0, 1.0]))
        assert result["full_rank"]
        assert result["rank"] == 2
        assert result["condition_number"] < 1e6

    def test_unidentifiable_detects_low_rank(self):
        """y = a*x + b*x — collinear, should detect rank deficiency."""
        x = np.linspace(0, 1, 100)
        u = np.ones(len(x))

        def sensitivity_fn(p, u_):
            return p[0] * x + p[1] * x   # a and b always appear as (a+b)*x

        result = compute_fim(sensitivity_fn, x, u, params=np.array([1.0, 1.0]))
        assert not result["full_rank"]   # rank 1, not 2


# ── Plant API ─────────────────────────────────────────────────────────────────

class TestPlantAPI:
    @pytest.fixture
    def api(self, tmp_path):
        from core.schemas import PlantContract
        from plants.inverted_pendulum import PendulumPlant
        from tools.budget_manager import BudgetManager
        from tools.experiment_db import ExperimentDatabase
        from tools.plant_api import PlantAPI

        plant    = PendulumPlant(seed=42)
        contract = PlantContract(
            name="test_pendulum",
            input_names=["torque"],
            output_names=["angle"],
            input_limits={"torque": (-2.0, 2.0)},
            sample_time=0.02,
        )
        budget = BudgetManager(total=50.0)
        db     = ExperimentDatabase(
            db_path=str(tmp_path / "test.db"),
            data_dir=str(tmp_path / "runs"),
        )
        return PlantAPI(plant, contract, budget, db, experiment_cost=1.0)

    def test_basic_run(self, api):
        u = lambda t: np.array([0.3])
        result = api.apply_input(u, (0.0, 2.0), 0.02, "identification", "prbs", "estimator")
        assert "run_id" in result
        assert result["n_samples"] > 0
        assert result["safety_status"] == "ok"

    def test_budget_debited(self, api):
        from tools.budget_manager import BudgetManager
        u = lambda t: np.array([0.0])
        api.apply_input(u, (0.0, 1.0), 0.02, "id", "prbs", "est")
        assert api._budget.spent == pytest.approx(1.0)

    def test_input_clipped(self, api):
        """Input above contract limit (2.0 N·m) should be clipped."""
        u = lambda t: np.array([10.0])   # way over the 2.0 limit
        result = api.apply_input(u, (0.0, 0.5), 0.02, "id", "step", "est")
        assert result["safety_status"] == "clipped"

    def test_budget_exhausted_raises(self, api):
        from tools.budget_manager import BudgetExhaustedError
        # Drain the budget
        api._budget.debit(50.0)
        u = lambda t: np.array([0.0])
        with pytest.raises(BudgetExhaustedError):
            api.apply_input(u, (0.0, 1.0), 0.02, "id", "prbs", "est")

    def test_run_stored_in_db(self, api):
        u = lambda t: np.array([0.1])
        result = api.apply_input(u, (0.0, 1.0), 0.02, "validation", "adversarial", "validation")
        runs = api._db.query_runs()
        assert any(r["id"] == result["run_id"] for r in runs)
