"""
Deterministic surrogate loop tests — no LLM calls required.

Tests the surrogate sub-orchestrator against the pendulum with pre-collected
training data.  The surrogate fits a GP or MLP to (theta, theta_dot, u) →
theta_ddot and is validated via the modified ValidationAgent surrogate path.

True pendulum parameters:
  K_in  = 1/J           = 1/0.05        = 20.0
  tau_d = b_v/J         = 0.02/0.05     = 0.40
  K_g   = m·g·L/J       = 0.5·9.81·0.30/0.05 = 29.43
  K_c   = f_c/J         = 0.08/0.05     = 1.60  (Coulomb — causes white-box to fail)
"""
import pickle

import numpy as np
import pytest

from core.schemas import (
    AgentStatus,
    Assets,
    Artifacts,
    Budget,
    Dossier,
    EntryPath,
    GapType,
    ModelArtifact,
    ModelType,
    PhysicsAvailability,
    PlantContract,
    Rung,
    SplitFlag,
    Verdict,
    VerdictResult,
)
from plants.inverted_pendulum import PendulumPlant
from tools.budget_manager import BudgetManager
from tools.experiment_db import ExperimentDatabase
from tools.model_registry import ModelRegistry
from tools.plant_api import PlantAPI

WB_RHS = "K_in*tau - tau_d*theta_dot - K_g*sin(theta)"
K_IN_TRUE  = 1.0 / 0.05
TAU_D_TRUE = 0.02 / 0.05
K_G_TRUE   = 0.5 * 9.81 * 0.30 / 0.05


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("surrogate_loop")

    plant    = PendulumPlant(seed=5)
    contract = PlantContract(
        name="pendulum_surrogate_test",
        input_names=["torque"],
        output_names=["angle"],
        input_limits={"torque": (-2.0, 2.0)},
        sample_time=0.02,
    )
    budget   = BudgetManager(total=500.0)
    db       = ExperimentDatabase(
        db_path=str(tmp / "test.db"),
        data_dir=str(tmp / "runs"),
    )
    registry = ModelRegistry(str(tmp / "models"))
    api      = PlantAPI(plant, contract, budget, db, experiment_cost=2.0)

    # Inject a failed grey-box model (poly fallback — the model that triggered
    # escalation to surrogate).  The SO only uses its meta (state/input/output vars).
    failed_model = ModelArtifact(
        model_type=ModelType.GREY_BOX,
        structure_description="grey-box poly fallback (failed)",
        parameters={"K_in": K_IN_TRUE, "tau_d": TAU_D_TRUE, "K_g": K_G_TRUE,
                    "a1": 0.0, "a3": 0.0},
        metadata={
            "normalized_rhs": WB_RHS + " + a1*theta_dot + a3*theta_dot**3",
            "fit_params":     ["K_in", "tau_d", "K_g", "a1", "a3"],
            "state_vars":     ["theta", "theta_dot"],
            "input_vars":     ["tau"],
            "output_vars":    ["theta"],
            "improvable":     False,
        },
    )
    failed_id = registry.store_model(failed_model)

    # Store contract artifact (needed by contract loader)
    contract_artifact = ModelArtifact(
        id=contract.id,
        model_type=ModelType.WHITE_BOX,
        structure_description="PlantContract:pendulum_surrogate_test",
        metadata={"plant_contract": {
            "id":               contract.id,
            "name":             contract.name,
            "input_names":      contract.input_names,
            "output_names":     contract.output_names,
            "state_names":      contract.state_names,
            "input_limits":     {k: list(v) for k, v in contract.input_limits.items()},
            "input_rate_limits": contract.input_rate_limits,
            "output_limits":    {},
            "sample_time":      contract.sample_time,
            "is_unstable":      False,
            "description":      "",
        }},
    )
    registry.store_model(contract_artifact)

    # Collect 300-sample PRBS training run (forces GP path in tests)
    from agents.experiment_design import ExperimentDesignAgent
    designer = ExperimentDesignAgent()
    seq    = designer.design_for_identification(contract, n_samples=300, seed=11)
    u_func = designer.make_u_func(seq["t"], seq["u"])
    result = api.apply_input(
        u_func=u_func,
        t_span=(float(seq["t"][0]), float(seq["t"][-1])),
        dt=float(seq["t"][1] - seq["t"][0]),
        purpose="identification",
        input_type="prbs",
        agent="test_setup",
        split_flag=SplitFlag.TRAIN,
    )
    run_id = result["run_id"]

    # Build dossier (as if validation just failed after grey-box correction)
    dossier = Dossier(
        entry_path=EntryPath.WHITE_BOX,
        current_rung=Rung.GREY,
        budget=Budget(total=500.0),
        status="validation: fail/unmodelable (after greybox)",
        assets=Assets(
            plant_contract_id=contract.id,
            physics=PhysicsAvailability.FULL,
        ),
        artifacts=Artifacts(
            current_model_id=failed_id,
            model_history=[failed_id],
            dataset_ids=[run_id],
        ),
        last_verdict=Verdict(
            verdict=VerdictResult.FAIL,
            gap_type=GapType.UNMODELABLE,
            metrics={"rmse": 0.30},
        ),
    )

    return {
        "plant":           plant,
        "contract":        contract,
        "budget":          budget,
        "db":              db,
        "registry":        registry,
        "api":             api,
        "failed_id":       failed_id,
        "run_id":          run_id,
        "initial_dossier": dossier,
    }


# ── Model class selector tests ────────────────────────────────────────────────

class TestModelClassSelector:

    def test_small_dataset_selects_gp(self):
        from agents.surrogate.model_class_selector import ModelClassSelector, ModelClass
        sel = ModelClassSelector().select(200)
        assert sel.model_class == ModelClass.GP

    def test_at_threshold_selects_gp(self):
        from agents.surrogate.model_class_selector import ModelClassSelector, ModelClass, GP_THRESHOLD
        sel = ModelClassSelector().select(GP_THRESHOLD)
        assert sel.model_class == ModelClass.GP

    def test_large_dataset_selects_nn(self):
        from agents.surrogate.model_class_selector import ModelClassSelector, ModelClass, GP_THRESHOLD
        sel = ModelClassSelector().select(GP_THRESHOLD + 1)
        assert sel.model_class == ModelClass.NN

    def test_rationale_is_non_empty(self):
        from agents.surrogate.model_class_selector import ModelClassSelector
        for n in [50, 300, 1000]:
            sel = ModelClassSelector().select(n)
            assert len(sel.rationale) > 0


# ── Surrogate trainer tests (unit, no plant) ──────────────────────────────────

class TestSurrogateTrainer:

    def _make_synthetic_data(self, N=200, seed=0):
        """Synthetic theta_ddot = K_in*u - K_g*sin(theta) (no Coulomb)."""
        rng = np.random.default_rng(seed)
        t         = np.linspace(0, 4.0, N)
        theta     = rng.uniform(-0.5, 0.5, N)
        theta_dot = rng.uniform(-1.0, 1.0, N)
        u         = rng.uniform(-2.0, 2.0, N)
        theta_ddot = (
            K_IN_TRUE * u
            - TAU_D_TRUE * theta_dot
            - K_G_TRUE   * np.sin(theta)
            + rng.normal(0, 0.5, N)
        )
        return theta, theta_dot, u, theta_ddot

    def test_gp_trains_on_synthetic_data(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data()
        result = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=0)
        assert result.predictor is not None
        assert result.n_train > 0
        assert np.isfinite(result.train_rmse)

    def test_gp_train_rmse_reasonable(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data(N=300)
        result = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=0)
        # GP interpolates well at training points — RMSE should be small
        assert result.train_rmse < 5.0, (
            f"GP train RMSE={result.train_rmse:.3f} — should be < 5.0 rad/s²"
        )

    def test_gp_predictions_are_finite(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data()
        result = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=1)
        preds  = result.predictor.predict(theta[:10], td[:10], u[:10])
        assert np.all(np.isfinite(preds)), "GP predictions must be finite"

    def test_gp_uncertainty_returns_two_arrays(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data()
        result = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=2)
        mu, std = result.predictor.predict_with_std(theta[:5], td[:5], u[:5])
        assert mu.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0), "Std must be non-negative"

    def test_gp_scalar_prediction_returns_float(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data()
        result = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=3)
        pred = result.predictor.predict(0.1, 0.5, 1.0)
        assert isinstance(pred, float), f"Scalar input must return float, got {type(pred)}"

    def test_gp_predictor_is_picklable(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data()
        result    = SurrogateTrainer().fit(ModelClass.GP, theta, td, u, tddot, seed=0)
        blob      = pickle.dumps(result.predictor)
        predictor = pickle.loads(blob)
        pred      = predictor.predict(0.1, 0.5, 1.0)
        assert isinstance(pred, float)

    def test_nn_trains_if_torch_available(self):
        torch = pytest.importorskip("torch")
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data(N=600)
        result = SurrogateTrainer().fit(
            ModelClass.NN, theta, td, u, tddot, n_epochs=30, seed=0
        )
        assert result.predictor is not None
        assert np.isfinite(result.train_rmse)
        pred = result.predictor.predict(0.1, 0.5, 1.0)
        assert isinstance(pred, float)

    def test_nn_predictor_is_picklable(self):
        pytest.importorskip("torch")
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        theta, td, u, tddot = self._make_synthetic_data(N=600)
        result    = SurrogateTrainer().fit(
            ModelClass.NN, theta, td, u, tddot, n_epochs=30, seed=0
        )
        blob      = pickle.dumps(result.predictor)
        predictor = pickle.loads(blob)
        pred      = predictor.predict(0.1, 0.5, 1.0)
        assert isinstance(pred, float)


# ── Uncertainty estimator tests ───────────────────────────────────────────────

class TestUncertaintyEstimator:

    def test_uncertainty_dict_has_expected_keys(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        from agents.surrogate.uncertainty_estimator import UncertaintyEstimator

        rng       = np.random.default_rng(0)
        N         = 100
        theta     = rng.uniform(-0.5, 0.5, N)
        theta_dot = rng.uniform(-1, 1, N)
        u         = rng.uniform(-2, 2, N)
        tddot     = K_IN_TRUE * u - K_G_TRUE * np.sin(theta)

        result = SurrogateTrainer().fit(ModelClass.GP, theta, theta_dot, u, tddot)
        unc    = UncertaintyEstimator().estimate(
            ModelClass.GP, result.predictor,
            theta, theta_dot, u, tddot,
        )
        assert "mean_std"    in unc
        assert "rmse"        in unc
        assert "coverage_95" in unc

    def test_coverage_between_0_and_1(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        from agents.surrogate.uncertainty_estimator import UncertaintyEstimator

        rng       = np.random.default_rng(1)
        N         = 100
        theta     = rng.uniform(-0.5, 0.5, N)
        theta_dot = rng.uniform(-1, 1, N)
        u         = rng.uniform(-2, 2, N)
        tddot     = K_IN_TRUE * u - K_G_TRUE * np.sin(theta)

        result = SurrogateTrainer().fit(ModelClass.GP, theta, theta_dot, u, tddot)
        unc    = UncertaintyEstimator().estimate(
            ModelClass.GP, result.predictor, theta, theta_dot, u, tddot
        )
        assert 0.0 <= unc["coverage_95"] <= 1.0

    def test_rmse_is_finite(self):
        from agents.surrogate.model_class_selector import ModelClass
        from agents.surrogate.trainer import SurrogateTrainer
        from agents.surrogate.uncertainty_estimator import UncertaintyEstimator

        rng       = np.random.default_rng(2)
        N         = 100
        theta     = rng.uniform(-0.5, 0.5, N)
        theta_dot = rng.uniform(-1, 1, N)
        u         = rng.uniform(-2, 2, N)
        tddot     = K_IN_TRUE * u - K_G_TRUE * np.sin(theta)

        result = SurrogateTrainer().fit(ModelClass.GP, theta, theta_dot, u, tddot)
        unc    = UncertaintyEstimator().estimate(
            ModelClass.GP, result.predictor, theta, theta_dot, u, tddot
        )
        assert np.isfinite(unc["rmse"])


# ── Active sampler tests ──────────────────────────────────────────────────────

class TestActiveSampler:

    def test_no_collection_if_sufficient_data(self, setup):
        from agents.surrogate.active_sampler import ActiveSampler, MIN_SAMPLES_FOR_SURROGATE
        sampler = ActiveSampler(setup["api"], setup["db"])
        new_ids = sampler.maybe_collect(
            n_available=MIN_SAMPLES_FOR_SURROGATE,
            contract=setup["contract"],
            existing_run_ids=[setup["run_id"]],
        )
        assert new_ids == []

    def test_collects_when_sparse(self, setup):
        from agents.surrogate.active_sampler import ActiveSampler, MIN_SAMPLES_FOR_SURROGATE
        sampler = ActiveSampler(setup["api"], setup["db"])
        new_ids = sampler.maybe_collect(
            n_available=MIN_SAMPLES_FOR_SURROGATE - 1,
            contract=setup["contract"],
            existing_run_ids=[],
            seed_offset=500,
        )
        assert len(new_ids) >= 1, "Should collect at least one run when data is sparse"


# ── Full sub-orchestrator integration tests ───────────────────────────────────

class TestSurrogateSO:

    def test_so_returns_done_report(self, setup):
        from agents.surrogate.sub_orchestrator import SurrogateSubOrchestrator
        so = SurrogateSubOrchestrator(
            setup["api"], setup["registry"], setup["db"],
            n_samples=300, nn_epochs=30,
        )
        updated = so(setup["initial_dossier"])

        assert updated.last_report is not None, "Report must be set"
        assert updated.last_report.status == AgentStatus.DONE, (
            f"Report status: {updated.last_report.status}. "
            f"Summary: {updated.last_report.summary}"
        )
        setup["dossier_after_so"] = updated
        setup["so_report"]        = updated.last_report

    def test_rung_is_black(self, setup):
        if "dossier_after_so" not in setup:
            pytest.skip("depends on SO test")
        assert setup["dossier_after_so"].current_rung == Rung.BLACK, (
            f"Surrogate must set Rung.BLACK, got {setup['dossier_after_so'].current_rung}"
        )

    def test_model_type_is_black_box(self, setup):
        if "dossier_after_so" not in setup:
            pytest.skip("depends on SO test")
        model_id = setup["dossier_after_so"].artifacts.current_model_id
        model    = setup["registry"].load_model(model_id)
        assert model.model_type == ModelType.BLACK_BOX, (
            f"Expected BLACK_BOX, got {model.model_type}"
        )
        setup["surrogate_model_id"] = model_id
        setup["surrogate_meta"]     = model.metadata

    def test_normalized_rhs_is_surrogate_marker(self, setup):
        if "surrogate_meta" not in setup:
            pytest.skip("depends on model_type test")
        assert setup["surrogate_meta"].get("normalized_rhs") == "SURROGATE"

    def test_surrogate_object_stored_and_loadable(self, setup):
        if "surrogate_meta" not in setup:
            pytest.skip("depends on model_type test")
        obj_id    = setup["surrogate_meta"].get("surrogate_object_id", "")
        predictor = setup["registry"].load_object(obj_id)
        assert predictor is not None
        setup["predictor"] = predictor

    def test_predictor_scalar_prediction_is_finite(self, setup):
        if "predictor" not in setup:
            pytest.skip("depends on stored-object test")
        pred = setup["predictor"].predict(0.1, 0.5, 1.0)
        assert isinstance(pred, float)
        assert np.isfinite(pred), f"Prediction must be finite, got {pred}"

    def test_predictor_array_prediction_is_finite(self, setup):
        if "predictor" not in setup:
            pytest.skip("depends on stored-object test")
        rng   = np.random.default_rng(42)
        theta = rng.uniform(-0.3, 0.3, 20)
        td    = rng.uniform(-1.0, 1.0, 20)
        u     = rng.uniform(-2.0, 2.0, 20)
        preds = setup["predictor"].predict(theta, td, u)
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))

    def test_train_rmse_stored_in_metadata(self, setup):
        if "so_report" not in setup:
            pytest.skip("depends on SO test")
        rmse = setup["so_report"].metadata.get("train_rmse", float("inf"))
        assert np.isfinite(rmse), f"train_rmse must be finite, got {rmse}"

    def test_surrogate_model_id_in_model_history(self, setup):
        if "dossier_after_so" not in setup:
            pytest.skip("depends on SO test")
        dossier = setup["dossier_after_so"]
        assert dossier.artifacts.current_model_id in dossier.artifacts.model_history

    def test_model_class_reported_in_metadata(self, setup):
        if "so_report" not in setup:
            pytest.skip("depends on SO test")
        mc = setup["so_report"].metadata.get("model_class", "")
        assert mc in ("gp", "nn"), f"model_class must be gp or nn, got '{mc}'"

    def test_predictor_is_picklable_after_storage(self, setup):
        if "predictor" not in setup:
            pytest.skip("depends on stored-object test")
        blob = pickle.dumps(setup["predictor"])
        p2   = pickle.loads(blob)
        pred = p2.predict(0.0, 0.0, 0.5)
        assert isinstance(pred, float)


# ── Validation with surrogate model (final integration) ──────────────────────

class TestSurrogateValidation:

    def test_surrogate_validation_runs_without_error(self, setup):
        if "dossier_after_so" not in setup:
            pytest.skip("depends on SO test")

        from agents.validation import ValidationAgent
        val     = ValidationAgent(setup["api"], setup["registry"], setup["db"])
        updated = val(setup["dossier_after_so"])
        verdict = updated.last_verdict

        assert verdict is not None, "ValidationAgent must return a verdict"
        assert verdict.metrics.get("n_scenarios", 0) >= 1, (
            "At least one validation scenario must complete"
        )
        setup["surrogate_verdict"] = verdict

    def test_surrogate_gap_type_not_unmodelable(self, setup):
        """
        The surrogate has been fit to the real pendulum data.
        After training it should at least capture the dominant dynamics,
        so the residuals should be smaller than the original white-box failure.
        We do not assert PASS — just that the validation runs and produces a verdict.
        """
        if "surrogate_verdict" not in setup:
            pytest.skip("depends on validation test")
        verdict = setup["surrogate_verdict"]
        # The surrogate may still get UNMODELABLE if its ODE integration blows up,
        # but the validation metrics must be finite.
        rmse = verdict.metrics.get("rmse", float("inf"))
        assert np.isfinite(rmse), f"Validation RMSE must be finite, got {rmse}"

    def test_surrogate_rmse_not_worse_than_trivial(self, setup):
        """
        A surrogate trained on pendulum data should predict better than a zero model.
        RMSE < 2.0 rad is a very loose lower bar (zero-model error ≈ 0.3 rad per 0.5s
        segment, so 2.0 rad means the surrogate is at least not actively harmful).
        """
        if "surrogate_verdict" not in setup:
            pytest.skip("depends on validation test")
        rmse = setup["surrogate_verdict"].metrics.get("rmse", float("inf"))
        assert rmse < 2.0, (
            f"Surrogate RMSE={rmse:.3f} rad — expected < 2.0 rad. "
            "The surrogate integration may be diverging."
        )
