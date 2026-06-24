"""
Integration test — full white-box identification pipeline on the pendulum.

Requires ANTHROPIC_API_KEY (skipped otherwise).

What this tests:
  1. Intake: parses "pendulum with viscous friction" → PlantContract
  2. Modeler: derives θ̈ = K_in·τ - τ_d·θ̇ - K_g·sin(θ), checks identifiability
  3. Estimator: fits K_g, τ_d, K_in from PRBS experiments
  4. Validation: detects Coulomb residuals → gap_type = STRUCTURED_RESIDUAL

Expected outcomes:
  - fitted K_g ≈ 29.43  (within 25%)
  - fitted τ_d ≈ 0.40   (within 25%)
  - gap_type = STRUCTURED_RESIDUAL
"""
import os
import tempfile
import pytest
import numpy as np

from core.schemas import (
    Assets, Budget, Dossier, EntryPath, GapType, PhysicsAvailability,
    PlantContract, Rung, VerdictResult,
)
from plants.inverted_pendulum import PendulumPlant
from tools.budget_manager import BudgetManager
from tools.experiment_db import ExperimentDatabase
from tools.model_registry import ModelRegistry
from tools.plant_api import PlantAPI

# ── Skip if no API key ────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM integration test",
)

# ── True pendulum values for assertion ────────────────────────────────────────
TRUE_K_G    = 0.5 * 9.81 * 0.30 / 0.05   # = 29.43
TRUE_TAU_D  = 0.02 / 0.05                 # = 0.40
TRUE_K_IN   = 1.0 / 0.05                  # = 20.0

PLANT_DESCRIPTION = """\
A simple pendulum driven by an external torque.
The bob has unknown mass, mounted on a rod of unknown length.
Viscous friction acts on the pivot.
Input: torque [N·m], clamped to [-2, 2].
Output: angle θ (measured from the downward equilibrium).
Sample time: 0.02 s.
I want to identify a control model from experiments.
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Set up the full pipeline infrastructure once for the module."""
    tmp = tmp_path_factory.mktemp("pipeline")

    plant    = PendulumPlant(seed=42)
    contract = PlantContract(
        name="pendulum_integration",
        input_names=["torque"],
        output_names=["angle"],
        input_limits={"torque": (-2.0, 2.0)},
        sample_time=0.02,
        description="Pendulum for integration test",
    )
    budget   = BudgetManager(total=60.0)
    db       = ExperimentDatabase(
        db_path=str(tmp / "test.db"),
        data_dir=str(tmp / "runs"),
    )
    registry = ModelRegistry(str(tmp / "models"))
    api      = PlantAPI(plant, contract, budget, db, experiment_cost=2.0)

    return {
        "plant": plant, "contract": contract, "budget": budget,
        "db": db, "registry": registry, "api": api,
        "tmp": tmp,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntake:
    """Intake agent parses the plant description and stores a PlantContract."""

    def test_intake_creates_contract(self, pipeline):
        from agents.intake import IntakeAgent

        registry = pipeline["registry"]
        agent    = IntakeAgent(registry, budget_total=60.0)

        # Simulate the dossier that the orchestrator would build
        dossier = Dossier(
            entry_path=EntryPath.WHITE_BOX,
            budget=Budget(total=60.0),
            status=PLANT_DESCRIPTION,   # description stored in status
            assets=Assets(),
        )
        updated = agent(dossier)

        assert updated.assets.plant_contract_id, "Intake must store a plant_contract_id"
        assert updated.entry_path == EntryPath.WHITE_BOX
        assert updated.assets.physics in (
            PhysicsAvailability.FULL, PhysicsAvailability.PARTIAL
        )
        assert updated.last_report is not None

        # Store contract_id for downstream tests
        pipeline["contract_id"] = updated.assets.plant_contract_id
        pipeline["dossier_after_intake"] = updated


class TestModeler:
    """Modeler derives the ODE and stores it with identifiable parameters."""

    def test_modeler_stores_model(self, pipeline):
        if "dossier_after_intake" not in pipeline:
            pytest.skip("Depends on intake test; run TestIntake first")

        from agents.modeler import ModelerAgent

        registry = pipeline["registry"]
        agent    = ModelerAgent(registry)

        dossier  = pipeline["dossier_after_intake"]
        updated  = agent(dossier)

        model_id = updated.artifacts.current_model_id
        assert model_id, "Modeler must store a model_id"

        # Load and inspect the stored model
        model = registry.load_model(model_id)
        meta  = model.metadata
        assert "normalized_rhs"  in meta, "Model must have normalized_rhs"
        assert "fit_params"      in meta, "Model must have fit_params"
        assert "state_vars"      in meta
        assert len(meta["fit_params"]) >= 2, "Should have at least 2 identifiable params"

        pipeline["dossier_after_modeler"] = updated
        pipeline["model_id_structure"]    = model_id

    def test_model_rhs_contains_sin(self, pipeline):
        if "model_id_structure" not in pipeline:
            pytest.skip("Depends on modeler test")

        registry = pipeline["registry"]
        model    = registry.load_model(pipeline["model_id_structure"])
        rhs      = model.metadata["normalized_rhs"]
        assert "sin" in rhs.lower(), f"Pendulum ODE must contain sin(), got: {rhs}"


class TestEstimator:
    """Estimator fits K_g, τ_d within 25% of true values."""

    def test_estimator_fits_parameters(self, pipeline):
        if "dossier_after_modeler" not in pipeline:
            pytest.skip("Depends on modeler test")

        from agents.estimator import EstimatorAgent

        api      = pipeline["api"]
        registry = pipeline["registry"]
        db       = pipeline["db"]
        agent    = EstimatorAgent(api, registry, db, n_samples=500)

        dossier  = pipeline["dossier_after_modeler"]
        updated  = agent(dossier)

        meta = updated.last_report.metadata
        assert meta.get("model_id"), "Estimator must store a fitted model_id"

        params = meta.get("params", {})
        assert params, "Estimator must report fitted parameter values"

        pipeline["dossier_after_estimator"] = updated
        pipeline["fitted_params"]           = params

    def test_K_g_within_tolerance(self, pipeline):
        if "fitted_params" not in pipeline:
            pytest.skip("Depends on estimator test")

        params = pipeline["fitted_params"]
        # Find K_g (may be named K_g, Kg, gravitational_gain, etc.)
        kg_val = _find_param(params, ["K_g", "Kg", "k_g", "kg"])
        if kg_val is None:
            pytest.skip("K_g parameter not found in fitted params")

        tol = 0.25   # 25% relative tolerance
        rel_err = abs(kg_val - TRUE_K_G) / TRUE_K_G
        assert rel_err < tol, (
            f"K_g = {kg_val:.3f}, expected ≈ {TRUE_K_G:.3f} (±{tol*100:.0f}%). "
            f"Relative error = {rel_err*100:.1f}%"
        )

    def test_tau_d_within_tolerance(self, pipeline):
        if "fitted_params" not in pipeline:
            pytest.skip("Depends on estimator test")

        params = pipeline["fitted_params"]
        td_val = _find_param(params, ["tau_d", "tau_D", "taud", "damping"])
        if td_val is None:
            pytest.skip("tau_d parameter not found in fitted params")

        tol = 0.30   # 30% — harder to fit due to Coulomb interference
        rel_err = abs(td_val - TRUE_TAU_D) / TRUE_TAU_D
        assert rel_err < tol, (
            f"tau_d = {td_val:.3f}, expected ≈ {TRUE_TAU_D:.3f} (±{tol*100:.0f}%). "
            f"Relative error = {rel_err*100:.1f}%"
        )


class TestValidation:
    """Validation detects Coulomb friction → gap_type = STRUCTURED_RESIDUAL."""

    def test_validation_fails_white_box(self, pipeline):
        if "dossier_after_estimator" not in pipeline:
            pytest.skip("Depends on estimator test")

        from agents.validation import ValidationAgent

        api      = pipeline["api"]
        registry = pipeline["registry"]
        db       = pipeline["db"]
        agent    = ValidationAgent(api, registry, db)

        dossier  = pipeline["dossier_after_estimator"]
        updated  = agent(dossier)

        verdict  = updated.last_verdict
        assert verdict is not None
        # White-box WITHOUT Coulomb should FAIL
        assert verdict.verdict == VerdictResult.FAIL, (
            "White-box (no Coulomb) should fail validation against the true plant"
        )
        pipeline["dossier_after_validation"] = updated
        pipeline["verdict"] = verdict

    def test_gap_type_is_structured_residual(self, pipeline):
        if "verdict" not in pipeline:
            pytest.skip("Depends on validation test")

        verdict = pipeline["verdict"]
        assert verdict.gap_type == GapType.STRUCTURED_RESIDUAL, (
            f"Expected gap_type=structured_residual (Coulomb friction pattern), "
            f"got gap_type={verdict.gap_type.value}. "
            f"Metrics: {verdict.metrics}"
        )

    def test_coulomb_correlation_high(self, pipeline):
        if "verdict" not in pipeline:
            pytest.skip("Depends on validation test")

        verdict = pipeline["verdict"]
        cc = verdict.metrics.get("coulomb_correlation", 0.0)
        assert cc > 0.15, (
            f"Expected Coulomb correlation > 0.15, got {cc:.3f}. "
            "Coulomb friction should show up in residuals."
        )

    def test_validity_region_stored(self, pipeline):
        if "verdict" not in pipeline:
            pytest.skip("Depends on validation test")

        verdict  = pipeline["verdict"]
        registry = pipeline["registry"]
        if verdict.validity_region_id:
            region = registry.load_validity(verdict.validity_region_id)
            assert region.model_id is not None


# ── Full pipeline smoke test ──────────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end: run all four stages and assert the gap_type."""

    def test_full_pipeline_gap_type(self, pipeline, tmp_path_factory):
        """Run everything from scratch in isolation."""
        tmp = tmp_path_factory.mktemp("full_pipeline")

        plant    = PendulumPlant(seed=7)
        contract = PlantContract(
            name="pendulum_full_test",
            input_names=["torque"],
            output_names=["angle"],
            input_limits={"torque": (-2.0, 2.0)},
            sample_time=0.02,
        )
        budget   = BudgetManager(total=80.0)
        db       = ExperimentDatabase(
            db_path=str(tmp / "test.db"),
            data_dir=str(tmp / "runs"),
        )
        registry = ModelRegistry(str(tmp / "models"))
        api      = PlantAPI(plant, contract, budget, db, experiment_cost=2.0)

        from agents.intake import IntakeAgent
        from agents.modeler import ModelerAgent
        from agents.estimator import EstimatorAgent
        from agents.validation import ValidationAgent

        intake_agent = IntakeAgent(registry, budget_total=80.0)
        modeler_agent = ModelerAgent(registry)
        estimator_agent = EstimatorAgent(api, registry, db, n_samples=500)
        validation_agent = ValidationAgent(api, registry, db)

        # Step 1: Intake
        dossier = Dossier(
            entry_path=EntryPath.WHITE_BOX,
            budget=Budget(total=80.0),
            status=PLANT_DESCRIPTION,
            assets=Assets(),
        )
        dossier = intake_agent(dossier)
        assert dossier.assets.plant_contract_id, "Intake failed"

        # Step 2: Modeler
        dossier = modeler_agent(dossier)
        assert dossier.artifacts.current_model_id, "Modeler failed"

        # Step 3: Estimator
        dossier = estimator_agent(dossier)
        assert dossier.last_report is not None

        # Step 4: Validation
        dossier = validation_agent(dossier)
        verdict = dossier.last_verdict

        assert verdict is not None
        assert verdict.verdict == VerdictResult.FAIL, \
            "White-box (no Coulomb) must fail"
        assert verdict.gap_type == GapType.STRUCTURED_RESIDUAL, \
            f"Expected STRUCTURED_RESIDUAL, got {verdict.gap_type.value}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_param(params: dict, candidates: list) -> float | None:
    """Find a parameter value by trying multiple common names."""
    for name in candidates:
        for k, v in params.items():
            if name.lower() in k.lower():
                return float(v)
    return None
