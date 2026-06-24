"""Tests for core/schemas.py — the contract layer."""
import pytest
from datetime import datetime

from core.schemas import (
    AgentStatus,
    ArtifactRef,
    Assets,
    Artifacts,
    Budget,
    Critique,
    DataCoverageReport,
    Dossier,
    EntryPath,
    ExcitationQuality,
    ExperimentRun,
    GapType,
    IdentifiabilityReport,
    IdentifiabilityResult,
    ModelArtifact,
    ModelType,
    PhysicsAvailability,
    PlantContract,
    Report,
    Rung,
    SafetyStatus,
    SplitFlag,
    ValidityRegion,
    Verdict,
    VerdictResult,
    WorstCase,
)


# ── Budget ────────────────────────────────────────────────────────────────────

class TestBudget:
    def test_remaining_auto_computed(self):
        b = Budget(total=100.0, spent=30.0)
        assert b.remaining == pytest.approx(70.0)

    def test_debit_reduces_remaining(self):
        b = Budget(total=100.0)
        b2 = b.debit(25.0)
        assert b2.spent     == pytest.approx(25.0)
        assert b2.remaining == pytest.approx(75.0)

    def test_debit_is_immutable(self):
        b = Budget(total=100.0)
        _ = b.debit(10.0)
        assert b.spent == 0.0   # original unchanged

    def test_exhausted_flag(self):
        b = Budget(total=10.0, spent=10.0)
        assert b.exhausted

    def test_allocate_slice(self):
        b = Budget(total=100.0)
        b2 = b.allocate_slice("identification", 60.0)
        assert b2.slice_allocations["identification"] == 60.0


# ── Verdict ───────────────────────────────────────────────────────────────────

class TestVerdict:
    def test_pass_verdict(self):
        v = Verdict(
            verdict=VerdictResult.PASS,
            gap_type=GapType.NONE,
            metrics={"rmse": 0.02},
        )
        assert v.verdict == VerdictResult.PASS
        assert v.gap_type == GapType.NONE

    def test_fail_with_structured_residual(self):
        v = Verdict(
            verdict=VerdictResult.FAIL,
            gap_type=GapType.STRUCTURED_RESIDUAL,
            metrics={"rmse": 0.18, "residual_whiteness": 0.41},
            critique_id="crit_01",
        )
        assert v.gap_type == GapType.STRUCTURED_RESIDUAL
        assert v.critique_id == "crit_01"

    def test_worst_case_embedded(self):
        wc = WorstCase(region={"theta": (-1.5, -0.8)}, error=0.35, scenario="low-velocity reversal")
        v  = Verdict(verdict=VerdictResult.FAIL, gap_type=GapType.STRUCTURED_RESIDUAL, worst_case=wc)
        assert v.worst_case.error == pytest.approx(0.35)

    def test_serialisation_roundtrip(self):
        v = Verdict(
            verdict=VerdictResult.FAIL,
            gap_type=GapType.FIXABLE,
            metrics={"rmse": 0.1},
        )
        v2 = Verdict.model_validate_json(v.model_dump_json())
        assert v2 == v


# ── Report ────────────────────────────────────────────────────────────────────

class TestReport:
    def test_basic_report(self):
        r = Report(
            agent="estimator",
            status=AgentStatus.DONE,
            summary="Converged after 3 iterations",
            produced=[ArtifactRef(id="m1", type="model", store="registry")],
        )
        assert r.agent == "estimator"
        assert len(r.produced) == 1

    def test_task_id_auto_generated(self):
        r1 = Report(agent="a", status=AgentStatus.DONE, summary="ok")
        r2 = Report(agent="a", status=AgentStatus.DONE, summary="ok")
        assert r1.task_id != r2.task_id


# ── Dossier ───────────────────────────────────────────────────────────────────

class TestDossier:
    def _make(self) -> Dossier:
        return Dossier(
            entry_path=EntryPath.WHITE_BOX,
            budget=Budget(total=200.0),
        )

    def test_initial_state(self):
        d = self._make()
        assert d.current_rung == Rung.WHITE
        assert d.status == "initialized"
        assert d.last_verdict is None

    def test_update_is_immutable(self):
        d  = self._make()
        d2 = d.update(status="running", current_rung=Rung.GREY)
        assert d.status == "initialized"        # original unchanged
        assert d2.status == "running"
        assert d2.current_rung == Rung.GREY

    def test_updated_at_changes(self):
        import time
        d  = self._make()
        time.sleep(0.01)
        d2 = d.update(status="x")
        assert d2.updated_at > d.updated_at

    def test_dossier_id_unique(self):
        assert self._make().id != self._make().id

    def test_serialisation_roundtrip(self):
        d  = self._make()
        d2 = Dossier.model_validate_json(d.model_dump_json())
        assert d2.id == d.id
        assert d2.budget.total == d.budget.total


# ── PlantContract ─────────────────────────────────────────────────────────────

class TestPlantContract:
    def test_construction(self):
        pc = PlantContract(
            name="pendulum",
            input_names=["torque"],
            output_names=["angle"],
            state_names=["theta", "theta_dot"],
            input_limits={"torque": (-2.0, 2.0)},
            sample_time=0.02,
        )
        assert pc.input_limits["torque"] == (-2.0, 2.0)
        assert pc.sample_time == pytest.approx(0.02)


# ── ExperimentRun ─────────────────────────────────────────────────────────────

class TestExperimentRun:
    def test_defaults(self):
        r = ExperimentRun(
            input_type="prbs",
            purpose="identification",
            originating_agent="estimator",
        )
        assert r.safety_status == SafetyStatus.OK
        assert r.split_flag    == SplitFlag.TRAIN
        assert r.provenance    == "system"
        assert r.cost          == pytest.approx(1.0)

    def test_ids_are_unique(self):
        r1 = ExperimentRun(input_type="prbs", purpose="id", originating_agent="a")
        r2 = ExperimentRun(input_type="prbs", purpose="id", originating_agent="a")
        assert r1.id != r2.id


# ── ModelArtifact ─────────────────────────────────────────────────────────────

class TestModelArtifact:
    def test_white_box(self):
        m = ModelArtifact(
            model_type=ModelType.WHITE_BOX,
            structure_description="J·θ̈ = τ - b·θ̇ + m·g·L·sin(θ)",
            parameters={"K_g": 29.43, "tau_d": 0.4},
        )
        assert m.version == 1
        assert "K_g" in m.parameters

    def test_serialisation(self):
        m  = ModelArtifact(model_type=ModelType.GREY_BOX, structure_description="hybrid")
        m2 = ModelArtifact.model_validate_json(m.model_dump_json())
        assert m2.id == m.id


# ── ValidityRegion ────────────────────────────────────────────────────────────

class TestValidityRegion:
    def test_construction(self):
        vr = ValidityRegion(
            model_id="m01",
            bounds={"theta": (-1.5, 1.5), "theta_dot": (-3.0, 3.0)},
            tolerance=0.05,
            achieved_rmse=0.032,
            coverage_fraction=0.94,
        )
        assert vr.bounds["theta"] == (-1.5, 1.5)
        assert vr.coverage_fraction == pytest.approx(0.94)
