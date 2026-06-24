"""
Tests for routing logic.

Architecture note: _route_after_validation now always returns ROUTER (the LLM
routing agent).  The deterministic routing decisions that were previously in
_route_after_validation have moved to _rule_based_fallback in core/router_agent.py,
which is used as a safe fallback if the LLM call fails.

These tests now cover:
  - _route_after_validation: always returns ROUTER
  - _route_after_router: reads last_report.recommended_next
  - _rule_based_fallback: the deterministic fallback logic
  - The other fixed edges (unchanged)
"""
import pytest

from core.schemas import (
    AgentStatus,
    Assets,
    Budget,
    Dossier,
    EntryPath,
    GapType,
    PhysicsAvailability,
    Report,
    Rung,
    Verdict,
    VerdictResult,
)
from core.orchestrator import (
    ESTIMATOR,
    GREYBOX_SO,
    MODELER,
    ROUTER,
    SHIP,
    SURROGATE_SO,
    VALIDATION,
    _route_after_intake,
    _route_after_modeler,
    _route_after_estimator,
    _route_after_validation,
    _route_after_router,
    _route_after_suborch,
)
from core.router_agent import _rule_based_fallback


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _dossier(
    entry_path=EntryPath.WHITE_BOX,
    rung=Rung.WHITE,
    verdict: Verdict | None = None,
    budget_remaining=100.0,
    physics=PhysicsAvailability.FULL,
    recommended_next: str | None = None,
) -> Dossier:
    last_report = None
    if recommended_next is not None:
        last_report = Report(
            agent="RouterAgent",
            status=AgentStatus.DONE,
            summary="test",
            recommended_next=recommended_next,
        )
    return Dossier(
        entry_path=entry_path,
        current_rung=rung,
        budget=Budget(total=200.0, spent=200.0 - budget_remaining),
        last_verdict=verdict,
        last_report=last_report,
        assets=Assets(physics=physics),
    )


def _verdict(result=VerdictResult.PASS, gap=GapType.NONE) -> Verdict:
    return Verdict(verdict=result, gap_type=gap)


# ── Intake routing ────────────────────────────────────────────────────────────

class TestIntakeRouting:
    def test_white_box_entry_goes_to_modeler(self):
        d = _dossier(entry_path=EntryPath.WHITE_BOX)
        assert _route_after_intake(d) == MODELER

    def test_simulator_entry_goes_to_validation(self):
        d = _dossier(entry_path=EntryPath.SIMULATOR)
        assert _route_after_intake(d) == VALIDATION

    def test_surrogate_entry_goes_to_validation(self):
        d = _dossier(entry_path=EntryPath.SURROGATE)
        assert _route_after_intake(d) == VALIDATION


# ── Fixed downstream edges ────────────────────────────────────────────────────

class TestFixedEdges:
    def test_modeler_always_goes_to_estimator(self):
        assert _route_after_modeler(_dossier()) == ESTIMATOR

    def test_estimator_always_goes_to_validation(self):
        assert _route_after_estimator(_dossier()) == VALIDATION

    def test_suborch_always_goes_to_validation(self):
        assert _route_after_suborch(_dossier()) == VALIDATION

    def test_validation_always_goes_to_router(self):
        """Validation now always routes to the LLM RouterAgent node."""
        for gap in GapType:
            for rung in Rung:
                d = _dossier(rung=rung, verdict=_verdict(VerdictResult.FAIL, gap))
                assert _route_after_validation(d) == ROUTER

    def test_validation_pass_also_goes_to_router(self):
        d = _dossier(verdict=_verdict(VerdictResult.PASS, GapType.NONE))
        assert _route_after_validation(d) == ROUTER


# ── Router node reads recommended_next ───────────────────────────────────────

class TestRouterEdge:
    def test_router_reads_recommended_next_greybox(self):
        d = _dossier(recommended_next="greybox_so")
        assert _route_after_router(d) == GREYBOX_SO

    def test_router_reads_recommended_next_surrogate(self):
        d = _dossier(recommended_next="surrogate_so")
        assert _route_after_router(d) == SURROGATE_SO

    def test_router_reads_recommended_next_ship(self):
        d = _dossier(recommended_next="ship")
        assert _route_after_router(d) == SHIP

    def test_router_falls_back_to_ship_on_invalid(self):
        d = _dossier(recommended_next="invalid_node")
        assert _route_after_router(d) == SHIP

    def test_router_falls_back_to_ship_when_no_report(self):
        d = _dossier(recommended_next=None)
        assert _route_after_router(d) == SHIP


# ── Rule-based fallback (used when LLM call fails) ───────────────────────────

class TestRuleBasedFallback:
    """Tests _rule_based_fallback — the same logic as the old _route_after_validation."""

    def test_pass_ships(self):
        d = _dossier(verdict=_verdict(VerdictResult.PASS, GapType.NONE))
        node, _ = _rule_based_fallback(d)
        assert node == SHIP

    def test_pass_ships_even_at_grey_rung(self):
        d = _dossier(rung=Rung.GREY, verdict=_verdict(VerdictResult.PASS))
        node, _ = _rule_based_fallback(d)
        assert node == SHIP

    def test_budget_exhausted_ships_regardless_of_gap(self):
        for gap in GapType:
            d = _dossier(verdict=_verdict(VerdictResult.FAIL, gap), budget_remaining=0.0)
            node, _ = _rule_based_fallback(d)
            assert node == SHIP

    def test_structured_residual_at_white_goes_to_greybox(self):
        d = _dossier(
            rung=Rung.WHITE,
            verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL),
        )
        node, _ = _rule_based_fallback(d)
        assert node == GREYBOX_SO

    def test_structured_residual_at_grey_first_attempt_retries_greybox(self):
        d = _dossier(
            rung=Rung.GREY,
            verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL),
        )
        node, _ = _rule_based_fallback(d)
        assert node == GREYBOX_SO

    def test_structured_residual_at_grey_already_retried_goes_to_surrogate(self):
        d = _dossier(
            rung=Rung.GREY,
            verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL),
        )
        # Simulate 4+ model history entries (already retried)
        d = d.model_copy(update={"artifacts": d.artifacts.model_copy(update={
            "model_history": ["m1", "m2", "m3", "m4"]
        })})
        node, _ = _rule_based_fallback(d)
        assert node == SURROGATE_SO

    def test_unmodelable_with_physics_at_white_goes_to_greybox(self):
        d = _dossier(
            rung=Rung.WHITE,
            verdict=_verdict(VerdictResult.FAIL, GapType.UNMODELABLE),
            physics=PhysicsAvailability.FULL,
        )
        node, _ = _rule_based_fallback(d)
        assert node == GREYBOX_SO

    def test_unmodelable_no_physics_at_white_goes_to_surrogate(self):
        d = _dossier(
            rung=Rung.WHITE,
            verdict=_verdict(VerdictResult.FAIL, GapType.UNMODELABLE),
            physics=PhysicsAvailability.NONE,
        )
        node, _ = _rule_based_fallback(d)
        assert node == SURROGATE_SO

    def test_unmodelable_at_grey_goes_to_surrogate(self):
        d = _dossier(
            rung=Rung.GREY,
            verdict=_verdict(VerdictResult.FAIL, GapType.UNMODELABLE),
        )
        node, _ = _rule_based_fallback(d)
        assert node == SURROGATE_SO

    def test_black_rung_always_ships(self):
        d = _dossier(
            rung=Rung.BLACK,
            verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL),
        )
        node, _ = _rule_based_fallback(d)
        assert node == SHIP

    def test_no_verdict_ships_defensively(self):
        d = _dossier(verdict=None)
        node, _ = _rule_based_fallback(d)
        assert node == SHIP

    def test_full_escalation_path(self):
        """white → grey → surrogate → ship."""
        d1 = _dossier(rung=Rung.WHITE, verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL))
        assert _rule_based_fallback(d1)[0] == GREYBOX_SO

        d2 = _dossier(rung=Rung.GREY, verdict=_verdict(VerdictResult.FAIL, GapType.STRUCTURED_RESIDUAL))
        d2 = d2.model_copy(update={"artifacts": d2.artifacts.model_copy(update={
            "model_history": ["m1", "m2", "m3", "m4"]
        })})
        assert _rule_based_fallback(d2)[0] == SURROGATE_SO

        d3 = _dossier(rung=Rung.BLACK, verdict=_verdict(VerdictResult.PASS, GapType.NONE))
        assert _rule_based_fallback(d3)[0] == SHIP
