"""
LangGraph orchestrator — agentic routing state machine.

Post-validation routing is handled by RouterAgent (LLM), which reads the full
attempt log and decides: greybox_so | surrogate_so | ship.

All other routing (intake → modeler, modeler → estimator, etc.) remains
deterministic — these are structural transitions with no ambiguity.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from core.schemas import Dossier, EntryPath

log = logging.getLogger(__name__)


# ── Node names ────────────────────────────────────────────────────────────────

INTAKE              = "intake"
MODELER             = "modeler"
EXPERIMENT_PLANNER  = "experiment_planner"
ESTIMATOR           = "estimator"
VALIDATION          = "validation"
ROUTER              = "router"
GREYBOX_SO          = "greybox_so"
SURROGATE_SO        = "surrogate_so"
SHIP                = "ship"


# ── Structural routing (deterministic — no ambiguity) ─────────────────────────

def _route_after_intake(state: Dossier) -> str:
    if state.entry_path == EntryPath.WHITE_BOX:
        return MODELER
    return VALIDATION


def _route_after_modeler(state: Dossier) -> str:
    return EXPERIMENT_PLANNER


def _route_after_experiment_planner(state: Dossier) -> str:
    return ESTIMATOR


def _route_after_estimator(state: Dossier) -> str:
    return VALIDATION


def _route_after_validation(state: Dossier) -> str:
    """After validation: always go to the LLM router for a reasoned decision."""
    return ROUTER


def _route_after_router(state: Dossier) -> str:
    """
    Read the RouterAgent's decision from last_report.recommended_next.
    Falls back to 'ship' if the field is missing or invalid.
    When the router chooses 'estimator', route through the ExperimentPlannerAgent
    first so the planner sees the full history before the estimator runs.
    """
    VALID = {ESTIMATOR, GREYBOX_SO, SURROGATE_SO, SHIP}
    decision = (
        state.last_report.recommended_next
        if state.last_report and state.last_report.recommended_next
        else "ship"
    )
    if decision not in VALID:
        log.warning("[orchestrator] RouterAgent returned invalid node '%s' — shipping", decision)
        return SHIP
    log.info("[orchestrator] RouterAgent → %s", decision)
    # Route re-estimation attempts through the planner first.
    if decision == ESTIMATOR:
        return EXPERIMENT_PLANNER
    return decision


def _route_after_suborch(state: Dossier) -> str:
    """After any sub-orchestrator (greybox / surrogate): always validate."""
    return VALIDATION


# ── Skeleton graph (for routing tests without LLM) ────────────────────────────

def build_graph() -> StateGraph:
    """Build a skeleton graph with stub nodes for routing tests."""
    graph = StateGraph(Dossier)

    for name in [INTAKE, MODELER, EXPERIMENT_PLANNER, ESTIMATOR, VALIDATION, ROUTER,
                 GREYBOX_SO, SURROGATE_SO, SHIP]:
        graph.add_node(name, _stub_node(name))

    graph.set_entry_point(INTAKE)
    graph.add_conditional_edges(INTAKE,             _route_after_intake,             _all_nodes())
    graph.add_conditional_edges(MODELER,            _route_after_modeler,            _all_nodes())
    graph.add_conditional_edges(EXPERIMENT_PLANNER, _route_after_experiment_planner, _all_nodes())
    graph.add_conditional_edges(ESTIMATOR,          _route_after_estimator,          _all_nodes())
    graph.add_conditional_edges(VALIDATION,         _route_after_validation,         _all_nodes())
    graph.add_conditional_edges(ROUTER,             _route_after_router,             _all_nodes())
    graph.add_conditional_edges(GREYBOX_SO,         _route_after_suborch,            _all_nodes())
    graph.add_conditional_edges(SURROGATE_SO,       _route_after_suborch,            _all_nodes())
    graph.add_edge(SHIP, END)

    return graph


# ── Real graph (production) ───────────────────────────────────────────────────

def build_real_graph(
    plant_api,
    registry,
    db,
    budget_total:      float = 200.0,
    api_key:           Optional[str] = None,
    model:             str = "claude-sonnet-4-6",
    n_samples:         int = 600,
    nn_epochs:         int = 200,
    retrieval_service: Optional[object] = None,
) -> StateGraph:
    """
    Build a LangGraph StateGraph with all real agents wired in.

    Graph topology:

        intake → modeler → experiment_planner → estimator → validation → ROUTER
                                    ↑                            ↑             ↓
                                    └────────────────────────────┘        greybox_so
                                    (router routes to experiment_planner        ↓
                                     before every estimator retry)        surrogate_so
                                                                               ↓
                                                                         [validation] → ROUTER → ship

    ExperimentPlannerAgent sits before every estimator invocation (both initial and retries).
    It reads the full dossier history and emits an ExperimentPlan that the estimator follows
    instead of its hard-coded defaults.
    """
    from agents.intake import IntakeAgent
    from agents.modeler import ModelerAgent
    from agents.experiment_planner import ExperimentPlannerAgent
    from agents.estimator import EstimatorAgent
    from agents.validation import ValidationAgent
    from agents.greybox.agent import GreyBoxAgent
    from agents.surrogate.agent import SurrogateAgent
    from agents.ship import ShipAgent
    from core.router_agent import RouterAgent

    g = StateGraph(Dossier)

    g.add_node(INTAKE,             IntakeAgent(registry, budget_total, model, api_key))
    g.add_node(MODELER,            ModelerAgent(registry, model, api_key,
                                                retrieval_service=retrieval_service))
    g.add_node(EXPERIMENT_PLANNER, ExperimentPlannerAgent(model=model, api_key=api_key))
    g.add_node(ESTIMATOR,          EstimatorAgent(plant_api, registry, db, n_samples=n_samples,
                                                  retrieval_service=retrieval_service))
    g.add_node(VALIDATION,         ValidationAgent(plant_api, registry, db))
    g.add_node(ROUTER,             RouterAgent(model=model, api_key=api_key))
    g.add_node(GREYBOX_SO,         GreyBoxAgent(plant_api, registry, db,
                                                model=model, api_key=api_key, n_samples=n_samples))
    g.add_node(SURROGATE_SO,       SurrogateAgent(plant_api, registry, db,
                                                  model=model, api_key=api_key))
    g.add_node(SHIP,               ShipAgent(registry, retrieval_service=retrieval_service))

    g.set_entry_point(INTAKE)

    g.add_conditional_edges(INTAKE,             _route_after_intake,             _all_nodes())
    g.add_conditional_edges(MODELER,            _route_after_modeler,            _all_nodes())
    g.add_conditional_edges(EXPERIMENT_PLANNER, _route_after_experiment_planner, _all_nodes())
    g.add_conditional_edges(ESTIMATOR,          _route_after_estimator,          _all_nodes())
    g.add_conditional_edges(VALIDATION,         _route_after_validation,         _all_nodes())
    g.add_conditional_edges(ROUTER,             _route_after_router,             _all_nodes())
    g.add_conditional_edges(GREYBOX_SO,         _route_after_suborch,            _all_nodes())
    g.add_conditional_edges(SURROGATE_SO,       _route_after_suborch,            _all_nodes())
    g.add_edge(SHIP, END)

    return g


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stub_node(name: str):
    def _node(state: Dossier) -> Dossier:
        return state.update(status=f"stub:{name}")
    _node.__name__ = name
    return _node


def _all_nodes() -> dict:
    names = [INTAKE, MODELER, EXPERIMENT_PLANNER, ESTIMATOR, VALIDATION, ROUTER,
             GREYBOX_SO, SURROGATE_SO, SHIP, END]
    return {n: n for n in names}


def replace_node(graph: StateGraph, name: str, fn) -> None:
    graph.nodes[name] = fn
