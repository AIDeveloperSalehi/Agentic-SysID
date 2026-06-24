"""
Blackboard helpers — typed read/write operations on the Dossier.

The orchestrator calls these instead of mutating the Dossier directly,
keeping all field-level logic in one place.
"""
from __future__ import annotations

from typing import List, Optional

from core.schemas import (
    ArtifactRef,
    Critique,
    Dossier,
    Report,
    Rung,
    Verdict,
)


def post_report(dossier: Dossier, report: Report) -> Dossier:
    """Record the last agent report on the dossier."""
    return dossier.update(last_report=report, status=f"{report.agent}: {report.summary[:80]}")


def post_verdict(dossier: Dossier, verdict: Verdict) -> Dossier:
    """Record a validation verdict and any embedded critique."""
    updates: dict = {"last_verdict": verdict}

    if verdict.critique_id:
        # The validation agent already stored the critique content; here we
        # open a stub so the orchestrator can see it in open_critiques.
        critique = Critique(
            id=verdict.critique_id,
            addressed_to="modeler",   # default; validation refines in metadata
            ref=verdict.validity_region_id or "",
        )
        open_critiques = list(dossier.open_critiques) + [critique]
        updates["open_critiques"] = open_critiques

    return dossier.update(**updates)


def add_artifact(dossier: Dossier, ref: ArtifactRef) -> Dossier:
    """Register a new artifact in the dossier's artifact list."""
    arts = dossier.artifacts.model_copy(deep=True)

    if ref.type == "model":
        history = list(arts.model_history)
        if arts.current_model_id:
            history.append(arts.current_model_id)
        arts = arts.model_copy(update={
            "current_model_id": ref.id,
            "model_history": history,
        })
    elif ref.type == "dataset":
        ids = list(arts.dataset_ids) + [ref.id]
        arts = arts.model_copy(update={"dataset_ids": ids})
    elif ref.type == "validity_region":
        rids = list(arts.validation_report_ids) + [ref.id]
        arts = arts.model_copy(update={"validation_report_ids": rids})

    return dossier.update(artifacts=arts)


def advance_rung(dossier: Dossier, next_rung: Rung) -> Dossier:
    """Descend one step on the fidelity ladder (white→grey→black)."""
    return dossier.update(current_rung=next_rung, status=f"escalated to {next_rung.value}-box")


def mark_critique_addressed(dossier: Dossier, critique_id: str) -> Dossier:
    """Flip a critique from open → addressed."""
    critiques = [
        c.model_copy(update={"status": "addressed"}) if c.id == critique_id else c
        for c in dossier.open_critiques
    ]
    return dossier.update(open_critiques=critiques)


def debit_budget(dossier: Dossier, cost: float) -> Dossier:
    """Debit the dossier's embedded budget copy (for display; real debit is in BudgetManager)."""
    return dossier.update(budget=dossier.budget.debit(cost))
