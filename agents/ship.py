"""
Ship agent — final delivery node.

Accepts the pipeline's best dossier, loads the current model from the
registry, writes a human-readable delivery summary to stdout, and
returns the dossier tagged "shipped".

No LLM calls are made here.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.schemas import AgentStatus, ArtifactRef, Dossier, Report, Rung
from tools.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_RUNG_TO_PATH = {
    Rung.WHITE: "white-box",
    Rung.GREY:  "grey-box",
    Rung.BLACK: "surrogate",
}


class ShipAgent:
    """
    Terminal pipeline node — logs the delivered model and marks the dossier shipped.
    Stores a RunSummary in episodic memory when a RetrievalService is provided.
    """

    def __init__(self, registry: ModelRegistry, retrieval_service=None):
        self._registry  = registry
        self._retrieval = retrieval_service

    def __call__(self, dossier: Dossier) -> Dossier:
        # Prefer the model with the lowest validated RMSE over the most-recently
        # produced model — the last agent to run may have made things worse.
        best_id    = dossier.artifacts.best_model_id
        current_id = dossier.artifacts.current_model_id
        model_id   = best_id if best_id else current_id

        if not model_id:
            report = Report(
                agent="ShipAgent",
                status=AgentStatus.FAILED,
                summary="Ship: no model_id in dossier — nothing to deliver.",
            )
            return dossier.update(status="shipped: no model", last_report=report)

        try:
            model  = self._registry.load_model(model_id)
            params = dict(model.parameters)
            rhs    = model.metadata.get("normalized_rhs", "—")
            mc     = model.metadata.get("model_class", "")
            mc_str = f" [{mc}]" if mc else ""
            # Derive rung from the shipped model's type rather than dossier.current_rung,
            # which may reflect the last agent run (not necessarily the best model).
            rung_val = {
                "white_box": Rung.WHITE,
                "grey_box":  Rung.GREY,
                "black_box": Rung.BLACK,
            }.get(model.model_type.value, dossier.current_rung)
            val_rmse = dossier.artifacts.best_val_rmse
            rmse_str = f", val_rmse={val_rmse:.4f}" if val_rmse is not None else ""
            summary = (
                f"Delivered {model.model_type.value}{mc_str} model "
                f"(rung={rung_val.value}, id={model_id}{rmse_str}). "
                f"Budget spent: {dossier.budget.spent:.1f}/{dossier.budget.total:.1f}. "
                f"Params: {params if params else 'n/a (surrogate)'}. "
                f"RHS: {rhs[:80]}{'…' if len(rhs) > 80 else ''}."
            )
        except Exception as exc:
            rung_val = dossier.current_rung
            summary  = (
                f"Shipped model {model_id} "
                f"(rung={rung_val.value}, budget={dossier.budget.spent:.1f}). "
                f"Could not load model details: {exc}"
            )
            params = {}
            rhs    = ""

        report = Report(
            agent="ShipAgent",
            status=AgentStatus.DONE,
            summary=summary,
            produced=[ArtifactRef(id=model_id, type="model", store="registry")],
            metadata={
                "model_id":     model_id,
                "rung":         rung_val.value,
                "budget_spent": dossier.budget.spent,
                "params":       params,
            },
        )

        logger.info("ShipAgent: %s", summary)
        updated = dossier.update(
            current_rung=rung_val,
            status=f"shipped: {model_id}",
            last_report=report,
        )

        if self._retrieval is not None:
            self._store_run_summary(updated, model_id, params, rhs)

        return updated

    def _store_run_summary(self, dossier: Dossier, model_id: str, params: dict, rhs: str) -> None:
        try:
            from memory.schemas import RunSummary
            model    = self._registry.load_model(model_id)
            meta     = model.metadata

            plant_name  = ""
            plant_desc  = ""
            contract_id = dossier.assets.plant_contract_id or ""
            if contract_id:
                try:
                    ca        = self._registry.load_model(contract_id)
                    pc        = ca.metadata.get("plant_contract", {})
                    plant_name = pc.get("name", "")
                    plant_desc = pc.get("description", "")
                except Exception:
                    pass

            rmse = -1.0
            if dossier.last_verdict and dossier.last_verdict.metrics:
                rmse = float(dossier.last_verdict.metrics.get("rmse", -1.0))

            corrections: list = []
            if dossier.current_rung == Rung.GREY:
                for term in ("tanh", "sign", "coulomb", "K_c", "f_c"):
                    if term.lower() in rhs.lower():
                        corrections.append("Coulomb friction")
                        break

            summary = RunSummary(
                plant_name=plant_name or model_id,
                plant_description=plant_desc or dossier.status,
                state_vars=meta.get("state_vars", []),
                input_vars=meta.get("input_vars", []),
                system_order=meta.get("system_order", len(meta.get("state_vars", []))),
                final_rhs=rhs,
                fitted_params={k: float(v) for k, v in params.items()},
                param_bounds={k: list(v) for k, v in meta.get("param_bounds", {}).items()},
                path_taken=_RUNG_TO_PATH.get(dossier.current_rung, "unknown"),
                corrections_applied=corrections,
                model_type=model.model_type.value,
                validation_rmse=rmse,
                budget_spent=float(dossier.budget.spent),
            )
            self._retrieval.episodic.add_run(summary)
            logger.info("ShipAgent: stored run summary in episodic memory (%s)", summary.run_id)
        except Exception as exc:
            logger.warning("ShipAgent: failed to store run summary: %s", exc)
