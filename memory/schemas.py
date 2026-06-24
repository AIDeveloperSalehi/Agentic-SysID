from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


def _uid() -> str:
    return str(uuid.uuid4())[:8]


class RunSummary(BaseModel):
    """Episodic memory record for one completed identification run."""
    run_id:               str              = Field(default_factory=_uid)
    plant_name:           str
    plant_description:    str
    state_vars:           List[str]
    input_vars:           List[str]
    system_order:         int
    final_rhs:            str
    fitted_params:        Dict[str, float] = {}
    param_bounds:         Dict[str, List[float]] = {}
    path_taken:           str              # "white-box" | "grey-box" | "surrogate"
    corrections_applied:  List[str]        = []
    model_type:           str
    validation_rmse:      float            = -1.0
    budget_spent:         float            = 0.0
    created_at:           str              = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_embedding_text(self) -> str:
        """Text that gets embedded — optimised for retrieval, not reading."""
        corrections = ", ".join(self.corrections_applied) or "none"
        params_str  = ", ".join(f"{k}={v:.3f}" for k, v in self.fitted_params.items())
        return (
            f"Plant: {self.plant_description}\n"
            f"System: states={self.state_vars}, inputs={self.input_vars}, order={self.system_order}\n"
            f"ODE: {self.final_rhs}\n"
            f"Path: {self.path_taken}. Corrections: {corrections}\n"
            f"Fitted: {params_str}"
        )

    def to_context_string(self) -> str:
        """Human-readable summary injected into agent context."""
        params_str  = ", ".join(f"{k}={v:.4f}" for k, v in self.fitted_params.items())
        bounds_str  = "\n".join(
            f"    {k}: [{v[0]:.2f}, {v[1]:.2f}]" for k, v in self.param_bounds.items()
        )
        corrections = ", ".join(self.corrections_applied) or "none"
        rmse_str    = f"{self.validation_rmse:.4f}" if self.validation_rmse >= 0 else "n/a"
        return (
            f"Plant   : {self.plant_name} — {self.plant_description[:120]}\n"
            f"ODE     : {self.final_rhs}\n"
            f"Fitted  : {params_str}\n"
            f"Bounds  :\n{bounds_str}\n"
            f"Path    : {self.path_taken} | Corrections: {corrections}\n"
            f"RMSE    : {rmse_str} | Budget: {self.budget_spent:.1f}"
        )


class DocumentChunk(BaseModel):
    """One retrievable chunk from a reference document."""
    chunk_id:    str = Field(default_factory=_uid)
    source:      str
    section:     str = ""
    text:        str
    system_type: str = ""
    created_at:  str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_embedding_text(self) -> str:
        prefix = f"[{self.system_type}] " if self.system_type else ""
        header = f"{prefix}{self.section}\n" if self.section else prefix
        return f"{header}{self.text}"

    def to_context_string(self) -> str:
        header = f"[{self.source}]" + (f" — {self.section}" if self.section else "")
        return f"{header}\n{self.text}"
