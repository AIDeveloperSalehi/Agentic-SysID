"""
Versioned model artifact registry.

Models are stored as JSON (parameters, metadata) + optional pickle (for complex
objects like GP models or neural networks).  Everything is referenced by ID.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schemas import ModelArtifact, ModelType, ValidityRegion


class ModelRegistry:
    """
    File-based model store.

    store_model      — persist a ModelArtifact; returns its id
    load_model       — retrieve ModelArtifact by id
    store_object     — store an arbitrary Python object (GP, NN, etc.) as pickle
    load_object      — retrieve it
    store_validity   — persist a ValidityRegion
    load_validity    — retrieve it
    list_models      — filter by type, sorted by version
    """

    def __init__(self, base_dir: str = "data/models"):
        self._base = Path(base_dir)
        self._models_dir   = self._base / "artifacts"
        self._objects_dir  = self._base / "objects"
        self._validity_dir = self._base / "validity"
        for d in (self._models_dir, self._objects_dir, self._validity_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Model artifacts ───────────────────────────────────────────────────────

    def store_model(self, model: ModelArtifact) -> str:
        path = self._models_dir / f"{model.id}.json"
        path.write_text(model.model_dump_json(indent=2))
        return model.id

    def load_model(self, model_id: str) -> ModelArtifact:
        path = self._models_dir / f"{model_id}.json"
        if not path.exists():
            raise KeyError(f"Model {model_id} not found in registry")
        return ModelArtifact.model_validate_json(path.read_text())

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        parent_id:  Optional[str] = None,
    ) -> List[ModelArtifact]:
        models = []
        for p in sorted(self._models_dir.glob("*.json")):
            m = ModelArtifact.model_validate_json(p.read_text())
            if model_type and m.model_type != model_type:
                continue
            if parent_id and m.parent_id != parent_id:
                continue
            models.append(m)
        return sorted(models, key=lambda m: m.created_at)

    def latest_model(self, model_type: Optional[ModelType] = None) -> Optional[ModelArtifact]:
        models = self.list_models(model_type=model_type)
        return models[-1] if models else None

    # ── Arbitrary objects (GP, NN weights, covariance matrices) ──────────────

    def store_object(self, obj_id: str, obj: Any) -> str:
        path = self._objects_dir / f"{obj_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj_id

    def load_object(self, obj_id: str) -> Any:
        path = self._objects_dir / f"{obj_id}.pkl"
        if not path.exists():
            raise KeyError(f"Object {obj_id} not found")
        with open(path, "rb") as f:
            return pickle.load(f)

    def store_covariance(self, cov_id: str, cov_matrix) -> str:
        """Store a numpy covariance matrix."""
        import numpy as np
        path = self._objects_dir / f"{cov_id}.npy"
        np.save(str(path), cov_matrix)
        return cov_id

    def load_covariance(self, cov_id: str):
        import numpy as np
        path = self._objects_dir / f"{cov_id}.npy"
        if not path.exists():
            raise KeyError(f"Covariance {cov_id} not found")
        return np.load(str(path))

    # ── Validity regions ──────────────────────────────────────────────────────

    def store_validity(self, region: ValidityRegion) -> str:
        path = self._validity_dir / f"{region.id}.json"
        path.write_text(region.model_dump_json(indent=2))
        return region.id

    def load_validity(self, region_id: str) -> ValidityRegion:
        path = self._validity_dir / f"{region_id}.json"
        if not path.exists():
            raise KeyError(f"ValidityRegion {region_id} not found")
        return ValidityRegion.model_validate_json(path.read_text())

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        return {
            "models":   len(list(self._models_dir.glob("*.json"))),
            "objects":  len(list(self._objects_dir.glob("*"))),
            "validity": len(list(self._validity_dir.glob("*.json"))),
        }
