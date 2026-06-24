"""
Central contract types for the agentic system-identification pipeline.

Every agent, tool, and store speaks these types.  Nothing routes on prose —
it routes on the enum fields defined here.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


def _uid() -> str:
    return str(uuid.uuid4())[:8]


# ── Enumerations ──────────────────────────────────────────────────────────────

class EntryPath(str, Enum):
    WHITE_BOX  = "white-box"
    SIMULATOR  = "simulator"
    SURROGATE  = "surrogate"


class Rung(str, Enum):
    WHITE = "white"
    GREY  = "grey"
    BLACK = "black"


class GapType(str, Enum):
    NONE                = "none"
    FIXABLE             = "fixable"
    STRUCTURED_RESIDUAL = "structured_residual"
    UNMODELABLE         = "unmodelable"


class VerdictResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class AgentStatus(str, Enum):
    DONE             = "done"
    NEEDS_USER_INPUT = "needs_user_input"
    FAILED           = "failed"


class PhysicsAvailability(str, Enum):
    FULL    = "full"
    PARTIAL = "partial"
    NONE    = "none"


class ModelType(str, Enum):
    WHITE_BOX = "white_box"
    GREY_BOX  = "grey_box"
    BLACK_BOX = "black_box"


class ExcitationQuality(str, Enum):
    SUFFICIENT = "sufficient_for_identification"
    VALIDATION_ONLY = "validation_only"
    INSUFFICIENT    = "insufficient"


class SplitFlag(str, Enum):
    TRAIN      = "train"
    VALIDATION = "validation"
    BOTH       = "both"


class SafetyStatus(str, Enum):
    OK      = "ok"
    CLIPPED = "clipped"
    ABORTED = "aborted"


# ── Sub-objects ───────────────────────────────────────────────────────────────

class ArtifactRef(BaseModel):
    id:    str
    type:  str   # "model" | "dataset" | "validity_region" | "covariance"
    store: str   # "registry" | "database" | "results"


class Budget(BaseModel):
    total:             float
    spent:             float = 0.0
    remaining:         Optional[float] = None
    slice_allocations: Dict[str, float] = {}

    @model_validator(mode="after")
    def _sync_remaining(self) -> "Budget":
        if self.remaining is None:
            self.remaining = self.total - self.spent
        return self

    def debit(self, cost: float) -> "Budget":
        new_spent = self.spent + cost
        return self.model_copy(update={
            "spent": new_spent,
            "remaining": self.total - new_spent,
        })

    def allocate_slice(self, name: str, amount: float) -> "Budget":
        allocs = dict(self.slice_allocations)
        allocs[name] = amount
        return self.model_copy(update={"slice_allocations": allocs})

    @property
    def exhausted(self) -> bool:
        return (self.remaining or 0.0) <= 0.0


class Assets(BaseModel):
    plant_contract_id: Optional[str]           = None
    physics:           PhysicsAvailability     = PhysicsAvailability.NONE
    simulator_id:      Optional[str]           = None
    surrogate_id:      Optional[str]           = None
    user_data_ids:     List[str]               = []


class Artifacts(BaseModel):
    current_model_id:      Optional[str] = None
    model_history:         List[str]     = []
    dataset_ids:           List[str]     = []
    validation_report_ids: List[str]     = []
    best_model_id:         Optional[str]   = None   # model with lowest val RMSE seen so far
    best_val_rmse:         Optional[float] = None   # None = no validated model yet


class Critique(BaseModel):
    id:           str = Field(default_factory=_uid)
    addressed_to: str
    ref:          str
    status:       Literal["open", "addressed"] = "open"
    description:  str = ""


class WorstCase(BaseModel):
    region:   Dict[str, Any]
    error:    float
    scenario: Optional[str] = None


# ── Attempt log ──────────────────────────────────────────────────────────────

class AttemptEntry(BaseModel):
    """
    One modelling attempt, written to the shared dossier after each validation.

    Every agent — estimator, greybox, surrogate — appends one of these so that
    subsequent agents (and the LLM router) can reason about what was tried and
    why it succeeded or failed.
    """
    rung:            str                    # "white" | "grey" | "black"
    agent:           str                    # e.g. "estimator", "greybox", "surrogate"
    model_class:     str                    # e.g. "NLS", "SINDY+seq", "RNN", "TRANSFORMER"
    n_train:         int                    # number of training samples used
    epochs:          Optional[int] = None   # training epochs (surrogate only)
    train_rmse:      float = float("nan")
    val_rmse:        Optional[float] = None # worst-case across scenarios; None = not yet validated
    scenario_rmse:   Dict[str, float] = {}  # per-scenario breakdown
    gap_type:        str = "unknown"         # from Verdict
    agent_reasoning: str = ""               # what the agent said it was doing / why


# ── Core routing objects ──────────────────────────────────────────────────────

class Verdict(BaseModel):
    """Posted by the validation agent; the orchestrator routes on this — never computes it."""
    verdict:                VerdictResult
    gap_type:               GapType
    metrics:                Dict[str, Any]     = {}
    validity_region_id:     Optional[str]      = None
    uncertainty_calibrated: bool               = False
    worst_case:             Optional[WorstCase] = None
    critique_id:            Optional[str]      = None
    failure_hypothesis:     Optional[str]      = None   # LLM diagnosis: what failed and why
    worst_case_inputs:      Optional[Dict[str, Any]] = None  # worst probe scenario details


class Report(BaseModel):
    """Every agent posts exactly one of these when it finishes."""
    agent:            str
    task_id:          str = Field(default_factory=_uid)
    status:           AgentStatus
    produced:         List[ArtifactRef]  = []
    summary:          str
    recommended_next: Optional[str]      = None
    metadata:         Dict[str, Any]     = {}


# ── Dossier (the blackboard) ──────────────────────────────────────────────────

class ExperimentPlan(BaseModel):
    """
    Produced by ExperimentPlannerAgent before each estimator invocation.
    Tells the estimator *what* to try; the estimator decides *how* to fit.
    """
    methods:            List[str]               # ordered list e.g. ["prbs", "multisine", "steps"]
    base_amplitude:     float                   # starting amplitude fraction (0–1)
    max_amplitude:      float                   # ceiling amplitude fraction (0–1)
    seg_len:            int                     # multi-shooting segment length in samples
    reasoning:          str                     # one sentence: why these choices
    amplitude_schedule: Optional[List[float]] = None  # explicit per-iteration amplitudes;
                                                      # overrides the base→max ramp when provided


class Dossier(BaseModel):
    """
    Lean shared state owned by the orchestrator.
    Contains only pointers and status — never raw data.
    """
    id:                 str        = Field(default_factory=_uid)
    entry_path:         EntryPath
    current_rung:       Rung       = Rung.WHITE
    status:             str        = "initialized"
    assets:             Assets     = Field(default_factory=Assets)
    budget:             Budget
    artifacts:          Artifacts  = Field(default_factory=Artifacts)
    open_critiques:     List[Critique]    = []
    last_verdict:       Optional[Verdict] = None
    last_report:        Optional[Report]  = None
    attempt_log:        List[AttemptEntry] = []   # full history of modelling attempts
    re_estimate_count:  int        = 0             # times router sent back to Estimator
    experiment_plan:    Optional[ExperimentPlan] = None  # set by ExperimentPlannerAgent
    created_at:         datetime   = Field(default_factory=datetime.utcnow)
    updated_at:         datetime   = Field(default_factory=datetime.utcnow)

    def update(self, **kwargs) -> "Dossier":
        """Immutable update — returns a new Dossier."""
        return self.model_copy(update={**kwargs, "updated_at": datetime.utcnow()})


# ── Plant contract ────────────────────────────────────────────────────────────

class PlantContract(BaseModel):
    id:                str  = Field(default_factory=_uid)
    name:              str
    input_names:       List[str]
    output_names:      List[str]
    state_names:       List[str]                             = []
    input_limits:      Dict[str, Tuple[float, float]]        # {name: (min, max)}
    input_rate_limits: Dict[str, float]                      = {}
    output_limits:     Dict[str, Tuple[float, float]]        = {}
    sample_time:       float
    x0:                Optional[List[float]]                 = None   # known initial state; None = unknown
    is_unstable:       bool                                  = False
    operating_envelope: Optional[Dict[str, Tuple[float, float]]] = None
    description:       str  = ""


# ── Model artifact ────────────────────────────────────────────────────────────

class ModelArtifact(BaseModel):
    id:                      str = Field(default_factory=_uid)
    version:                 int = 1
    model_type:              ModelType
    structure_description:   str
    parameters:              Dict[str, float]  = {}
    parameter_covariance_id: Optional[str]     = None
    uncertainty_id:          Optional[str]     = None
    validity_region_id:      Optional[str]     = None
    parent_id:               Optional[str]     = None
    created_at:              datetime          = Field(default_factory=datetime.utcnow)
    metadata:                Dict[str, Any]    = {}


# ── Experiment run record ─────────────────────────────────────────────────────

class ExperimentRun(BaseModel):
    """Metadata row stored in the DB.  Raw arrays live in data/runs/{id}.npz."""
    id:               str  = Field(default_factory=_uid)
    input_type:       str                          # "prbs" | "multisine" | "step" | "adversarial"
    purpose:          str                          # "identification" | "validation" | "active_sampling"
    originating_agent: str
    cost:             float                        = 1.0
    safety_status:    SafetyStatus                 = SafetyStatus.OK
    split_flag:       SplitFlag                    = SplitFlag.TRAIN
    provenance:       Literal["system", "user_supplied"] = "system"
    n_samples:        int                          = 0
    operating_region: Dict[str, Tuple[float, float]] = {}
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    metadata:         Dict[str, Any] = {}


# ── Data steward report ───────────────────────────────────────────────────────

class DataCoverageReport(BaseModel):
    run_ids:    List[str]
    coverage:   Dict[str, Tuple[float, float]]
    excitation: ExcitationQuality
    quality:    float                              # 0–1
    usable_for: List[Literal["identify", "validate", "train"]]
    gaps:       List[str]


# ── Identifiability report ────────────────────────────────────────────────────

class IdentifiabilityResult(str, Enum):
    FULL    = "full"
    PARTIAL = "partial"
    NONE    = "none"


class IdentifiabilityReport(BaseModel):
    identifiable:             IdentifiabilityResult
    non_identifiable_params:  List[str] = []
    recommendation:           str       = ""
    reparameterized_model_id: Optional[str] = None


# ── Validity region ───────────────────────────────────────────────────────────

class ValidityRegion(BaseModel):
    id:            str = Field(default_factory=_uid)
    model_id:      str
    bounds:        Dict[str, Tuple[float, float]]   # variable → (min, max) certified range
    tolerance:     float                             # max RMSE within this region
    achieved_rmse: float
    coverage_fraction: float                         # fraction of test points that pass
    created_at:    datetime = Field(default_factory=datetime.utcnow)
    metadata:      Dict[str, Any] = {}
