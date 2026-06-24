"""
Model class and paradigm selection for the surrogate sub-orchestrator.

Paradigm A (ODE-based)
  The surrogate learns a dynamics equation  f: (state, u) → highest_derivative
  and integrates it via an ODE solver.  Matches the physics backbone.
  Model classes: GP, NN

Paradigm B (input-output sequence model)
  The surrogate learns directly from observed (y, u) sequences without needing
  state reconstruction.  Works when state estimates are unreliable.
  Model classes: NARX, RNN, TRANSFORMER

Paradigm selection is based on physics availability:
  FULL/PARTIAL → Paradigm A  (physics-informed structure available)
  NONE         → Paradigm B  (no physics structure; learn from raw I/O)

Within each paradigm, model class is chosen by dataset size.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GP_THRESHOLD = 500   # samples — GP below, NN above (Paradigm A)

# Paradigm B thresholds
NARX_THRESHOLD        = 400   # NARX below, RNN above
RNN_THRESHOLD         = 2000  # RNN below, Transformer above


class Paradigm(str, Enum):
    ODE          = "ode"           # learns dynamics eq; integrates via ODE solver
    INPUT_OUTPUT = "input_output"  # learns sequence model from observed I/O


class ModelClass(str, Enum):
    # Paradigm A — ODE-based
    GP          = "gp"           # exact RBF-GP; N ≤ GP_THRESHOLD
    NN          = "nn"           # small MLP; N > GP_THRESHOLD
    # Paradigm B — input-output sequence models
    NARX        = "narx"         # Nonlinear AutoRegressive with eXogenous inputs
    RNN         = "rnn"          # LSTM recurrent network
    TRANSFORMER = "transformer"  # causal decoder-only Transformer


@dataclass
class SelectionResult:
    model_class: ModelClass
    paradigm:    Paradigm
    n_samples:   int
    rationale:   str


class ModelClassSelector:
    """Selects paradigm and model class from dataset size and physics availability."""

    def select(
        self,
        n_samples:  int,
        paradigm:   Paradigm = Paradigm.ODE,
    ) -> SelectionResult:
        """
        Select model class within the given paradigm.

        Backward-compatible: ``select(n)`` defaults to Paradigm.ODE and returns
        GP or NN, identical to the original behavior.
        """
        if paradigm == Paradigm.ODE:
            return self._select_ode(n_samples)
        else:
            return self._select_io(n_samples)

    def select_paradigm(
        self,
        physics_available: bool,
    ) -> Paradigm:
        """Decide which paradigm to use based on physics availability."""
        return Paradigm.ODE if physics_available else Paradigm.INPUT_OUTPUT

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _select_ode(n_samples: int) -> SelectionResult:
        if n_samples <= GP_THRESHOLD:
            return SelectionResult(
                model_class=ModelClass.GP,
                paradigm=Paradigm.ODE,
                n_samples=n_samples,
                rationale=(
                    f"N={n_samples} ≤ {GP_THRESHOLD}: "
                    "ODE paradigm — numpy RBF-GP (exact, calibrated uncertainty)"
                ),
            )
        return SelectionResult(
            model_class=ModelClass.NN,
            paradigm=Paradigm.ODE,
            n_samples=n_samples,
            rationale=(
                f"N={n_samples} > {GP_THRESHOLD}: "
                "ODE paradigm — torch MLP (scalable)"
            ),
        )

    @staticmethod
    def _select_io(n_samples: int) -> SelectionResult:
        if n_samples <= NARX_THRESHOLD:
            return SelectionResult(
                model_class=ModelClass.NARX,
                paradigm=Paradigm.INPUT_OUTPUT,
                n_samples=n_samples,
                rationale=(
                    f"N={n_samples} ≤ {NARX_THRESHOLD}: "
                    "I/O paradigm — NARX with GP regressor"
                ),
            )
        elif n_samples <= RNN_THRESHOLD:
            return SelectionResult(
                model_class=ModelClass.RNN,
                paradigm=Paradigm.INPUT_OUTPUT,
                n_samples=n_samples,
                rationale=(
                    f"N={n_samples} ≤ {RNN_THRESHOLD}: "
                    "I/O paradigm — LSTM recurrent network"
                ),
            )
        return SelectionResult(
            model_class=ModelClass.TRANSFORMER,
            paradigm=Paradigm.INPUT_OUTPUT,
            n_samples=n_samples,
            rationale=(
                f"N={n_samples} > {RNN_THRESHOLD}: "
                "I/O paradigm — causal decoder Transformer"
            ),
        )
