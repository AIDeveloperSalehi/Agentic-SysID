"""
Global budget manager.

Tracks total/spent/remaining experiment budget, allocates per-phase slices,
debits each run, and enforces the global stopping rule.
No loop can spin past a depleted budget.
"""
from __future__ import annotations

import threading
from typing import Optional

from core.schemas import Budget


class BudgetManager:
    """Thread-safe budget tracker.  One instance per pipeline run."""

    def __init__(self, total: float):
        self._lock   = threading.Lock()
        self._budget = Budget(total=total)

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def budget(self) -> Budget:
        with self._lock:
            return self._budget

    @property
    def remaining(self) -> float:
        return self.budget.remaining or 0.0

    @property
    def spent(self) -> float:
        return self.budget.spent

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def check_stop(self, required: float = 1.0) -> bool:
        """Return True when budget is too low to run another experiment."""
        return self.remaining < required

    # ── Mutation ──────────────────────────────────────────────────────────────

    def allocate_slice(self, name: str, amount: float) -> Budget:
        """Reserve a named budget slice for a phase/agent."""
        with self._lock:
            self._budget = self._budget.allocate_slice(name, amount)
            return self._budget

    def debit(self, cost: float, description: str = "") -> Budget:
        """Subtract cost from remaining budget.  Raises if already exhausted."""
        with self._lock:
            if self._budget.remaining is not None and self._budget.remaining < cost:
                raise BudgetExhaustedError(
                    f"Requested {cost:.2f} but only {self._budget.remaining:.2f} remaining."
                    + (f" ({description})" if description else "")
                )
            self._budget = self._budget.debit(cost)
            return self._budget

    def try_debit(self, cost: float, description: str = "") -> Optional[Budget]:
        """Like debit() but returns None instead of raising when budget is low."""
        try:
            return self.debit(cost, description)
        except BudgetExhaustedError:
            return None

    def summary(self) -> str:
        b = self.budget
        pct = 100 * b.spent / b.total if b.total > 0 else 0
        return (
            f"Budget: {b.spent:.1f}/{b.total:.1f} spent ({pct:.0f}%), "
            f"{b.remaining:.1f} remaining"
        )


class BudgetExhaustedError(RuntimeError):
    pass
