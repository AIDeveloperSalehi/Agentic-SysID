"""Abstract base class for all physical plants (real and simulated)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class BasePlant(ABC):
    """
    Interface every plant must satisfy.

    apply_input receives a callable u(t) and a time vector, integrates the
    plant dynamics, adds measurement noise, and returns (t, y) where y has
    shape (n_outputs, n_samples).
    """

    @abstractmethod
    def apply_input(
        self,
        u_func: Callable[[float], np.ndarray],
        t_span: tuple[float, float],
        dt: float,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the plant.

        Parameters
        ----------
        u_func : callable(t) → ndarray(n_inputs,)
        t_span : (t_start, t_end)
        dt     : sample interval in seconds
        x0     : initial state; uses plant default if None

        Returns
        -------
        t  : (N,)          time vector
        u  : (n_inputs, N) applied inputs
        y  : (n_outputs, N) noisy outputs
        """

    @property
    @abstractmethod
    def n_inputs(self) -> int: ...

    @property
    @abstractmethod
    def n_outputs(self) -> int: ...

    @property
    @abstractmethod
    def n_states(self) -> int: ...

    @property
    @abstractmethod
    def default_x0(self) -> np.ndarray: ...
