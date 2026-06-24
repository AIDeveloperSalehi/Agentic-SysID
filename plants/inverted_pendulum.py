"""
Pendulum plant with viscous + Coulomb friction.

True ODE (hidden from the identification pipeline):
    J·θ̈ = τ(t) - b_v·θ̇ - f_c·tanh(θ̇/ε) - m·g·L·sin(θ)

  θ=0  →  straight down (stable equilibrium, gravity restores)
  Positive τ  →  counter-clockwise torque

The tanh approximation with small ε replaces sign() to keep the ODE
smooth for the integrator while still exhibiting strongly nonlinear
stick-slip behaviour near θ̇≈0 that a linear viscous-only model cannot fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from plants.base_plant import BasePlant


@dataclass
class PendulumParams:
    """Ground-truth parameters — kept private from the identification agents."""
    J:         float = 0.05    # kg·m²  moment of inertia
    m:         float = 0.5     # kg     bob mass
    L:         float = 0.30    # m      length to CoM
    b_v:       float = 0.02    # N·m·s  viscous friction coefficient
    f_c:       float = 0.08    # N·m    Coulomb friction magnitude
    g:         float = 9.81    # m/s²
    noise_std: float = 0.001   # rad    angle measurement noise
    coulomb_smooth_eps: float = 0.01  # rad/s  tanh smoothing width


class PendulumPlant(BasePlant):
    """
    Simulated pendulum driven by an external torque.

    Only angle θ is measured (angular velocity is not directly observed),
    which creates a joint state-parameter estimation challenge.
    """

    def __init__(self, params: Optional[PendulumParams] = None, seed: int = 42):
        self._p = params or PendulumParams()
        self._rng = np.random.default_rng(seed)

    # ── BasePlant interface ───────────────────────────────────────────────────

    @property
    def n_inputs(self) -> int:  return 1
    @property
    def n_outputs(self) -> int: return 1   # angle only
    @property
    def n_states(self) -> int:  return 2   # [θ, θ̇]
    @property
    def default_x0(self) -> np.ndarray:
        return np.array([0.3, 0.0])        # 0.3 rad offset, at rest

    def apply_input(
        self,
        u_func: Callable[[float], np.ndarray],
        t_span: tuple[float, float],
        dt: float,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x0 is None:
            x0 = self.default_x0

        t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)

        sol = solve_ivp(
            fun=lambda t, x: self._ode(t, x, u_func),
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        t  = sol.t
        u  = np.array([u_func(ti)[0] for ti in t]).reshape(1, -1)
        y  = sol.y[0:1, :] + self._rng.normal(0, self._p.noise_std, (1, len(t)))

        return t, u, y

    # ── Private ───────────────────────────────────────────────────────────────

    def _ode(
        self,
        t: float,
        x: np.ndarray,
        u_func: Callable[[float], np.ndarray],
    ) -> list[float]:
        p = self._p
        theta, theta_dot = x
        tau = float(u_func(t)[0])

        # Coulomb friction with tanh smoothing (avoids stiff discontinuity)
        friction = p.b_v * theta_dot + p.f_c * np.tanh(theta_dot / p.coulomb_smooth_eps)
        theta_ddot = (tau - friction - p.m * p.g * p.L * np.sin(theta)) / p.J

        return [theta_dot, theta_ddot]

    def simulate_noiseless(
        self,
        u_func: Callable[[float], np.ndarray],
        t_span: tuple[float, float],
        dt: float,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (t, u, theta, theta_dot) without measurement noise.
        Used internally for visualisation and ground-truth comparisons.
        """
        if x0 is None:
            x0 = self.default_x0
        t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
        sol = solve_ivp(
            fun=lambda t, x: self._ode(t, x, u_func),
            t_span=t_span, y0=x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
        )
        t  = sol.t
        u  = np.array([u_func(ti)[0] for ti in t])
        return t, u, sol.y[0], sol.y[1]   # t, u, theta, theta_dot

    # ── Convenience: linearized model around θ=0 ─────────────────────────────

    def linearize(self) -> dict:
        """
        First-order Taylor expansion around (θ, θ̇, τ) = (0, 0, 0).

        Returns continuous-time A, B, C, D matrices and the composite
        parameters that are identifiable from angle measurements alone.
        """
        p = self._p
        K_g = p.m * p.g * p.L / p.J   # gravitational stiffness
        tau_d = p.b_v / p.J            # damping time constant (viscous only)
        # Note: Coulomb friction linearises to b_v + f_c/eps at θ̇=0
        tau_d_eff = (p.b_v + p.f_c / p.coulomb_smooth_eps) / p.J

        A = np.array([[0.0,   1.0   ],
                      [-K_g, -tau_d]])
        B = np.array([[0.0],
                      [1.0 / p.J]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        return {
            "A": A, "B": B, "C": C, "D": D,
            "K_g": K_g, "tau_d": tau_d, "tau_d_eff": tau_d_eff,
            "1/J": 1.0 / p.J,
        }

    @property
    def params(self) -> PendulumParams:
        """Expose true parameters (used only by tests and visualisation)."""
        return self._p
