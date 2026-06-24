"""
Template plant — copy this file and implement your own system.

Quick start
-----------
1. Copy this file:
       cp plants/my_plant_template.py plants/my_system.py

2. Rename the class, fill in your true ODE (or hardware interface) in
   _ode() and apply_input(), and set the four metadata properties.

3. In your config YAML set:
       plant:
         class: "plants.my_system.MySystem"

4. Run the pipeline:
       python main.py --config configs/my_system.yaml

The pipeline ONLY calls apply_input().  The _ode() helper is called
internally by apply_input() for a simulated plant.  For a hardware plant
you can delete _ode() entirely and read from the real device instead.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from plants.base_plant import BasePlant


class MyPlant(BasePlant):
    """
    Replace this docstring with a one-line description of your system.

    The identification pipeline never sees the true parameters stored here.
    It only observes the noisy outputs returned by apply_input().
    """

    def __init__(self, noise_std: float = 0.01, seed: int = 42):
        """
        Store your plant's TRUE parameters here.

        These values are used inside _ode() / apply_input() to simulate
        (or interface with) the real system.  The pipeline agents never
        read these attributes — they are completely hidden from the LLM.

        Any keyword argument you add here can be set from the config:

            plant:
              class: "plants.my_system.MySystem"
              noise_std: 0.005      ← passed to __init__ as a kwarg
              some_param: 1.23      ← same

        The `seed` argument is always injected from the CLI (--seed).
        """
        self._rng = np.random.default_rng(seed)
        self._noise_std = noise_std

        # ── Replace with your true system parameters ───────────────────────
        # Example: DC motor
        #   States  : [current i (A), shaft speed ω (rad/s)]
        #   Input   : armature voltage V (V)
        #   Output  : shaft speed ω (rad/s)
        self._R  = 1.0     # Ω     armature resistance
        self._L  = 0.005   # H     armature inductance
        self._Kb = 0.05    # V·s/rad  back-EMF constant
        self._Kt = 0.05    # N·m/A   torque constant
        self._J  = 0.001   # kg·m²   rotor inertia
        self._b  = 0.001   # N·m·s   viscous friction

    # ── Metadata ──────────────────────────────────────────────────────────────
    # These four properties tell the pipeline the dimensions of your plant.
    # They do NOT expose parameters or dynamics — just sizes.

    @property
    def n_inputs(self) -> int:
        """Number of actuator/input channels."""
        return 1   # voltage, torque, flow rate, …

    @property
    def n_outputs(self) -> int:
        """Number of measured output channels."""
        return 1   # speed, angle, pressure, …

    @property
    def n_states(self) -> int:
        """Total number of ODE state variables (may be > n_outputs)."""
        return 2   # [current, speed] — one is unmeasured

    @property
    def default_x0(self) -> np.ndarray:
        """Default initial state used when x0 is not specified by an experiment."""
        return np.zeros(self.n_states)

    # ── Core interface ────────────────────────────────────────────────────────

    def apply_input(
        self,
        u_func: Callable[[float], np.ndarray],
        t_span: tuple[float, float],
        dt: float,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one experiment and return the measured result.

        Parameters
        ----------
        u_func : callable  t (float) → ndarray(n_inputs,)
            Input signal as a function of time.  The pipeline generates
            this for you (PRBS, multisine, chirp, …).
        t_span : (t_start, t_end)
            Experiment duration in seconds.
        dt : float
            Sampling interval in seconds (= 1 / sample_rate).
        x0 : ndarray(n_states,) or None
            Initial state.  If None, default_x0 is used.

        Returns
        -------
        t : ndarray (N,)
            Time vector, shape (N,).
        u : ndarray (n_inputs, N)
            Input applied at each sample, shape (n_inputs, N).
        y : ndarray (n_outputs, N)
            Noisy measured outputs, shape (n_outputs, N).

        Important
        ---------
        - y must have shape (n_outputs, N), even for a single-output system
          (use y[0:1, :] not y[0, :] when slicing the state).
        - Add measurement noise to y here, not in _ode().
        - Keep x0 updates consistent: the pipeline may set non-default x0
          to restart experiments from a specific operating point.
        """
        if x0 is None:
            x0 = self.default_x0

        t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)

        # ── Simulated plant (ODE) — KEEP THIS BLOCK ───────────────────────────
        sol = solve_ivp(
            fun=lambda t, x: self._ode(t, x, u_func),
            t_span=t_span,
            y0=list(x0),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        u = np.array([u_func(ti) for ti in t]).T          # (n_inputs, N)

        # Select which state(s) are measured — adjust the slice to match
        # the output_state_index you set in store_model (Modeler config).
        # Here: shaft speed = state index 1.
        y_clean = sol.y[1:2, :]                            # (n_outputs, N)
        y = y_clean + self._rng.normal(0, self._noise_std, y_clean.shape)

        return t, u, y

        # ── Real hardware alternative — replace the ODE block above ───────────
        # Delete everything from "sol = solve_ivp" to "return t, u, y" and
        # use something like:
        #
        # t = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
        # u_arr = np.zeros((self.n_inputs, len(t)))
        # y_arr = np.zeros((self.n_outputs, len(t)))
        # for k, tk in enumerate(t):
        #     uk = u_func(tk)
        #     send_to_hardware(uk)          # your hardware driver
        #     yk = read_from_hardware()     # your hardware driver
        #     u_arr[:, k] = uk
        #     y_arr[:, k] = yk
        # return t, u_arr, y_arr
        # ─────────────────────────────────────────────────────────────────────

    # ── True dynamics (simulated plant only) ─────────────────────────────────

    def _ode(
        self,
        t: float,
        x: np.ndarray,
        u_func: Callable[[float], np.ndarray],
    ) -> list[float]:
        """
        Your true plant ODE — hidden from the identification pipeline.

        Write dx/dt = f(x, u) here.  This is called by apply_input()
        via scipy.integrate.solve_ivp; the pipeline never calls it directly.

        Replace the DC motor equations below with your own system.
        Delete this method entirely if using a real hardware plant.

        Example: DC motor with armature dynamics
        ----------------------------------------
        States : x = [i (current), ω (shaft speed)]
        Input  : V (armature voltage)

        L·di/dt  = V − R·i − Kb·ω
        J·dω/dt  = Kt·i − b·ω
        """
        i, omega = x
        V = float(u_func(t)[0])

        di_dt     = (V - self._R * i - self._Kb * omega) / self._L
        domega_dt = (self._Kt * i - self._b * omega) / self._J

        return [di_dt, domega_dt]
