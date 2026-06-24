"""
Solver toolkit — deterministic numerical routines.

Agents call these from code; results go to stores.  No LLM reasoning here.

Sections:
  1. Input generators     (PRBS, multisine, chirp, step sequences)
  2. ODE simulation       (simulate a white/grey-box model)
  3. Nonlinear least-squares identification
  4. Linear identification (ARX, subspace)
  5. Metrics & residual diagnostics
  6. Fisher Information Matrix (identifiability numerics)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import linalg, optimize, signal


# ── 1. Input generators ───────────────────────────────────────────────────────

def generate_prbs(
    n_samples:  int,
    dt:         float,
    amplitude:  float = 1.0,
    clock_div:  int   = 5,
    seed:       int   = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pseudo-Random Binary Sequence — broadband excitation for linear ID.

    Returns (t, u) where u alternates between ±amplitude.
    """
    rng  = np.random.default_rng(seed)
    n_switches = n_samples // clock_div + 1
    bits = rng.choice([-1, 1], size=n_switches)
    u    = np.repeat(bits, clock_div)[:n_samples].astype(float) * amplitude
    t    = np.arange(n_samples) * dt
    return t, u


def generate_multisine(
    n_samples:   int,
    dt:          float,
    frequencies: list[float],   # Hz
    amplitude:   float = 1.0,
    seed:        int   = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-sine with random phases — flat power spectrum at chosen frequencies.
    """
    rng    = np.random.default_rng(seed)
    t      = np.arange(n_samples) * dt
    phases = rng.uniform(0, 2 * np.pi, len(frequencies))
    u      = sum(
        np.sin(2 * np.pi * f * t + phi)
        for f, phi in zip(frequencies, phases)
    )
    u = u / np.max(np.abs(u)) * amplitude
    return t, u


def generate_chirp(
    n_samples: int,
    dt:        float,
    f0:        float = 0.1,
    f1:        float = 5.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear chirp sweeping from f0 to f1 Hz."""
    t = np.arange(n_samples) * dt
    u = amplitude * signal.chirp(t, f0, t[-1], f1, method="linear")
    return t, u


def generate_steps(
    levels:      list[float],
    hold_time:   float,
    dt:          float,
) -> tuple[np.ndarray, np.ndarray]:
    """Staircase of steps — useful for static curve fitting."""
    samples_per = max(1, int(hold_time / dt))
    u = np.concatenate([np.full(samples_per, lv) for lv in levels])
    t = np.arange(len(u)) * dt
    return t, u


# ── 2. ODE simulation ─────────────────────────────────────────────────────────

def simulate_ode(
    f:      Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t:      np.ndarray,
    u:      np.ndarray,
    x0:     np.ndarray,
    C:      Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a continuous-time ODE  ẋ = f(t, x, u(t))  at the given time points.

    Parameters
    ----------
    f   : (t, x, u) → ẋ  (state derivative)
    t   : (N,) time vector
    u   : (n_inputs, N) input array
    x0  : (n_states,) initial state
    C   : (n_outputs, n_states) output matrix; identity if None

    Returns (x, y) both shape (n_states or n_outputs, N)
    """
    from scipy.integrate import solve_ivp

    n_inputs = u.shape[0]

    def _interp_u(ti):
        idx = int(np.searchsorted(t, ti, side="right")) - 1
        idx = np.clip(idx, 0, u.shape[1] - 1)
        return u[:, idx]

    def _rhs(ti, xi):
        ui = _interp_u(ti)
        return f(ti, xi, ui)

    sol = solve_ivp(_rhs, (t[0], t[-1]), x0, t_eval=t, method="RK45",
                    rtol=1e-6, atol=1e-8)
    x = sol.y   # (n_states, N)

    if C is None:
        C = np.eye(x.shape[0])
    y = C @ x
    return x, y


# ── 2b. UKF Smoother ─────────────────────────────────────────────────────────

def ukf_smooth(
    f_step: Callable,
    params: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    y_obs: np.ndarray,
    n_states: int,
    output_index: int,
    Q: np.ndarray,
    R: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unscented Kalman Smoother: UKF forward pass + RTS backward pass.

    Estimates the full hidden state (e.g. [θ, θ̇]) at every time step from
    noisy scalar output measurements y_obs, using the compiled one-step
    dynamics f_step (from ``make_rk4_step``).

    The RTS backward pass refines every filtered estimate using future
    measurements, giving smoother and less biased velocity estimates than
    the Savitzky-Golay approach — eliminating the K_d bias that arises
    when SG-estimated velocities at segment boundaries are fed to the NLS.

    Parameters
    ----------
    f_step       : callable(params, x, u, dt) → x_next  — batched RK4 step
    params       : current physics parameter vector (fixed during smoothing)
    t, u         : time (N,) and input (N,) arrays
    y_obs        : scalar output measurements (N,)
    n_states     : state dimension (2 for a 2nd-order pendulum)
    output_index : which state component is observed (0 = position)
    Q            : (n_states, n_states) process noise covariance
    R            : scalar measurement noise variance (noise_std²)
    x0           : (n_states,) initial state; defaults to [y_obs[0], 0, ...]
    P0           : (n_states, n_states) initial covariance

    Returns
    -------
    (n_states, N) smoothed state array — row 0 is position, row 1 velocity
    """
    N = len(t)
    n = n_states

    # ── Van der Merwe sigma-point parameters (α=1, β=2, κ=0) ────────────────
    # λ = α²(n+κ)−n = 0  →  c = √n,  W_m[0]=0,  W_c[0]=2,  W_m/c[i≠0]=1/(2n)
    lam     = 0.0
    n_sig   = 2 * n + 1
    c_scale = np.sqrt(float(n) + lam)          # = √n
    Wm      = np.full(n_sig, 0.5 / (n + lam))  # 1/(2n) for all
    Wm[0]   = lam / (n + lam)                  # 0
    Wc      = Wm.copy()
    Wc[0]  += 1.0 - 1.0 + 2.0                 # += (1−α²+β) = 2

    # Measurement selector: H x = x[output_index]
    H    = np.zeros((1, n))
    H[0, output_index] = 1.0

    # Initial conditions
    if x0 is None:
        x0 = np.zeros(n)
        x0[output_index] = float(y_obs[0])
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    if P0 is None:
        P0 = np.eye(n)
        P0[output_index, output_index] = R
        for _i in range(n):
            if _i != output_index:
                P0[_i, _i] = 10.0
    else:
        P0 = np.asarray(P0, dtype=float).copy()

    # ── Storage for RTS backward pass ────────────────────────────────────────
    x_filt  = np.zeros((N, n))
    P_filt  = np.zeros((N, n, n))
    x_pred  = np.zeros((N, n))       # x_pred[k]  = predicted mean  arriving at k
    P_pred  = np.zeros((N, n, n))    # P_pred[k]  = predicted cov   arriving at k
    P_cross = np.zeros((N, n, n))    # P_cross[k] = cross-cov P(x_{k-1}, x_{k|k-1})

    # ── Initial measurement update (k = 0) ───────────────────────────────────
    Sy  = float(H @ P0 @ H.T) + R
    Kg  = (P0 @ H.T) / Sy
    x_c = x0 + Kg.ravel() * (float(y_obs[0]) - float(H @ x0))
    P_c = P0 - Kg * Sy * Kg.T
    x_filt[0] = x_c
    P_filt[0] = P_c

    # ── Forward pass ─────────────────────────────────────────────────────────
    for k in range(N - 1):
        dt_k = float(t[k + 1] - t[k])
        uk   = float(u[k])

        # Cholesky of (n+λ)·P  (add jitter for numerical safety)
        nP = (n + lam) * P_c
        try:
            Lk = np.linalg.cholesky(nP + 1e-10 * np.eye(n))
        except np.linalg.LinAlgError:
            Lk = np.linalg.cholesky(nP + 1e-6 * np.eye(n))

        # Sigma points: centre + 2n columns
        sigmas = np.empty((n, n_sig))
        sigmas[:, 0] = x_c
        for j in range(n):
            sigmas[:, j + 1]     = x_c + Lk[:, j]
            sigmas[:, j + 1 + n] = x_c - Lk[:, j]

        # Propagate all sigma points through RK4 in one vectorised call
        try:
            sp_prop = f_step(params, sigmas, uk, dt_k)
            bad = ~np.isfinite(sp_prop)
            if bad.any():
                sp_prop = np.where(bad, sigmas, sp_prop)
        except Exception:
            sp_prop = sigmas.copy()

        # Predicted mean, covariance, and cross-covariance
        x_pr = sp_prop @ Wm
        P_pr = Q.copy()
        P_cr = np.zeros((n, n))
        for j in range(n_sig):
            dp = sp_prop[:, j] - x_pr
            dc = sigmas[:, j] - x_c
            P_pr += Wc[j] * np.outer(dp, dp)
            P_cr += Wc[j] * np.outer(dc, dp)

        # Measurement update with y_obs[k+1]
        Sy_pr = float(H @ P_pr @ H.T) + R
        Kg_pr = (P_pr @ H.T) / Sy_pr
        x_up  = x_pr + Kg_pr.ravel() * (float(y_obs[k + 1]) - float(H @ x_pr))
        P_up  = P_pr - Kg_pr * Sy_pr * Kg_pr.T

        # Store for RTS
        x_pred [k + 1] = x_pr
        P_pred [k + 1] = P_pr
        P_cross[k + 1] = P_cr

        x_c = x_up
        P_c = P_up
        x_filt[k + 1] = x_c
        P_filt[k + 1] = P_c

    # ── RTS backward smoother pass ────────────────────────────────────────────
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(N - 2, -1, -1):
        Pp = P_pred[k + 1] + 1e-12 * np.eye(n)
        try:
            G = P_cross[k + 1] @ np.linalg.solve(Pp, np.eye(n))
        except np.linalg.LinAlgError:
            G = P_cross[k + 1] @ np.linalg.pinv(Pp)
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])
        dP = P_smooth[k + 1] - P_pred[k + 1]
        P_smooth[k] = P_filt[k] + G @ dP @ G.T

    return x_smooth.T   # (n_states, N)


# ── 3. Nonlinear least-squares identification ─────────────────────────────────

def nonlinear_least_squares(
    residual_fn:     Callable[[np.ndarray], np.ndarray],
    p0:              np.ndarray,
    bounds:          tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
    method:          str = "trf",
    max_nfev:        int = 500,
    ftol:            float = 1e-8,
    gtol:            float = 1e-8,
) -> dict:
    """
    Levenberg-Marquardt / TRF nonlinear least-squares.

    Returns dict with:
        params      — fitted parameter vector
        covariance  — approximate parameter covariance (from Jacobian)
        residuals   — final residual vector
        cost        — 0.5 * sum(residuals²)
        success     — bool
        message     — str
    """
    result = optimize.least_squares(
        residual_fn, p0, bounds=bounds, method=method,
        max_nfev=max_nfev, ftol=ftol, xtol=1e-8, gtol=gtol,
    )

    # Covariance from Jacobian: Σ ≈ σ² (JᵀJ)⁻¹
    try:
        J     = result.jac
        JtJ   = J.T @ J
        s2    = (2 * result.cost) / max(len(result.fun) - len(p0), 1)
        cov   = np.linalg.pinv(JtJ) * s2
    except Exception:
        cov = np.full((len(p0), len(p0)), np.nan)

    return {
        "params":     result.x,
        "covariance": cov,
        "residuals":  result.fun,
        "cost":       float(result.cost),
        "success":    result.success,
        "message":    result.message,
    }


# ── 4. Linear identification ──────────────────────────────────────────────────

def fit_arx(
    y: np.ndarray,
    u: np.ndarray,
    na: int = 2,
    nb: int = 2,
    nk: int = 1,
) -> dict:
    """
    ARX model identification: A(q)y = B(q)u + e
    Returns OLS parameter estimates and covariance.
    """
    N = len(y)
    max_lag = max(na, nb + nk - 1)
    rows = N - max_lag

    Phi = np.zeros((rows, na + nb))
    for i in range(rows):
        k = i + max_lag
        if na > 0:
            Phi[i, :na] = -y[np.arange(k - 1, k - na - 1, -1)]
        if nb > 0:
            Phi[i, na:] = u[np.arange(k - nk, k - nk - nb, -1)]

    Y = y[max_lag:]
    theta, res, rank, sv = np.linalg.lstsq(Phi, Y, rcond=None)
    s2 = (np.sum((Y - Phi @ theta)**2)) / max(rows - len(theta), 1)
    cov = s2 * np.linalg.pinv(Phi.T @ Phi)
    return {"params": theta, "covariance": cov, "na": na, "nb": nb, "nk": nk}


def fit_subspace_n4sid(
    y: np.ndarray,
    u: np.ndarray,
    n_order: int = 2,
    n_block: int = 10,
) -> dict:
    """
    Simplified N4SID-style subspace identification.
    Returns A, B, C, D state-space matrices for a discrete-time model.

    Note: This is a lightweight implementation for moderate-sized data.
    For production use, wrap system_identification libraries.
    """
    N = len(y)
    p, l = 1, 1   # scalar y, scalar u

    # Build block-Hankel matrices
    i_max = N - 2 * n_block
    if i_max < n_order + 1:
        raise ValueError("Not enough data for subspace ID with these parameters.")

    Y = np.array([y[j:j+n_block] for j in range(i_max)]).T        # (n_block, i_max)
    U = np.array([u[j:j+n_block] for j in range(i_max)]).T

    # Oblique projection via SVD
    W = np.vstack([U, Y])
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    S_trunc = S[:n_order]
    gamma_hat = np.diag(np.sqrt(S_trunc)) @ Vt[:n_order, :]

    # Least-squares system matrices from shifted Hankel
    X  = gamma_hat[:, :-1]
    X1 = gamma_hat[:, 1:]
    U_ = U[:, :-1]
    Y_ = Y[0:1, :-1]

    coeff, _, _, _ = np.linalg.lstsq(
        np.vstack([X, U_]).T, np.vstack([X1, Y_]).T, rcond=None,
    )
    A = coeff[:n_order, :n_order].T
    B = coeff[:n_order, n_order:].T
    C = coeff[n_order:, :n_order].T
    D = coeff[n_order:, n_order:].T

    return {"A": A, "B": B, "C": C, "D": D, "order": n_order}


# ── 5. Metrics & residual diagnostics ─────────────────────────────────────────

def compute_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_ref:     Optional[np.ndarray] = None,
) -> dict:
    """
    Compute standard identification metrics.

    Returns RMSE, normalised RMSE (against signal range or y_ref range),
    R², max absolute error.
    """
    err    = y_pred - y_true
    rmse   = float(np.sqrt(np.mean(err**2)))
    mae    = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2     = 1.0 - ss_res / max(ss_tot, 1e-12)

    ref     = y_ref if y_ref is not None else y_true
    y_range = float(np.ptp(ref)) or 1.0
    nrmse   = rmse / y_range

    return {
        "rmse":  rmse,
        "nrmse": nrmse,
        "mae":   mae,
        "r2":    float(r2),
    }


def residual_whiteness_test(residuals: np.ndarray, n_lags: int = 20) -> dict:
    """
    Ljung-Box whiteness test on residuals.

    Returns the test statistic, p-value, and a boolean indicating
    whether residuals are consistent with white noise (p > 0.05).
    """
    from scipy.stats import chi2

    N = len(residuals)
    r = residuals - np.mean(residuals)
    r0 = np.dot(r, r) / N

    if r0 < 1e-20:
        # Zero-variance residuals → perfectly correlated (or all-sentinel), not white.
        return {"Q_stat": float("inf"), "p_value": 0.0, "is_white": False}

    acf_sq_sum = 0.0
    for lag in range(1, min(n_lags + 1, N)):
        rk = np.dot(r[lag:], r[:-lag]) / N / r0
        acf_sq_sum += rk**2 / (N - lag)

    Q    = N * (N + 2) * acf_sq_sum
    dof  = n_lags
    pval = float(1.0 - chi2.cdf(Q, dof))
    return {"Q_stat": float(Q), "p_value": pval, "is_white": pval > 0.05}


def residual_correlation_with_input(
    residuals: np.ndarray,
    u:         np.ndarray,
    n_lags:    int = 20,
) -> dict:
    """
    Cross-correlation between residuals and input.
    Structured residuals (model missing dynamics) show non-zero cross-correlation.
    """
    r  = residuals - residuals.mean()
    ui = u - u.mean()
    N  = min(len(r), len(ui))
    r, ui = r[:N], ui[:N]

    xc = np.array([
        np.dot(r[lag:], ui[:N-lag]) / (N * np.std(r) * np.std(ui) + 1e-12)
        for lag in range(n_lags + 1)
    ])
    max_xc = float(np.max(np.abs(xc)))
    return {"cross_corr": xc.tolist(), "max_cross_corr": max_xc,
            "is_structured": max_xc > 0.15}


# ── 6. Fisher Information Matrix ──────────────────────────────────────────────

def compute_fim(
    sensitivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t:              np.ndarray,
    u:              np.ndarray,
    params:         np.ndarray,
    noise_var:      float = 1.0,
    dp:             float = 1e-4,
) -> dict:
    """
    Estimate the Fisher Information Matrix numerically using finite differences.

    sensitivity_fn(params, u) → y_pred of shape (N,)

    Returns FIM, its condition number, eigenvalues, and a rank assessment.
    """
    n_params = len(params)
    y0       = sensitivity_fn(params, u)
    N        = len(y0)

    S = np.zeros((N, n_params))
    for i in range(n_params):
        p_hi = params.copy(); p_hi[i] += dp
        p_lo = params.copy(); p_lo[i] -= dp
        S[:, i] = (sensitivity_fn(p_hi, u) - sensitivity_fn(p_lo, u)) / (2 * dp)

    FIM   = S.T @ S / noise_var
    evals = np.linalg.eigvalsh(FIM)
    cond  = float(evals[-1] / max(evals[0], 1e-14))
    rank  = int(np.sum(evals > 1e-10 * evals[-1]))

    return {
        "FIM":        FIM,
        "eigenvalues": evals.tolist(),
        "condition_number": cond,
        "rank":       rank,
        "full_rank":  rank == n_params,
    }
