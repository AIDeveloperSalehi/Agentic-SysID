"""
Experiment-design toolkit — creates input sequences for two purposes:

  Informative excitation  — maximize parameter information for identification
  Adversarial probing     — find worst-case divergence for validation

Agents call these; this module does not touch the plant.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.schemas import PlantContract
from tools.solver_toolkit import (
    generate_chirp,
    generate_multisine,
    generate_prbs,
    generate_steps,
)


# ── Identification inputs ─────────────────────────────────────────────────────

def design_identification_input(
    contract: PlantContract,
    n_samples: int = 500,
    method: str = "prbs",
    frequency_range: Optional[Tuple[float, float]] = None,
    amplitude_fraction: float = 0.70,
    seed: int = 0,
) -> dict:
    """
    Create an informative input sequence for system identification.

    Parameters
    ----------
    contract          : PlantContract (provides limits and sample time)
    n_samples         : number of samples
    method            : "prbs" | "multisine" | "chirp" | "steps"
    frequency_range   : (f_lo, f_hi) Hz; defaults to [0.1, Nyquist/2]
    amplitude_fraction: fraction of the contract amplitude limit to use
    seed              : random seed

    Returns
    -------
    dict with keys: t, u (1-D), method, description, input_type
    """
    dt = contract.sample_time
    input_name = contract.input_names[0]
    lo, hi = contract.input_limits.get(input_name, (-1.0, 1.0))
    # Symmetric amplitude centred at zero
    half_range = min(abs(lo), abs(hi))
    amplitude = half_range * amplitude_fraction

    f_lo = (frequency_range or (0.1, 1.0 / (4.0 * dt)))[0]
    f_hi = (frequency_range or (0.1, 1.0 / (4.0 * dt)))[1]
    f_hi = min(f_hi, 0.5 / dt)   # Nyquist cap

    if method == "prbs":
        t, u = generate_prbs(n_samples, dt, amplitude=amplitude, seed=seed)
        desc = f"PRBS amplitude={amplitude:.3f}, N={n_samples}"
        itype = "prbs"

    elif method == "multisine":
        n_freqs = min(8, max(3, int(np.log2(n_samples))))
        freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freqs).tolist()
        t, u = generate_multisine(n_samples, dt, frequencies=freqs,
                                   amplitude=amplitude, seed=seed)
        desc = f"Multisine {n_freqs} freqs [{f_lo:.2f}–{f_hi:.2f}] Hz amplitude={amplitude:.3f}"
        itype = "multisine"

    elif method == "chirp":
        t, u = generate_chirp(n_samples, dt, f0=f_lo, f1=f_hi, amplitude=amplitude)
        desc = f"Chirp {f_lo:.2f}–{f_hi:.2f} Hz amplitude={amplitude:.3f}"
        itype = "chirp"

    elif method == "steps":
        n_levels = 7
        levels = np.linspace(-amplitude, amplitude, n_levels).tolist()
        hold_time = n_samples * dt / n_levels
        t, u = generate_steps(levels, hold_time, dt)
        n_samples = len(t)
        desc = f"Staircase {n_levels} levels ±{amplitude:.3f}"
        itype = "steps"

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: prbs, multisine, chirp, steps")

    u = np.clip(u, lo, hi)
    return {"t": t, "u": u, "method": method, "description": desc, "input_type": itype}


def design_compound_identification_input(
    contract: PlantContract,
    n_samples: int = 800,
    seed: int = 0,
) -> dict:
    """
    Broadband compound excitation: PRBS + multisine concatenated.
    Covers both low and high frequencies in one experiment.
    """
    half = n_samples // 2
    r1 = design_identification_input(contract, half, "prbs",      seed=seed)
    r2 = design_identification_input(contract, half, "multisine", seed=seed + 1)

    dt = contract.sample_time
    t = np.arange(n_samples) * dt
    u = np.concatenate([r1["u"], r2["u"]])[:n_samples]
    return {
        "t": t, "u": u,
        "method": "compound",
        "description": f"Compound PRBS+multisine N={n_samples}",
        "input_type": "compound",
    }


# ── Adversarial / validation inputs ──────────────────────────────────────────

def design_adversarial_inputs(
    contract: PlantContract,
    n_samples_per_scenario: int = 300,
    n_scenarios: int = 3,
    seed: int = 42,
) -> List[dict]:
    """
    Design adversarial validation inputs that stress-test model assumptions.

    All scenario parameters are derived from the PlantContract (sample_time,
    input_limits) and the scenario window length — no plant-specific constants.

    Scenarios:
      1. low_freq_sine    — exactly one cycle per window; exercises slow zero-crossings
                            and static nonlinearities (Coulomb, stiction, dead-zone)
      2. near_saturation  — near-limit PRBS; tests large-signal / nonlinear regime
      3. broadband_chirp  — sweeps from below f_slow to 35% of Nyquist;
                            frequency-domain coverage across the observable bandwidth

    Returns
    -------
    list of dicts {t, u, description, scenario_type}, length = n_scenarios
    """
    dt = contract.sample_time
    input_name = contract.input_names[0]
    lo, hi = contract.input_limits.get(input_name, (-1.0, 1.0))
    half_range = min(abs(lo), abs(hi))

    N        = n_samples_per_scenario
    T_total  = N * dt
    nyquist  = 0.5 / dt

    # One full cycle per scenario window — "slowest meaningful" test frequency
    f_slow     = max(1.0 / T_total, 0.01)
    # Chirp ceiling: well within observable bandwidth, capped at 10 Hz
    f_sweep_hi = min(nyquist * 0.35, 10.0)
    f_chirp_lo = max(f_slow * 0.5, 0.01)

    scenarios: List[dict] = []

    # 1. Low-frequency sine: one full cycle per window.
    #    Near-zero crossings stress Coulomb/stiction and any slow nonlinearity.
    t1 = np.arange(N) * dt
    u1 = half_range * 0.35 * np.sin(2 * np.pi * f_slow * t1)
    scenarios.append({
        "t": t1,
        "u": np.clip(u1, lo, hi),
        "description": (
            f"Low-frequency sine {f_slow:.3f} Hz — slow zero-crossings "
            "and static nonlinearities"
        ),
        "scenario_type": "low_freq_sine",
    })

    # 2. Near-saturation PRBS: 90% of input range.
    #    Pushes into large-signal nonlinear regime unseen during nominal training.
    t2, u2 = generate_prbs(N, dt, amplitude=half_range * 0.90, seed=seed)
    scenarios.append({
        "t": t2,
        "u": np.clip(u2, lo, hi),
        "description": "Near-saturation PRBS (90% range) — large-signal nonlinear regime",
        "scenario_type": "near_saturation",
    })

    # 3. Broadband chirp: f_chirp_lo → f_sweep_hi.
    #    Reveals frequency-domain model errors across the observable bandwidth.
    t3, u3 = generate_chirp(N, dt, f0=f_chirp_lo, f1=f_sweep_hi, amplitude=half_range * 0.55)
    scenarios.append({
        "t": t3,
        "u": np.clip(u3, lo, hi),
        "description": (
            f"Broadband chirp {f_chirp_lo:.3f}–{f_sweep_hi:.2f} Hz "
            "— frequency-domain validation"
        ),
        "scenario_type": "broadband_chirp",
    })

    return scenarios[:n_scenarios]


# ── Utility ───────────────────────────────────────────────────────────────────

def compute_information_content(
    t: np.ndarray,
    u: np.ndarray,
    sensitivity_fn,
    noise_var: float = 1e-6,
) -> dict:
    """
    Estimate parameter information content of an input via D-optimal criterion.

    sensitivity_fn(u) → S of shape (N, n_params)
    Returns log-det(FIM) and per-parameter marginal info (diagonal of FIM).
    """
    S = sensitivity_fn(u)
    FIM = S.T @ S / max(noise_var, 1e-14)
    sign, logdet = np.linalg.slogdet(FIM)
    diag = np.diag(FIM).tolist()
    max_diag = float(np.max(np.abs(diag))) if diag else 1.0
    thresh = 1e-10 * max_diag
    return {
        "logdet_fim": float(logdet) if sign > 0 else -1e30,
        "fim_diagonal": diag,
        "rank": int(np.sum(np.linalg.eigvalsh(FIM) > thresh)),
    }
