"""
Residual diagnosis and strategy selection for the grey-box loop.
All logic is deterministic — no LLM.

Two entry points:
  diagnose(t, theta, eps)            — legacy Coulomb/poly interface (kept for
                                       backward compatibility and unit tests).
  diagnose_general(states, u, eps,   — general interface using a feature library;
                   state_names,        works for any system order or dynamics.
                   input_names)        Replaces the hardcoded Coulomb check in the
                                       sub-orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.signal import savgol_filter


# ── Constants ─────────────────────────────────────────────────────────────────

COULOMB_CORR_THRESHOLD  = 0.30   # sign(velocity) correlation threshold
Z_SCORE_THRESHOLD       = 2.0    # K_c z-score threshold
MIN_HIGH_VEL_FRACTION   = 0.10   # min fraction of fast samples
VEL_THRESHOLD           = 0.05   # threshold for "high velocity"

SINDY_CORR_THRESHOLD    = 0.15   # min correlation to attempt sparse identification
GP_CORR_THRESHOLD       = 0.05   # min correlation to attempt GP correction


# ── Enums and data classes ────────────────────────────────────────────────────

class Strategy(str, Enum):
    COULOMB_TERM  = "coulomb_term"    # Coulomb friction: extend with K_c·tanh → WHITE_BOX
    POLY_FALLBACK = "poly_fallback"   # polynomial additive correction → GREY_BOX
    SINDY         = "sindy"           # general sparse symbolic correction → GREY_BOX
    GP_CORRECTION = "gp_correction"   # GP non-parametric correction → GREY_BOX


@dataclass
class DiagnosisResult:
    """Result from the legacy diagnose() method — kept for backward compatibility."""
    strategy:      Strategy
    coulomb_corr:  float
    K_c_estimate:  Optional[float]
    n_high_vel:    int
    rationale:     str


@dataclass
class Diagnosis:
    """Result from the general diagnose_general() method."""
    strategy:        Strategy
    top_features:    List[str]
    correlations:    List[float]
    max_correlation: float


# ── Main class ────────────────────────────────────────────────────────────────

class StrategySelector:
    """
    Diagnoses acceleration-domain residuals and selects the correction strategy.
    """

    # ── Legacy interface (kept for existing tests and backward compat) ────────

    def diagnose(
        self,
        t:        np.ndarray,    # (N,) time
        theta:    np.ndarray,    # (N,) measured position (used to estimate velocity)
        eps_ddot: np.ndarray,    # (N,) acceleration residuals
    ) -> DiagnosisResult:
        """
        Diagnose residuals using the Coulomb / polynomial detector.

        Estimates velocity numerically from theta, then checks sign(velocity)
        correlation.  Returns DiagnosisResult with COULOMB_TERM or POLY_FALLBACK.
        """
        N = len(t)

        if not np.any(np.isfinite(eps_ddot)):
            return DiagnosisResult(
                strategy=Strategy.POLY_FALLBACK,
                coulomb_corr=0.0,
                K_c_estimate=None,
                n_high_vel=0,
                rationale="all-NaN residuals — using polynomial fallback",
            )

        theta_dot    = _smooth_gradient(t, theta)
        coulomb_corr = _sign_correlation(eps_ddot, theta_dot)

        mask = np.abs(theta_dot) > VEL_THRESHOLD
        n_hv = int(mask.sum())

        k_c_est  = 0.0
        z_score  = 0.0
        coulomb_detected = coulomb_corr > COULOMB_CORR_THRESHOLD

        if n_hv >= max(5, int(MIN_HIGH_VEL_FRACTION * N)):
            eps_hv   = eps_ddot[mask]
            sgn_hv   = np.sign(theta_dot[mask])
            k_c_est  = float(-np.mean(eps_hv * sgn_hv))
            se_k_c   = float(np.std(eps_hv) / np.sqrt(n_hv) + 1e-12)
            z_score  = abs(k_c_est) / se_k_c
            if z_score > Z_SCORE_THRESHOLD and k_c_est > 0.1:
                coulomb_detected = True
            k_c_est = float(np.clip(k_c_est, 0.05, 20.0))
        else:
            k_c_est = 1.0
            n_hv    = 0

        if coulomb_detected:
            return DiagnosisResult(
                strategy=Strategy.COULOMB_TERM,
                coulomb_corr=coulomb_corr,
                K_c_estimate=k_c_est,
                n_high_vel=n_hv,
                rationale=(
                    f"Coulomb detected (corr={coulomb_corr:.3f}, z={z_score:.1f}); "
                    f"K_c_estimate={k_c_est:.3f} from {n_hv} high-vel samples"
                ),
            )
        else:
            return DiagnosisResult(
                strategy=Strategy.POLY_FALLBACK,
                coulomb_corr=coulomb_corr,
                K_c_estimate=None,
                n_high_vel=0,
                rationale=(
                    f"No clear Coulomb signal (corr={coulomb_corr:.3f}, z={z_score:.1f}); "
                    "using polynomial additive correction"
                ),
            )

    # ── General interface (used by the redesigned sub-orchestrator) ───────────

    def diagnose_general(
        self,
        states:      np.ndarray,   # (system_order, N)
        u:           np.ndarray,   # (N,)
        eps_ddot:    np.ndarray,   # (N,) acceleration residuals
        state_names: List[str],
        input_names: List[str],
    ) -> Diagnosis:
        """
        Diagnose residuals using a general feature library (SINDy-style).

        Builds candidate basis functions from state and input data, computes
        their correlation with the residual, and selects the correction strategy
        purely from the data — no system-specific assumptions:

          SINDY         — some feature dominates (max_corr >= threshold);
                          use sparse symbolic identification
          GP_CORRECTION — no dominant feature; use non-parametric correction
        """
        from tools.feature_library import FeatureLibrary

        valid = np.isfinite(eps_ddot)
        if valid.sum() < 10:
            return Diagnosis(
                strategy=Strategy.GP_CORRECTION,
                top_features=[], correlations=[], max_correlation=0.0,
            )

        eps_v    = eps_ddot[valid]
        states_v = states[:, valid]
        u_v      = u[valid]

        lib = FeatureLibrary()
        Theta, names = lib.build(states_v, u_v, state_names, input_names)
        corr_map = lib.feature_correlations(Theta, names, eps_v)

        sorted_feats = sorted(corr_map.items(), key=lambda x: -x[1])
        top5 = sorted_feats[:5]
        top_names = [f for f, _ in top5]
        top_corrs  = [c for _, c in top5]
        max_corr   = top_corrs[0] if top_corrs else 0.0

        strategy = Strategy.SINDY if max_corr >= SINDY_CORR_THRESHOLD else Strategy.GP_CORRECTION

        return Diagnosis(
            strategy=strategy,
            top_features=top_names,
            correlations=top_corrs,
            max_correlation=max_corr,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth_gradient(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay first derivative using the filter's built-in deriv parameter."""
    N = len(y)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    try:
        wl = max(5, min(21, (N // 10) * 2 + 1))
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, N if N % 2 == 1 else N - 1)
        return savgol_filter(y, window_length=wl, polyorder=min(3, wl - 1),
                             deriv=1, delta=dt)
    except Exception:
        return np.gradient(y, t)


def _sign_correlation(residuals: np.ndarray, velocity: np.ndarray) -> float:
    """
    |corr(residuals, sign(velocity))| — Coulomb friction signature.
    Values > 0.25–0.30 indicate a strong structured friction residual.
    """
    sign_td = np.sign(velocity)
    r = residuals - residuals.mean()
    s = sign_td  - sign_td.mean()
    denom = (np.std(r) * np.std(s)) + 1e-12
    return float(min(np.abs(np.mean(r * s)) / denom, 1.0))
