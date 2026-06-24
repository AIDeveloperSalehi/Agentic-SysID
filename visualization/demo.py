"""
Demo visualization: side-by-side pendulum animation with PRBS input.

Layout (3 rows):
  Row 0 (full width) : PRBS torque τ(t) input signal
  Row 1 (2 columns)  : Actual plant pendulum | Identified model pendulum
  Row 2 (full width) : θ(t) actual (blue) vs predicted (orange) overlaid

Usage:
  python visualization/demo.py                          # best validated model in registry
  python visualization/demo.py --model-id abc123        # specific model
  python visualization/demo.py --output demo.mp4        # save to file (requires conda ffmpeg)
  python visualization/demo.py --output demo.gif        # save as GIF (no ffmpeg needed)
  python visualization/demo.py --duration 10 --amplitude 0.5

If --output demo.mp4 fails with an ffmpeg error, install the conda-forge version:
  conda install -c conda-forge ffmpeg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Make sure project root is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plants.inverted_pendulum import PendulumPlant
from tools.model_registry import ModelRegistry
from tools.solver_toolkit import generate_prbs, generate_steps, generate_chirp


# ── Composite input generator ────────────────────────────────────────────────

# Segment type → background color for the input panel shading
_SEG_COLORS = {
    "step":          "#c77dff",   # purple
    "sine":          "#4cc9f0",   # blue
    "low_freq_sine": "#4cc9f0",
    "prbs":          "#7bed9f",   # green
    "chirp":         "#ffd166",   # yellow
}

DEFAULT_SEGMENTS = [
    {"type": "step",          "duration": 3.0, "label": "Step sequence"},
    {"type": "low_freq_sine", "duration": 3.0, "label": "Low-freq sine"},
    {"type": "prbs",          "duration": 4.0, "label": "PRBS"},
]


def generate_composite_input(
    segments: list[dict],
    dt: float,
    amplitude: float = 0.5,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Concatenate multiple input segments into one (t, u) pair.

    Each segment dict: {"type": str, "duration": float, "label": str}
    Amplitude is shared unless overridden per-segment with "amplitude".

    Returns
    -------
    t          : (N,) time vector starting at 0
    u          : (N,) concatenated input signal
    boundaries : list of dicts with t_start, t_end, label, color (for visualization)
    """
    t_chunks, u_chunks, boundaries = [], [], []
    t_offset = 0.0

    for i, seg in enumerate(segments):
        seg_type = seg["type"]
        dur      = float(seg["duration"])
        amp      = float(seg.get("amplitude", amplitude))
        label    = seg.get("label", seg_type)
        n        = max(1, int(round(dur / dt)))

        if seg_type in ("step",):
            n_levels  = 5
            levels    = list(np.linspace(-amp, amp, n_levels))
            hold_time = dur / n_levels
            _, u_seg  = generate_steps(levels, hold_time, dt)
            u_seg     = u_seg[:n]
            if len(u_seg) < n:
                u_seg = np.pad(u_seg, (0, n - len(u_seg)), constant_values=u_seg[-1])

        elif seg_type in ("sine", "low_freq_sine"):
            t_local  = np.arange(n) * dt
            freq     = 1.0 / dur          # one full cycle across the segment
            u_seg    = amp * np.sin(2 * np.pi * freq * t_local)

        elif seg_type == "prbs":
            _, u_seg = generate_prbs(n, dt, amplitude=amp, clock_div=5, seed=seed + i)

        elif seg_type == "chirp":
            f0 = seg.get("f0", 0.1)
            f1 = seg.get("f1", min(0.5 / dt * 0.3, 5.0))
            _, u_seg = generate_chirp(n, dt, f0=f0, f1=f1, amplitude=amp)

        else:
            raise ValueError(f"Unknown segment type '{seg_type}'. "
                             "Choose: step, sine, low_freq_sine, prbs, chirp")

        t_seg = np.arange(n) * dt + t_offset
        t_chunks.append(t_seg)
        u_chunks.append(u_seg)
        boundaries.append({
            "t_start": t_offset,
            "t_end":   t_offset + n * dt,
            "label":   label,
            "color":   _SEG_COLORS.get(seg_type, "#e0e0e0"),
        })
        t_offset += n * dt

    t = np.concatenate(t_chunks)
    u = np.concatenate(u_chunks)
    return t, u, boundaries


# ── Best-model selector ───────────────────────────────────────────────────────

def best_validated_model(registry: ModelRegistry) -> Optional[str]:
    """
    Return the model_id of the model with the lowest achieved_rmse across all
    stored validity regions.  This matches what ShipAgent does: it picks
    dossier.artifacts.best_model_id, which is the running minimum val_rmse.

    Falls back to registry.latest_model() if no validity regions exist.
    """
    validity_dir = registry._validity_dir
    best_id   = None
    best_rmse = float("inf")

    for path in validity_dir.glob("*.json"):
        try:
            region = registry.load_validity(path.stem)
            if region.achieved_rmse < best_rmse and region.model_id:
                best_rmse = region.achieved_rmse
                best_id   = region.model_id
        except Exception:
            continue

    if best_id is not None:
        return best_id

    latest = registry.latest_model()
    return latest.id if latest else None

matplotlib.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#0f3460",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#0f3460",
    "grid.alpha":        0.4,
    "lines.linewidth":   2,
    "font.size":         10,
})

_BLUE   = "#4cc9f0"   # actual plant
_ORANGE = "#f72585"   # identified model
_GREEN  = "#7bed9f"   # input signal
_GREY   = "#8d99ae"
_YELLOW = "#ffd166"
_PIVOT  = "#ffffff"


# ── Model simulator loader ────────────────────────────────────────────────────

def load_simulator(model_id: str, registry: ModelRegistry):
    """
    Load the shipped model and return (simulator_fn, fitted_params, metadata).

    simulator_fn(params, t, u, x0=None) -> theta_array

    Handles all model types: white-box NLS, grey-box corrected, GP-corrected,
    SINDy output-corrected, surrogate ODE, I/O surrogate.
    """
    import numpy as np
    from tools.symbolic_math import make_ode_simulator

    model  = registry.load_model(model_id)
    meta   = model.metadata
    rhs    = meta.get("normalized_rhs", "")
    params = meta.get("fit_params", [])
    state_v = meta.get("state_vars", [])
    input_v = meta.get("input_vars", [])
    system_order = meta.get("system_order", len(state_v) if state_v else 2)
    osi = meta.get("output_state_index", 0)

    if rhs == "SURROGATE":
        from agents.validation import _make_surrogate_simulator, _make_io_surrogate_simulator
        from agents.surrogate.model_class_selector import Paradigm
        predictor    = registry.load_object(meta.get("surrogate_object_id", ""))
        paradigm_str = meta.get("surrogate_paradigm", "ode")
        if paradigm_str == Paradigm.INPUT_OUTPUT.value:
            sim = _make_io_surrogate_simulator(predictor, osi)
        else:
            sim = _make_surrogate_simulator(predictor, n_states=system_order,
                                            output_state_index=osi)
        return sim, np.array([]), meta

    if rhs == "RESIDUAL_CORRECTED":
        from agents.validation import _make_residual_sequence_simulator
        phys_model = registry.load_model(meta.get("physics_model_id", ""))
        phys_meta  = phys_model.metadata
        phys_rhs   = phys_meta.get("normalized_rhs", "")
        phys_params = phys_meta.get("fit_params", [])
        phys_p     = np.array([phys_model.parameters.get(p, 1.0) for p in phys_params])
        phys_sim   = make_ode_simulator(
            phys_rhs, phys_params, state_v, input_v,
            highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
            output_state_index=osi,
        )
        residual_predictor = registry.load_object(meta.get("residual_object_id", ""))
        sim = _make_residual_sequence_simulator(phys_sim, phys_p, residual_predictor)
        return sim, np.array([]), meta

    if rhs == "GP_CORRECTED":
        correction_fn = None
        cid = meta.get("correction_object_id", "")
        if cid:
            try:
                correction_fn = registry.load_object(cid)
            except Exception:
                pass
        sim = make_ode_simulator(
            meta.get("base_rhs", ""), params, state_v, input_v,
            highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
            output_state_index=osi,
            correction_fn=correction_fn,
        )
        return sim, np.array([model.parameters.get(p, 1.0) for p in params]), meta

    if rhs == "SINDY_OUTPUT_CORRECTED":
        from tools.feature_library import FeatureLibrary
        from agents.estimator import _estimate_hidden_states
        phys_model = registry.load_model(meta.get("physics_model_id", ""))
        phys_meta  = phys_model.metadata
        phys_rhs   = phys_meta.get("normalized_rhs", "")
        phys_params = phys_meta.get("fit_params", [])
        phys_p     = np.array([phys_model.parameters.get(p, 1.0) for p in phys_params])
        phys_sim   = make_ode_simulator(
            phys_rhs, phys_params, state_v, input_v,
            highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
            output_state_index=osi,
        )
        correction_coeffs = meta.get("correction_coeffs", {})

        def _sindy_sim(param_values, t_arr, u_arr, x0=None):
            y_phys = phys_sim(phys_p, t_arr, u_arr, x0=x0)
            if not correction_coeffs or np.any(np.isnan(y_phys)):
                return y_phys
            so = meta.get("system_order", system_order)
            states_ph = _estimate_hidden_states(t_arr, y_phys, so)
            Theta, feat_names = FeatureLibrary().build(states_ph, u_arr, state_v, input_v)
            c = np.array([correction_coeffs.get(n, 0.0) for n in feat_names])
            return y_phys + Theta @ c

        return _sindy_sim, np.array([]), meta

    # Default: white-box or grey-box NLS ODE
    sim = make_ode_simulator(
        rhs, params, state_v, input_v,
        highest_deriv_var=state_v[-1] + "_ddot" if state_v else "x_ddot",
        output_state_index=osi,
    )
    fitted_p = np.array([model.parameters.get(p, 1.0) for p in params])
    return sim, fitted_p, meta


# ── Run simulation ─────────────────────────────────────────────────────────────

def run_demo_simulation(
    model_id: str,
    registry: ModelRegistry,
    segments: Optional[list[dict]] = None,
    amplitude: float = 0.5,
    dt: float = 0.02,
    seed: int = 7,
    reinit_at_segments: bool = True,
) -> dict:
    """
    Run plant and model on a composite multi-segment input.
    Returns all arrays needed for the animation.

    segments: list of segment dicts (see generate_composite_input).
              Defaults to DEFAULT_SEGMENTS (step → sine → PRBS).
    reinit_at_segments: if True (default), re-initialize the model from the
              plant's true state at the start of each segment so that open-loop
              integration error does not accumulate across the full horizon.
    """
    plant = PendulumPlant()
    x0    = plant.default_x0.copy()   # [theta0, theta_dot0]

    if segments is None:
        segments = DEFAULT_SEGMENTS

    t, u, boundaries = generate_composite_input(segments, dt, amplitude=amplitude, seed=seed)

    # ── Actual plant (noiseless for clean animation) ──────────────────────────
    u_func = lambda ti: np.array([float(u[int(np.clip(np.searchsorted(t, ti, side="right") - 1, 0, len(u) - 1))])])
    t_plant, u_plant, theta_actual, thetadot_actual = plant.simulate_noiseless(
        u_func=u_func,
        t_span=(float(t[0]), float(t[-1])),
        dt=dt,
        x0=x0,
    )

    # Trim to common length (solve_ivp may return slightly fewer points)
    N = min(len(t_plant), len(u_plant), len(theta_actual))
    t             = t_plant[:N]
    u             = u_plant[:N]
    theta_actual  = theta_actual[:N]
    thetadot_actual = thetadot_actual[:N]

    # ── Identified model ──────────────────────────────────────────────────────
    simulator, fitted_p, meta = load_simulator(model_id, registry)

    if reinit_at_segments and len(boundaries) > 1:
        # Simulate each segment independently, seeding from the plant's true
        # state at the segment boundary so errors don't compound across segments.
        theta_pred = np.full(N, np.nan)
        for seg in boundaries:
            i0 = int(np.searchsorted(t, seg["t_start"], side="left"))
            i1 = int(np.searchsorted(t, seg["t_end"],   side="right"))
            i1 = min(i1, N)
            if i0 >= N:
                break
            x0_seg  = np.array([theta_actual[i0], thetadot_actual[i0]])
            t_seg   = t[i0:i1]
            u_seg   = u[i0:i1]
            try:
                y_seg = simulator(fitted_p, t_seg, u_seg, x0=x0_seg)
            except Exception as exc:
                print(f"[warn] Segment '{seg['label']}' simulator failed ({exc})")
                y_seg = np.full(i1 - i0, np.nan)
            if y_seg is None:
                y_seg = np.full(i1 - i0, np.nan)
            n_fill = min(len(y_seg), i1 - i0)
            theta_pred[i0:i0 + n_fill] = y_seg[:n_fill]
    else:
        try:
            theta_pred = simulator(fitted_p, t, u, x0=x0)
        except Exception as exc:
            print(f"[warn] Simulator failed ({exc}), filling with NaN")
            theta_pred = np.full(N, np.nan)

    if theta_pred is None or len(theta_pred) < N:
        theta_pred_full = np.full(N, np.nan)
        if theta_pred is not None:
            theta_pred_full[:len(theta_pred)] = theta_pred
        theta_pred = theta_pred_full
    else:
        theta_pred = theta_pred[:N]

    model_label   = _model_label(meta)
    param_compare = build_param_comparison(registry, model_id)

    return {
        "t":               t,
        "u":               u,
        "theta_actual":    theta_actual,
        "theta_pred":      theta_pred,
        "model_label":     model_label,
        "L":               plant.params.L,
        "model_id":        model_id,
        "param_compare":   param_compare,
        "segments":        boundaries,
        "reinit_segments": reinit_at_segments,
    }


def _model_label(meta: dict) -> str:
    rhs = meta.get("normalized_rhs", "")
    mc  = meta.get("model_class", "")
    if rhs == "SURROGATE":
        paradigm = meta.get("surrogate_paradigm", "ode")
        return f"Black-box surrogate ({paradigm})"
    if rhs in ("RESIDUAL_CORRECTED", "GP_CORRECTED", "SINDY_OUTPUT_CORRECTED"):
        return f"Grey-box ({rhs.replace('_', ' ').title()})"
    if mc:
        return f"White-box ({mc})"
    if rhs:
        short = rhs[:60] + ("…" if len(rhs) > 60 else "")
        return f"White-box: {short}"
    return "Identified model"


# ── Parameter comparison ───────────────────────────────────────────────────────

# Maps identified parameter names → (formula string, lambda(PendulumParams) → true value)
_PENDULUM_TRUE_COMPOSITES = {
    "K_u": ("1/J",       lambda p: 1.0 / p.J),
    "K_d": ("b_v/J",     lambda p: p.b_v / p.J),
    "K_s": ("m·g·L/J",   lambda p: p.m * p.g * p.L / p.J),
    "K_c": ("f_c/J",     lambda p: p.f_c / p.J),
}


def build_param_comparison(registry: ModelRegistry, model_id: str) -> list[dict]:
    """
    Return one dict per fitted parameter with true vs identified composite values.

    Each dict:  {name, formula, true_val, id_val, pct_err}

    Parameters not in the known-composites table are included with true_val=None
    (no ground-truth available — e.g. surrogate coefficients).
    """
    from plants.inverted_pendulum import PendulumParams
    p = PendulumParams()   # ground-truth plant parameters

    model     = registry.load_model(model_id)
    fitted    = model.parameters
    fit_names = model.metadata.get("fit_params", [])

    rows = []
    for name in fit_names:
        id_val = float(fitted.get(name, float("nan")))
        if name in _PENDULUM_TRUE_COMPOSITES:
            formula, fn = _PENDULUM_TRUE_COMPOSITES[name]
            true_val    = fn(p)
            pct_err     = (id_val - true_val) / true_val * 100.0 if true_val != 0 else float("nan")
        else:
            formula  = "?"
            true_val = None
            pct_err  = float("nan")
        rows.append({
            "name":     name,
            "formula":  formula,
            "true_val": true_val,
            "id_val":   id_val,
            "pct_err":  pct_err,
        })
    return rows


# ── Animation ─────────────────────────────────────────────────────────────────

def animate_demo(
    data: dict,
    interval_ms: int = 40,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Build and return the FuncAnimation.

    Figure layout (GridSpec 4×2):
      [0, 0:2]  τ(t) PRBS input
      [1, 0  ]  Actual pendulum
      [1, 1  ]  Model pendulum
      [2, 0:2]  θ(t) actual vs predicted
      [3, 0:2]  Parameter comparison (static bar charts, one per parameter)
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    t             = data["t"]
    u             = data["u"]
    theta_act     = data["theta_actual"]
    theta_pred    = data["theta_pred"]
    L             = data["L"]
    model_label   = data["model_label"]
    param_compare = data.get("param_compare", [])
    seg_bounds    = data.get("segments", [])   # [{t_start, t_end, label, color}, ...]
    reinit_segs   = data.get("reinit_segments", False)
    N             = len(t)

    # Subsample to ~25 fps
    target_fps  = 1000 / interval_ms
    n_frames    = min(N, int((t[-1] - t[0]) * target_fps))
    frame_idx   = np.linspace(0, N - 1, n_frames, dtype=int)

    # ── Figure / axes ─────────────────────────────────────────────────────────
    has_params = bool(param_compare and any(r["true_val"] is not None for r in param_compare))
    height_ratios = [0.7, 2.2, 1.0, 1.1] if has_params else [0.7, 2.2, 1.0]
    n_rows        = 4 if has_params else 3
    fig_h         = 13 if has_params else 10

    fig = plt.figure(figsize=(14, fig_h), facecolor="#1a1a2e")
    gs  = GridSpec(
        n_rows, 2, figure=fig,
        height_ratios=height_ratios,
        hspace=0.40, wspace=0.30,
    )

    ax_input  = fig.add_subplot(gs[0, :])    # full-width top
    ax_pact   = fig.add_subplot(gs[1, 0])    # actual pendulum
    ax_pmod   = fig.add_subplot(gs[1, 1])    # model pendulum
    ax_theta  = fig.add_subplot(gs[2, :])    # full-width bottom

    for ax in (ax_input, ax_pact, ax_pmod, ax_theta):
        ax.set_facecolor("#16213e")
        ax.grid(True, alpha=0.25)

    # ── Input panel (static + running cursor) ─────────────────────────────────
    ax_input.set_xlim(t[0], t[-1])
    y_pad = max(abs(u.max()), abs(u.min())) * 0.15
    y_lo, y_hi = u.min() - y_pad, u.max() + y_pad
    ax_input.set_ylim(y_lo, y_hi)
    ax_input.set_ylabel("τ (N·m)", fontsize=9)
    ax_input.set_xlabel("Time (s)", fontsize=9)
    ax_input.axhline(0, color=_GREY, lw=0.8, alpha=0.4)

    # Segment shading and labels (drawn before signal so they sit in background)
    for seg in seg_bounds:
        col = seg["color"]
        ax_input.axvspan(seg["t_start"], seg["t_end"],
                         alpha=0.10, color=col, zorder=0)
        if seg["t_start"] > t[0]:
            ax_input.axvline(seg["t_start"], color=_GREY,
                             lw=0.9, ls="--", alpha=0.45, zorder=1)
        ax_input.text(
            (seg["t_start"] + seg["t_end"]) / 2,
            y_hi - y_pad * 0.25,
            seg["label"],
            ha="center", va="top", color=col,
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e",
                      alpha=0.65, edgecolor="none"),
        )

    input_title = "Input Torque" if not seg_bounds else "Input Torque — composite excitation"
    ax_input.set_title(input_title, color="#e0e0e0", fontsize=10, pad=4)
    ax_input.plot(t, u, color=_GREY, lw=1.2, alpha=0.30, zorder=2)
    (input_trace,)  = ax_input.plot([], [], color=_GREEN, lw=1.8, zorder=3)
    (input_cursor,) = ax_input.plot([], [], "|", color=_YELLOW, ms=14, mew=1.8, zorder=4)

    # Build a lookup: for each frame index, which segment color is active?
    def _seg_color_at(tk: float) -> str:
        for seg in seg_bounds:
            if seg["t_start"] <= tk < seg["t_end"]:
                return seg["color"]
        return _YELLOW

    # ── Pendulum panels ───────────────────────────────────────────────────────
    pad = L * 1.45

    def _setup_pend_ax(ax, color, label):
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad, pad)
        ax.set_aspect("equal")
        ax.set_title(label, color=color, fontsize=10, pad=4)
        ax.axhline(0, color=_GREY, lw=0.6, alpha=0.25)
        ax.axvline(0, color=_GREY, lw=0.6, alpha=0.25)
        ax.plot(0, 0, "o", color=_PIVOT, ms=5, zorder=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        trail_x = np.full(50, np.nan)
        trail_y = np.full(50, np.nan)
        (trail,)  = ax.plot([], [], "-", color=color, alpha=0.35, lw=1.5)
        (rod,)    = ax.plot([], [], "-", color=_YELLOW, lw=3.5, solid_capstyle="round")
        bob       = mpatches.Circle((0, -L), radius=L * 0.09, color=color, zorder=7)
        ax.add_patch(bob)
        time_lbl  = ax.text(
            0.03, 0.97, "", transform=ax.transAxes,
            color=_GREY, fontsize=8, va="top", ha="left",
        )
        theta_lbl = ax.text(
            0.97, 0.97, "", transform=ax.transAxes,
            color=color, fontsize=8, va="top", ha="right",
        )
        return rod, bob, trail, trail_x, trail_y, time_lbl, theta_lbl

    (rod_act, bob_act, trail_act, tx_act, ty_act,
     tlbl_act, thlbl_act) = _setup_pend_ax(ax_pact, _BLUE,   "Actual Plant")
    (rod_mod, bob_mod, trail_mod, tx_mod, ty_mod,
     tlbl_mod, thlbl_mod) = _setup_pend_ax(ax_pmod, _ORANGE, model_label)

    # ── θ(t) comparison panel — also shade segments ───────────────────────────
    for seg in seg_bounds:
        ax_theta.axvspan(seg["t_start"], seg["t_end"],
                         alpha=0.06, color=seg["color"], zorder=0)
        if seg["t_start"] > t[0]:
            ax_theta.axvline(seg["t_start"], color=_GREY,
                             lw=0.7, ls="--", alpha=0.35, zorder=1)

    ax_theta.plot(t, np.degrees(theta_act),  color=_BLUE,   lw=1.2, alpha=0.25)
    ax_theta.plot(t, np.degrees(theta_pred), color=_ORANGE, lw=1.2, alpha=0.25, ls="--")
    pred_label = "Predicted θ (re-init per segment)" if reinit_segs else "Predicted θ"
    (line_act,)  = ax_theta.plot([], [], color=_BLUE,   lw=2.0, label="Actual θ")
    (line_pred,) = ax_theta.plot([], [], color=_ORANGE, lw=1.8, ls="--", label=pred_label)

    # Mark re-initialization points with a dotted vertical line
    if reinit_segs:
        for seg in seg_bounds[1:]:   # skip first — no re-init at t=0
            ax_theta.axvline(seg["t_start"], color=_ORANGE,
                             lw=1.2, ls=":", alpha=0.55, zorder=2)
            ax_theta.text(
                seg["t_start"], 0.97, " ↺",
                transform=ax_theta.get_xaxis_transform(),
                color=_ORANGE, fontsize=9, va="top", alpha=0.80,
            )
    ax_theta.set_xlim(t[0], t[-1])
    all_theta = np.concatenate([
        np.degrees(theta_act),
        np.degrees(theta_pred[np.isfinite(theta_pred)]) if np.any(np.isfinite(theta_pred)) else np.array([0.0]),
    ])
    th_pad = max(5.0, (all_theta.max() - all_theta.min()) * 0.12)
    ax_theta.set_ylim(all_theta.min() - th_pad, all_theta.max() + th_pad)
    ax_theta.set_ylabel("θ (deg)", fontsize=9)
    ax_theta.set_xlabel("Time (s)", fontsize=9)
    ax_theta.legend(fontsize=8, loc="upper right", framealpha=0.3)

    # RMSE annotation (static, computed once)
    valid = np.isfinite(theta_pred)
    if valid.sum() > 10:
        rmse = float(np.sqrt(np.mean((theta_act[valid] - theta_pred[valid]) ** 2)))
        ax_theta.text(
            0.02, 0.97,
            f"RMSE = {np.degrees(rmse):.2f}°  ({rmse*1000:.2f} mrad)",
            transform=ax_theta.transAxes,
            color=_GREY, fontsize=8, va="top",
        )

    # ── Parameter comparison panel (static) ──────────────────────────────────
    if has_params:
        _render_param_panel(fig, gs, param_compare)

    seg_types = " → ".join(s.get("label", s.get("type", "")) for s in seg_bounds) if seg_bounds else "PRBS"
    fig.suptitle(
        f"System Identification Demo  |  {seg_types}",
        color="#e0e0e0", fontsize=12, y=0.995, fontweight="bold",
    )

    # ── Update function ───────────────────────────────────────────────────────
    trail_len = 50

    def _update(frame):
        k  = frame_idx[frame]
        tk = t[k]

        # Input trace + cursor (cursor color follows active segment)
        input_trace.set_data(t[:k+1], u[:k+1])
        input_cursor.set_data([tk], [u[k]])
        input_cursor.set_color(_seg_color_at(tk))

        # θ traces
        line_act.set_data(t[:k+1],  np.degrees(theta_act[:k+1]))
        line_pred.set_data(t[:k+1], np.degrees(theta_pred[:k+1]))

        # Helper: update one pendulum panel
        def _update_pend(theta, rod, bob, trail, tx, ty, tlbl, thlbl, color):
            bx = L * np.sin(theta[k])
            by = -L * np.cos(theta[k])
            rod.set_data([0, bx], [0, by])
            bob.center = (bx, by)

            i0 = max(0, k - trail_len + 1)
            xs = L * np.sin(theta[i0:k+1])
            ys = -L * np.cos(theta[i0:k+1])
            trail.set_data(xs, ys)

            tlbl.set_text(f"t = {tk:.2f} s")
            thlbl.set_text(f"θ = {np.degrees(theta[k]):+.1f}°")
            return rod, bob, trail

        _update_pend(theta_act,  rod_act, bob_act, trail_act, tx_act, ty_act,
                     tlbl_act, thlbl_act, _BLUE)
        _update_pend(theta_pred, rod_mod, bob_mod, trail_mod, tx_mod, ty_mod,
                     tlbl_mod, thlbl_mod, _ORANGE)

        return (input_trace, input_cursor, line_act, line_pred,
                rod_act, bob_act, trail_act,
                rod_mod, bob_mod, trail_mod,
                tlbl_act, thlbl_act, tlbl_mod, thlbl_mod)

    anim = FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=True,
    )

    if save_path:
        _save_animation(anim, save_path, fps=1000 // interval_ms)

    return anim


def _render_param_panel(fig, gs, param_compare: list[dict]) -> None:
    """
    Render a static parameter comparison row: one mini bar chart per parameter.

    Each mini chart shows two vertical bars (True=blue, Identified=orange) on
    the same absolute scale, with values and % error annotated.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    known = [r for r in param_compare if r["true_val"] is not None]
    n     = len(known)
    if n == 0:
        ax = fig.add_subplot(gs[3, :])
        ax.set_facecolor("#16213e")
        ax.axis("off")
        ax.text(0.5, 0.5, "Black-box model — no interpretable parameters",
                ha="center", va="center", color=_GREY, fontsize=11,
                transform=ax.transAxes)
        return

    gs_p  = GridSpecFromSubplotSpec(1, n, subplot_spec=gs[3, :], wspace=0.45)

    # Error thresholds for annotation colour
    def _err_color(pct):
        a = abs(pct)
        if a < 5:   return _GREEN
        if a < 15:  return _YELLOW
        return _ORANGE

    bar_w = 0.35

    for i, row in enumerate(known):
        ax = fig.add_subplot(gs_p[0, i])
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")

        true_v = row["true_val"]
        id_v   = row["id_val"]
        pct    = row["pct_err"]
        ec     = _err_color(pct)

        # Bars: x=0 → True, x=1 → Identified
        ax.bar(0, true_v, width=bar_w, color=_BLUE,   alpha=0.85, label="True")
        ax.bar(1, id_v,   width=bar_w, color=_ORANGE, alpha=0.85, label="Identified")

        # Value annotations above each bar
        y_top = max(true_v, id_v)
        y_pad = abs(y_top) * 0.12 + 0.05
        ax.text(0, true_v + y_pad * 0.4, f"{true_v:.3f}",
                ha="center", va="bottom", color=_BLUE,   fontsize=8, fontweight="bold")
        ax.text(1, id_v   + y_pad * 0.4, f"{id_v:.3f}",
                ha="center", va="bottom", color=_ORANGE, fontsize=8, fontweight="bold")

        # % error badge centred between bars
        sign  = "+" if pct >= 0 else ""
        badge = f"{sign}{pct:.1f}%"
        ax.text(0.5, true_v * 0.55, badge,
                ha="center", va="center", color=ec,
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a2e", alpha=0.7,
                          edgecolor=ec, linewidth=1.2))

        # Axis cosmetics
        ax.set_xlim(-0.55, 1.55)
        y_max = max(true_v, id_v)
        ax.set_ylim(0, y_max + y_pad * 1.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["True", "ID"], fontsize=8)
        ax.set_ylabel("Value", fontsize=7, color=_GREY)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_title(
            f"{row['name']}  =  {row['formula']}",
            color="#e0e0e0", fontsize=9, pad=4,
        )
        ax.grid(axis="y", alpha=0.2)

        # Legend only on first panel
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.25,
                      handlelength=1.0, handletextpad=0.4)

    # Row label on the left margin
    fig.text(
        0.01, gs[3, :].get_position(fig).y0 + gs[3, :].get_position(fig).height / 2,
        "Parameters",
        va="center", ha="left", color=_GREY, fontsize=9, rotation=90,
    )


def _save_animation(anim: FuncAnimation, save_path: str, fps: int = 25) -> None:
    """
    Save animation to file.  MP4 uses ffmpeg; falls back to GIF on failure.
    GIF uses pillow directly (no system ffmpeg needed).
    """
    import subprocess

    sp = save_path.lower()
    if sp.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved to {save_path}")
        return

    # MP4 path — try ffmpeg, fall back to GIF if it fails
    try:
        anim.save(save_path, writer="ffmpeg", fps=fps, dpi=130)
        print(f"Saved to {save_path}")
    except (subprocess.CalledProcessError, BrokenPipeError, OSError) as exc:
        gif_path = save_path.rsplit(".", 1)[0] + ".gif"
        print(
            f"\n[warn] ffmpeg failed ({type(exc).__name__}).\n"
            f"       The Homebrew ffmpeg has a broken OpenGL dependency on macOS 15.\n"
            f"       Fix: conda install -c conda-forge ffmpeg\n"
            f"       Falling back to GIF: {gif_path}\n"
        )
        anim.save(gif_path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved to {gif_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pendulum SysID demo animation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-id",  default=None,
                        help="Model ID in registry (default: best validated model)")
    parser.add_argument("--output",    default=None,
                        help="Save path (.mp4 or .gif). Omit to display interactively.")
    parser.add_argument("--amplitude", type=float, default=0.5,
                        help="Torque amplitude in N·m applied to all segments (default: 0.5)")
    parser.add_argument("--dt",        type=float, default=0.02,
                        help="Sample time in seconds (default: 0.02)")
    parser.add_argument("--seed",      type=int,   default=7,
                        help="Random seed for stochastic segments (default: 7)")
    parser.add_argument("--segments",  default=None,
                        help=(
                            "Comma-separated segment specs: type:duration  (default: step:3,sine:3,prbs:4)\n"
                            "Types: step, sine, prbs, chirp\n"
                            "Example: --segments step:2,sine:4,prbs:4"
                        ))
    parser.add_argument("--data-dir",  default="data/models",
                        help="Model registry directory (default: data/models)")
    parser.add_argument("--no-reinit", action="store_true",
                        help="Disable per-segment re-initialization (run model open-loop over full horizon)")
    args = parser.parse_args()

    registry = ModelRegistry(base_dir=args.data_dir)

    if args.model_id:
        model_id = args.model_id
    else:
        model_id = best_validated_model(registry)
        if model_id is None:
            print("No models found in registry. Run the pipeline first (python main.py).")
            sys.exit(1)
        print(f"Using best validated model: {model_id}")

    # Parse segments
    if args.segments:
        segments = []
        _labels = {"step": "Step sequence", "sine": "Low-freq sine",
                   "low_freq_sine": "Low-freq sine", "prbs": "PRBS", "chirp": "Chirp"}
        for spec in args.segments.split(","):
            spec = spec.strip()
            if ":" not in spec:
                parser.error(f"Bad segment spec '{spec}'. Use type:duration, e.g. prbs:4")
            stype, sdur = spec.split(":", 1)
            segments.append({
                "type":     stype.strip(),
                "duration": float(sdur.strip()),
                "label":    _labels.get(stype.strip(), stype.strip()),
            })
    else:
        segments = DEFAULT_SEGMENTS

    seg_desc = " + ".join(f"{s['label']} ({s['duration']}s)" for s in segments)
    total    = sum(s["duration"] for s in segments)
    print(f"Simulating {total:.1f}s  [{seg_desc}]  amplitude={args.amplitude} N·m ...")

    data = run_demo_simulation(
        model_id=model_id,
        registry=registry,
        segments=segments,
        amplitude=args.amplitude,
        dt=args.dt,
        seed=args.seed,
        reinit_at_segments=not args.no_reinit,
    )
    print(f"Model: {data['model_label']}")

    valid = np.isfinite(data["theta_pred"])
    if valid.sum() > 10:
        rmse = float(np.sqrt(np.mean((data["theta_actual"][valid] - data["theta_pred"][valid]) ** 2)))
        print(f"Demo RMSE: {np.degrees(rmse):.3f}°  ({rmse*1000:.3f} mrad)")

    anim = animate_demo(data, save_path=args.output)

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
