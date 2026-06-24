"""
Pendulum visualisation suite.

Four entry points:
  animate_pendulum()    — animated swing with torque indicator and angle trace
  plot_phase_portrait() — θ vs θ̇ with coloured trajectory
  plot_model_comparison() — true plant vs one or more model predictions
  plot_residual_map()   — 2-D heatmap of model error in (θ, θ̇) space
  plot_control_response() — closed-loop response under a designed controller
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#0f3460",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#0f3460",
    "grid.alpha":        0.5,
    "lines.linewidth":   2,
})

_BLUE   = "#4cc9f0"
_ORANGE = "#f72585"
_GREEN  = "#7bed9f"
_YELLOW = "#ffd166"
_GREY   = "#8d99ae"


# ── 1. Animated pendulum ──────────────────────────────────────────────────────

def animate_pendulum(
    t:       np.ndarray,
    theta:   np.ndarray,
    u:       np.ndarray,
    L:       float = 0.30,
    title:   str   = "Pendulum",
    interval_ms: int = 40,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Animate the pendulum swinging.

    Parameters
    ----------
    t       : (N,) time vector
    theta   : (N,) angle in radians (0 = hanging down)
    u       : (N,) applied torque
    L       : rod length for display
    save_path: if given, save to file (MP4 or GIF)
    """
    fig = plt.figure(figsize=(12, 6), facecolor="#1a1a2e")
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1])
    ax_pend  = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_torq  = fig.add_subplot(gs[1, 1])

    for ax in (ax_pend, ax_theta, ax_torq):
        ax.set_facecolor("#16213e")
        ax.grid(True, alpha=0.3)

    ax_pend.set_xlim(-L * 1.4, L * 1.4)
    ax_pend.set_ylim(-L * 1.4, L * 1.4)
    ax_pend.set_aspect("equal")
    ax_pend.set_title("Pendulum", color="#e0e0e0", pad=8)
    ax_pend.axhline(0, color=_GREY, alpha=0.3, lw=1)
    ax_pend.axvline(0, color=_GREY, alpha=0.3, lw=1)

    # Pivot marker
    ax_pend.plot(0, 0, "o", color=_GREY, ms=6, zorder=5)

    trail_len = min(60, len(t))
    trail_x = np.full(trail_len, np.nan)
    trail_y = np.full(trail_len, np.nan)

    (trail_line,) = ax_pend.plot([], [], "-", color=_BLUE, alpha=0.4, lw=1.5)
    (rod_line,)   = ax_pend.plot([], [], "-", color=_YELLOW, lw=3)
    bob_patch     = mpatches.Circle((0, 0), radius=0.02, color=_ORANGE, zorder=6)
    ax_pend.add_patch(bob_patch)
    torq_arrow    = ax_pend.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=_GREEN, lw=2),
    )
    time_text = ax_pend.text(
        0.02, 0.96, "", transform=ax_pend.transAxes,
        color=_GREY, fontsize=9, va="top",
    )

    # Static traces (full data in background)
    ax_theta.plot(t, np.degrees(theta), color=_GREY, alpha=0.25, lw=1)
    ax_torq.plot(t,  u,                 color=_GREY, alpha=0.25, lw=1)
    (theta_trace,) = ax_theta.plot([], [], color=_BLUE,   lw=1.5)
    (torq_trace,)  = ax_torq.plot([],  [], color=_GREEN,  lw=1.5)
    ax_theta.set_ylabel("θ (deg)", fontsize=8)
    ax_torq.set_ylabel("τ (N·m)", fontsize=8)
    ax_torq.set_xlabel("Time (s)", fontsize=8)
    ax_theta.set_xlim(t[0], t[-1])
    ax_torq.set_xlim(t[0], t[-1])
    ax_theta.set_ylim(np.degrees(theta).min() - 5, np.degrees(theta).max() + 5)
    ax_torq.set_ylim(u.min() - 0.1, u.max() + 0.1)

    fig.suptitle(title, color="#e0e0e0", fontsize=13, y=1.01)
    plt.tight_layout()

    # Subsample to ~25 fps for smooth animation regardless of data density
    n_frames = min(len(t), int((t[-1] - t[0]) / (interval_ms / 1000)))
    frame_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)

    def _update(frame):
        k = frame_idx[frame]
        bx = L * np.sin(theta[k])
        by = -L * np.cos(theta[k])

        rod_line.set_data([0, bx], [0, by])
        bob_patch.center = (bx, by)

        # Trail
        idx0 = max(0, k - trail_len + 1)
        tx = L * np.sin(theta[idx0:k+1])
        ty = -L * np.cos(theta[idx0:k+1])
        trail_line.set_data(tx, ty)

        # Torque arrow at pivot
        arrow_scale = 0.15 * L
        torq_arrow.xy      = (arrow_scale * u[k], 0)
        torq_arrow.xytext  = (0, 0)

        theta_trace.set_data(t[:k+1], np.degrees(theta[:k+1]))
        torq_trace.set_data(t[:k+1],  u[:k+1])
        time_text.set_text(f"t = {t[k]:.2f} s")
        return rod_line, bob_patch, trail_line, torq_arrow, theta_trace, torq_trace, time_text

    anim = FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=True,
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg" if save_path.endswith(".mp4") else "pillow",
                  fps=1000 // interval_ms, dpi=120)

    return anim


# ── 2. Phase portrait ─────────────────────────────────────────────────────────

def plot_phase_portrait(
    trajectories: List[Dict],
    title: str = "Phase Portrait",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot one or more (theta, theta_dot) trajectories.

    Parameters
    ----------
    trajectories : list of dicts with keys:
        theta     : (N,) rad
        theta_dot : (N,) rad/s
        label     : str
        color     : optional matplotlib colour
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")

    colors = [_BLUE, _ORANGE, _GREEN, _YELLOW]
    for i, traj in enumerate(trajectories):
        theta     = np.degrees(traj["theta"])
        theta_dot = np.degrees(traj.get("theta_dot", np.gradient(traj["theta"])))
        col       = traj.get("color", colors[i % len(colors)])
        ax.plot(theta, theta_dot, "-", color=col, label=traj.get("label", f"traj {i}"),
                alpha=0.9, lw=1.8)
        ax.plot(theta[0], theta_dot[0], "o", color=col, ms=7)
        ax.plot(theta[-1], theta_dot[-1], "s", color=col, ms=7)

    ax.set_xlabel("θ (deg)", fontsize=11)
    ax.set_ylabel("θ̇ (deg/s)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


# ── 3. Model comparison ───────────────────────────────────────────────────────

def plot_model_comparison(
    t:             np.ndarray,
    y_true:        np.ndarray,
    models:        List[Dict],
    title:         str = "Model Comparison",
    y_label:       str = "θ (rad)",
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Compare true plant output against one or more model predictions.

    models : list of dicts with keys:
        y      : (N,) predicted output
        label  : str
        color  : optional
    """
    n_models = len(models)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), facecolor="#1a1a2e",
        gridspec_kw={"height_ratios": [2, 1]},
    )

    colors = [_BLUE, _ORANGE, _GREEN, _YELLOW]

    ax_out, ax_err = axes
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.grid(True, alpha=0.3)

    ax_out.plot(t, y_true, "-", color=_GREY, lw=2.5, label="True plant", alpha=0.9)
    for i, m in enumerate(models):
        col = m.get("color", colors[i % len(colors)])
        ax_out.plot(t, m["y"], "--", color=col, lw=1.8, label=m.get("label", f"Model {i}"))

    ax_out.set_ylabel(y_label, fontsize=11)
    ax_out.set_title(title, fontsize=13, pad=8)
    ax_out.legend(fontsize=9)

    for i, m in enumerate(models):
        col   = m.get("color", colors[i % len(colors)])
        error = m["y"] - y_true
        rmse  = float(np.sqrt(np.mean(error**2)))
        ax_err.plot(t, error, "-", color=col, lw=1.5,
                    label=f'{m.get("label", f"Model {i}")}  RMSE={rmse:.4f}')

    ax_err.axhline(0, color=_GREY, lw=1, alpha=0.5)
    ax_err.set_xlabel("Time (s)", fontsize=11)
    ax_err.set_ylabel("Residual", fontsize=11)
    ax_err.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 4. Residual heatmap in state space ────────────────────────────────────────

def plot_residual_map(
    theta_grid:     np.ndarray,
    omega_grid:     np.ndarray,
    error_grid:     np.ndarray,
    title:          str = "Residual Map in State Space",
    ax:             Optional[plt.Axes] = None,
    save_path:      Optional[str] = None,
) -> plt.Axes:
    """
    2-D heatmap of model error in (θ, θ̇) space.

    Parameters
    ----------
    theta_grid : (M, K) meshgrid of θ values in degrees
    omega_grid : (M, K) meshgrid of θ̇ values in deg/s
    error_grid : (M, K) model error (e.g. RMSE) at each cell
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")

    pcm = ax.pcolormesh(
        theta_grid, omega_grid, error_grid,
        cmap="plasma", shading="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Error", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e0e0e0")

    ax.set_xlabel("θ (deg)", fontsize=11)
    ax.set_ylabel("θ̇ (deg/s)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ── 5. Control response ───────────────────────────────────────────────────────

def plot_control_response(
    t:              np.ndarray,
    theta:          np.ndarray,
    theta_ref:      np.ndarray,
    u:              np.ndarray,
    title:          str = "Closed-Loop Control Response",
    extra_responses: Optional[List[Dict]] = None,
    save_path:      Optional[str] = None,
) -> plt.Figure:
    """
    Show closed-loop tracking response: angle vs reference + control effort.

    extra_responses : optional list of dicts {theta, label, color}
                      for comparing controllers (e.g. true-model vs identified-model)
    """
    fig, (ax_t, ax_u) = plt.subplots(
        2, 1, figsize=(12, 7), facecolor="#1a1a2e",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for ax in (ax_t, ax_u):
        ax.set_facecolor("#16213e")
        ax.grid(True, alpha=0.3)

    colors = [_BLUE, _ORANGE, _GREEN, _YELLOW]

    ax_t.plot(t, np.degrees(theta_ref), "--", color=_GREY, lw=2, label="Reference", alpha=0.8)
    ax_t.plot(t, np.degrees(theta),     "-",  color=_BLUE,  lw=2, label="Controller (identified model)")
    if extra_responses:
        for i, resp in enumerate(extra_responses):
            col = resp.get("color", colors[(i + 1) % len(colors)])
            ax_t.plot(t, np.degrees(resp["theta"]), "-", color=col, lw=1.8,
                      label=resp.get("label", f"Response {i+1}"), alpha=0.85)

    ax_t.set_ylabel("θ (deg)", fontsize=11)
    ax_t.set_title(title, fontsize=13, pad=8)
    ax_t.legend(fontsize=9)

    ax_u.plot(t, u, "-", color=_GREEN, lw=1.8, label="Control torque")
    ax_u.axhline(0, color=_GREY, lw=1, alpha=0.5)
    ax_u.set_xlabel("Time (s)", fontsize=11)
    ax_u.set_ylabel("τ (N·m)", fontsize=11)
    ax_u.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 6. Validity region overlay ────────────────────────────────────────────────

def plot_validity_overlay(
    theta_range:  Tuple[float, float],
    omega_range:  Tuple[float, float],
    valid_bounds: Dict[str, Tuple[float, float]],
    trajectories: Optional[List[Dict]] = None,
    title:        str = "Validity Region",
    ax:           Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Draw the certified validity region as a rectangle in state space,
    optionally overlaying experiment trajectories.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")

    ax.set_xlim(*[np.degrees(x) for x in theta_range])
    ax.set_ylim(*[np.degrees(x) for x in omega_range])

    # Valid region rectangle
    th_lo, th_hi = [np.degrees(v) for v in valid_bounds.get("theta", theta_range)]
    om_lo, om_hi = [np.degrees(v) for v in valid_bounds.get("theta_dot", omega_range)]
    rect = mpatches.Rectangle(
        (th_lo, om_lo), th_hi - th_lo, om_hi - om_lo,
        linewidth=2, edgecolor=_GREEN, facecolor=_GREEN, alpha=0.15,
        label="Certified validity region",
    )
    ax.add_patch(rect)

    if trajectories:
        colors = [_BLUE, _ORANGE, _YELLOW]
        for i, traj in enumerate(trajectories):
            col = traj.get("color", colors[i % len(colors)])
            ax.plot(np.degrees(traj["theta"]), np.degrees(traj.get("theta_dot", [])),
                    "-", color=col, label=traj.get("label", f"traj {i}"), lw=1.5)

    ax.set_xlabel("θ (deg)", fontsize=11)
    ax.set_ylabel("θ̇ (deg/s)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax
