# Estimator Agent — Developer Reference

`agents/estimator.py` · `EstimatorAgent`

Deterministic parameter-fitting agent. Runs an inner **excite → estimate** loop
on the plant until parameter uncertainty (coefficient of variation) drops below
10 % or the iteration budget is exhausted. No LLM is involved.

---

## Purpose

Given a model structure stored in the registry, `EstimatorAgent`:

1. Designs informative PRBS experiments and applies them to the plant.
2. Computes an OLS warm-start from numerical derivatives (no ODE integration).
3. Fits parameters via simulation-based nonlinear least-squares (NLS) with
   multi-shooting to prevent trajectory drift.
4. Repeats until the covariance criterion is met or `MAX_INNER_ITER` is reached.
5. Stores the fitted model and covariance matrix in the registry and returns a
   `Report`.

---

## Constructor

```python
EstimatorAgent(
    plant_api:  PlantAPI,
    registry:   ModelRegistry,
    db:         ExperimentDatabase,
    n_samples:  int = 600,          # samples per experiment (~12 s at dt=0.02)
)
```

The agent is wired at pipeline startup (`orchestrator.build_real_graph()`).
Callers inside the graph invoke it as a callable node:

```python
dossier = estimator(dossier)
```

The `__call__` method delegates to `run(model_id, contract_id)` and writes the
fitted model ID and covariance ID back to `dossier.artifacts`.

---

## Inner excite ↔ estimate loop

```
for iteration in range(MAX_INNER_ITER):           # MAX_INNER_ITER = 5
    1. Design PRBS input (seed = iteration → deterministic across reruns)
    2. Apply to plant via PlantAPI (tagged as SplitFlag.TRAIN)
    3. On first iteration only: run OLS initial guess
    4. Build multi-shooting residual function  (seg_len = 25 samples = 0.5 s)
    5. Run NLS (scipy LM, max_nfev=300)
    6. Update best_params / best_cov if fit improved
    7. Compute CV = std(param_i) / |param_i| for all params
    8. If all CV < COV_TARGET_CV (0.10): converged → break
```

### Key constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_INNER_ITER` | 5 | Maximum excite↔estimate repetitions |
| `COV_TARGET_CV` | 0.10 | Convergence: all param CVs < 10 % |
| `seg_len` | 25 | Multi-shooting segment length (samples) |
| `n_samples` | 600 | Samples per PRBS experiment (default) |

---

## OLS warm-start

On the first dataset, the estimator solves for an initial parameter guess using
ordinary least squares on numerically differentiated outputs — no ODE integration
needed. For a model linear in parameters:

```
θ̈ = Σⱼ pⱼ · (∂rhs/∂pⱼ)(θ, θ̇, u)
```

SymPy computes each regressor `∂rhs/∂pⱼ` symbolically; the columns are evaluated
on smoothed output signals (Savitzky–Golay, order 3). Parameters listed in
`p0_override` (e.g. `K_c` from the grey-box sub-orchestrator) are kept as-is and
are never overwritten by OLS.

---

## Multi-shooting

Single-shot trajectory simulation over a 12 s PRBS experiment diverges when the
model is misspecified (e.g. missing Coulomb term) — a ~25 % parameter error
causes ~0.5 rad drift per 0.5 s. Multi-shooting fixes this by re-initializing the
integrator at every `seg_len`-sample boundary from the measured position and a
Savitzky–Golay velocity estimate.

---

## Initial guess priority

1. `p0_override` (pre-specified, e.g. `K_c` from grey-box) — never overwritten
2. OLS on the first dataset (for params not in override)
3. 10 % of upper bound from `param_bounds` if OLS fails
4. `1.0` as a last resort

---

## Registry writes

After the loop the agent stores two artifacts:

| Artifact | Key | Description |
|---|---|---|
| Fitted model | `fitted_id` | `ModelArtifact` with `parameters` dict and `parent_id` |
| Covariance | `fitted_id + "_cov"` | `np.ndarray` (n_params × n_params) stored via `store_covariance` |

The model type (`WHITE_BOX` or `GREY_BOX`) is inherited from the parent model so
the dossier's rung is preserved correctly across iterations.

---

## Report metadata

```python
{
    "model_id":       str,           # registry key of the fitted model
    "covariance_id":  str,           # registry key of the covariance matrix
    "converged":      bool,          # True if CV < 10 % for all params
    "params":         dict[str,float],
    "run_ids":        list[str],     # DB run IDs used during fitting
    "stalled_reason": str,           # non-empty if stopped by budget/safety
}
```

The orchestrator reads `metadata["model_id"]` to advance `dossier.artifacts.current_model_id`.

---

## Design notes

- **No LLM.** All decisions (convergence check, OLS, NLS) are deterministic.
- **Budget safety.** `PlantAPI.apply_input` raises if the budget is exhausted or
  a safety limit is breached; the estimator catches this, sets `stalled_reason`,
  and returns with whatever parameters it has so far.
- **Covariance from Jacobian.** `nonlinear_least_squares` uses scipy's `lm`
  backend and returns `fit["covariance"]` — the parameter covariance estimated
  from the residual Jacobian at the solution. This propagates directly to CV.
