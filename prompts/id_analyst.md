# ID Analyst — Developer Reference

`agents/id_analyst.py` · `IDAnalyst`

Service agent for structural and practical identifiability analysis. Invoked
synchronously — no LLM, no async. Returns an `IdentifiabilityReport` that the
caller uses to decide whether to proceed, reparameterize, or redesign the
experiment.

---

## Purpose

Before fitting, the pipeline must know whether the parameters in a candidate
ODE structure can, in principle, be recovered from input–output data. Two
questions are answered in sequence:

1. **Structural identifiability** — is the model uniquely identifiable from
   perfect, noise-free I/O data? Checked symbolically via SymPy
   (`check_structural_identifiability`). If no parameters pass this test, the
   report is returned immediately without consulting data.

2. **Practical identifiability** — given the *actual* data already collected,
   does the Fisher Information Matrix (FIM) have full rank? A rank-deficient FIM
   means some parameter direction is unobservable in practice (not enough
   excitation). Computed via `compute_fim` when `existing_t`/`existing_u` are
   supplied.

---

## Callers

| Caller | When | Why |
|---|---|---|
| `ModelerAgent` | After ODE structure is set, before committing it | Catches non-identifiable lumped params early |
| `EstimatorAgent` | Before starting the inner excite↔estimate loop | Confirms which params are recoverable given available data |

Both callers instantiate `IDAnalyst()` directly and call `.analyze()`.

---

## Interface

```python
from agents.id_analyst import IDAnalyst

analyst = IDAnalyst()
report  = analyst.analyze(
    normalized_rhs = "K_u*u - K_d*x_dot - K_s*f(x)",
    fit_params     = ["K_u", "K_d", "K_s"],
    state_vars     = ["x", "x_dot"],
    input_vars     = ["u"],
    lumped_names   = {"coeff_u": "K_u", ...},       # optional
    existing_t     = t_array,                        # optional — enables FIM check
    existing_u     = u_array,                        # optional — enables FIM check
    noise_var      = 1e-6,                           # assumed output noise variance
)
```

### Parameters

| Name | Type | Description |
|---|---|---|
| `normalized_rhs` | `str` | RHS of the highest-derivative ODE, using `fit_params` symbol names |
| `fit_params` | `list[str]` | Parameters to test |
| `state_vars` | `list[str]` | State variable names (same order as ODE) |
| `input_vars` | `list[str]` | Input/actuator names |
| `lumped_names` | `dict`, optional | Human-readable aliases for lumped parameter keys |
| `existing_t`, `existing_u` | `np.ndarray`, optional | Time vector and input array; FIM check is skipped if absent |
| `noise_var` | `float` | Measurement noise variance for FIM weighting (default `1e-6`) |

### Return value — `IdentifiabilityReport`

```python
class IdentifiabilityReport(BaseModel):
    identifiable:             IdentifiabilityResult   # "full" | "partial" | "none"
    non_identifiable_params:  list[str]               # params that cannot be recovered
    recommendation:           str                     # human-readable action hint
    reparameterized_model_id: str | None              # set by caller after reparam
```

`IdentifiabilityResult` values and their meaning:

| Value | Meaning |
|---|---|
| `FULL` | All params identifiable; proceed with estimation |
| `PARTIAL` | Some params unrecoverable; reparameterize or fix those params |
| `NONE` | No params identifiable; model structure must change |

---

## Decision logic

```
structural check
    → "none"          → return NONE immediately (no FIM check)
    → "partial/full"  → if data provided, run FIM check
                            → FIM rank-deficient → PARTIAL + list of weak params
                            → FIM full rank      → FULL
```

Structural non-identifiability always dominates: a structurally non-identifiable
parameter cannot be rescued by collecting more data.

The FIM check uses nominal parameter values of `1.0` for all params (because the
ODE is assumed linear-in-parameters at this stage). The near-zero eigenvalue
heuristic marks the last `n_small` parameters (in `fit_params` order) as
practically non-identifiable.

---

## Design notes

- **No LLM.** The analysis is pure SymPy + numpy. The agent holds no state
  between calls; instantiate fresh or reuse the same instance freely.
- **Silent FIM failure.** If `compute_fim` raises (e.g. simulator integration
  fails), the exception is caught and the structural result is returned as-is —
  practical identifiability is treated as unknown rather than failing the run.
- **`lumped_names`** is forwarded to `check_structural_identifiability` so that
  the SymPy report uses readable names rather than generated coefficient symbols.
