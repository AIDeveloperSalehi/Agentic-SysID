# Data Steward — Developer Reference

`agents/data_steward.py` · `DataSteward`

Service agent for dataset coverage and excitation quality assessment. Called
before any new experiment is requested to avoid redundant or uninformative data
collection. No LLM, no side effects — pure DB queries and spectral analysis.

---

## Purpose

The Data Steward answers one question before an agent requests more data:
*does sufficient, sufficiently-exciting data already exist for the intended
purpose?* This prevents the pipeline from re-running expensive plant experiments
when the DB already contains what is needed.

Callers use the report to decide:
- Skip experiment design entirely (data is already sufficient).
- Record which gaps remain and pass them to `ExperimentDesignAgent`.
- Gate surrogate training on the train/validation split.

---

## Constructor

```python
DataSteward(db: ExperimentDatabase)
```

Takes a single dependency: the shared experiment database. Instantiated once at
pipeline startup and injected wherever needed.

---

## Primary interface — `assess_coverage`

```python
report = steward.assess_coverage(
    purpose = "identification",   # optional filter
    split   = SplitFlag.TRAIN,    # optional filter
)
```

### Parameters

| Name | Type | Description |
|---|---|---|
| `purpose` | `str`, optional | Filter runs by purpose label (`"identification"`, `"validation"`, …) |
| `split` | `SplitFlag`, optional | Filter by `TRAIN`, `VALIDATION`, or `BOTH` |

Both filters are passed directly to `ExperimentDatabase.query_runs()`.

### Return value — `DataCoverageReport`

```python
class DataCoverageReport(BaseModel):
    run_ids:    list[str]
    coverage:   dict[str, tuple[float, float]]   # {"output": (min, max), "input": (min, max)}
    excitation: ExcitationQuality                # "sufficient_for_identification" | "validation_only" | "insufficient"
    quality:    float                            # 0–1 scalar score
    usable_for: list[str]                        # subset of ["identify", "validate", "train"]
    gaps:       list[str]                        # human-readable gap descriptions
```

#### `ExcitationQuality` values

| Value | Meaning |
|---|---|
| `SUFFICIENT` | Input has enough spectral content for system identification |
| `VALIDATION_ONLY` | Signal has amplitude but low spectral richness; only usable for model validation |
| `INSUFFICIENT` | Essentially zero excitation or too few samples |

#### `usable_for` population rules

| Tag | Condition |
|---|---|
| `"identify"` | `excitation == SUFFICIENT` |
| `"validate"` | `total_samples >= 100` |
| `"train"` | `total_samples >= 100` |

---

## Excitation quality check

`_assess_excitation(u)` runs a two-criterion heuristic on the concatenated input
signal:

1. **Spectral fraction** — fraction of input energy at non-DC frequencies
   (FFT bins 1 … N/2). A PRBS or multisine should have > 50 % non-DC energy.
2. **Amplitude ratio** — RMS / peak amplitude. Near-zero RMS means the input is
   effectively flat.

```
quality = clip(spectral_fraction × amplitude_ratio × 2, 0, 1)

if spectral_fraction > 0.5 and rms > 1e-6:  → SUFFICIENT
elif rms > 1e-6:                             → VALIDATION_ONLY
else:                                        → INSUFFICIENT
```

---

## Convenience methods

```python
steward.has_identification_data() -> bool
# True if assess_coverage(purpose="identification") returns "identify" in usable_for

steward.list_run_ids(purpose=None) -> list[str]
# All run IDs in the DB, optionally filtered by purpose
```

---

## Typical call pattern

```python
steward  = DataSteward(db)
coverage = steward.assess_coverage(purpose="identification")

if "identify" in coverage.usable_for:
    # Skip new experiments; use coverage.run_ids directly
else:
    # Pass coverage.gaps to ExperimentDesignAgent
    gaps = coverage.gaps
```

---

## Design notes

- **No LLM, no writes.** The steward only reads from the database. It is safe to
  call at any point without side effects.
- **Empty DB case.** If no runs match the filters, the report has
  `excitation=INSUFFICIENT`, `usable_for=[]`, and a single gap entry
  `"No experiments in database."` — callers don't need to special-case `None`.
- **Concatenation across runs.** Coverage and excitation are computed on the
  full concatenated output/input across all matching runs, not per-run.
  This means multiple short experiments are treated as equivalent to one long one.
- **`quality` score.** The 0–1 float is informational; routing decisions use the
  `ExcitationQuality` enum and `usable_for` list, not the raw score.
