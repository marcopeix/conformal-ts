# conformal-ts — Project Guide for Claude Code

## What this package is

`conformal-ts` is a Python library providing **adaptive conformal prediction methods for time series forecasting**. It wraps any forecasting model (Nixtla, Darts, sktime, callable, TSFMs) and produces calibrated prediction intervals using methods designed for non-exchangeable, sequential data.

The package's positioning is **TS-native CP for forecasting**, complementing MAPIE (which is sklearn-shaped and split-CP-focused) by prioritizing online and adaptive methods (ACI, AgACI, NexCP, EnbPI, SPCI, CQR).

## Architecture in one paragraph

Three abstractions in `src/conformal_ts/base.py` define the entire package:
`ForecasterAdapter` (uniform interface over forecasting libraries),
`ScoreFunction` (nonconformity scoring), and `ConformalMethod` (the user-facing
calibrate/predict object). Every file in `methods/`, `adapters/`, and
`nonconformity/` implements one of these. Adapters convert library-specific
objects (Nixtla DataFrames, Darts TimeSeries, etc.) to numpy arrays at the
boundary so methods never see framework types.

## Repo layout

- `src/conformal_ts/base.py` — abstract contracts (READ FIRST)
- `src/conformal_ts/methods/` — one file per CP method (split, aci, ag_aci, nex_cp, enbpi, spci, cqr)
- `src/conformal_ts/adapters/` — one file per supported library (callable, nixtla, darts, sktime, tsfm)
- `src/conformal_ts/nonconformity/` — score functions (absolute, normalized, quantile)
- `src/conformal_ts/calibration/` — calibration window utilities (rolling, expanding, block bootstrap)
- `src/conformal_ts/diagnostics/` — coverage and interval-quality metrics
- `src/conformal_ts/datasets/` — small built-in datasets for tests/examples
- `tests/unit/` — fast tests, no external data
- `tests/integration/` — slow tests across adapters
- `benchmarks/` — comparison vs MAPIE / paper baselines (not run in CI)
- `examples/` — runnable notebooks

## Type and shape conventions

- **All array shapes documented in docstrings.** Always.
- `Series`: 2D `np.ndarray[float]`, shape `(n_series, T)`. A panel of time series.
  Single-series users pass `n_series=1`.
- `Forecast`: 3D `np.ndarray[float]`, shape `(n_series, n_samples, horizon)`.
  Always 3D, even when `n_series=1` or `horizon=1`.
- `Interval`: 4D `np.ndarray[float]`, shape `(n_series, n_samples, horizon, 2)`.
  Last axis is `(lower, upper)`.
- The leading axis is **always** `n_series`. Iterate, slice, and reduce over it
  the way pandas users iterate over groupby keys.
- No pandas in `methods/` or `nonconformity/`. Pandas only inside `adapters/`.

## Capabilities are mixins

- Optional adapter features are mixins in `src/conformal_ts/capabilities.py`:
  `SupportsRefit`, `SupportsQuantiles`, `SupportsBootstrap`.
- Adapters inherit only the mixins they implement.
- Methods declare requirements via the class attribute
  `REQUIRED_CAPABILITIES: tuple[type, ...] = (SupportsQuantiles,)` etc.
- The constructor of `ConformalMethod` checks each required mixin via
  `isinstance`. **Never check capabilities inside `predict()`.**

## Invariants every method must satisfy

1. `alpha` lives in `(0, 1)`. Validate in `__init__`.
2. `calibrate()` must precede `predict()`. Raise `CalibrationError` otherwise.
3. Capability requirements (`REQUIRES_QUANTILES`, `REQUIRES_BOOTSTRAP`) are
   class attributes checked in `ConformalMethod.__init__`. **Never check them
   inside `predict()`.**
4. `predict()` is deterministic given fitted state and input.
5. Online methods (`IS_ONLINE = True`) implement non-trivial `update()`.
   Offline methods inherit the no-op default.

## Style and tooling

- Python 3.10+. Use modern syntax: `X | None`, `list[int]`, PEP 695 where natural.
- `from __future__ import annotations` at the top of every module.
- Type-annotate all public functions. Use `numpy.typing.NDArray`.
- Docstrings: NumPy style (Parameters / Returns / Raises sections).
- Format and lint with `ruff` (config in pyproject.toml).
- Type-check with `mypy --strict` on `src/`. Tests can be looser.
- 100-char line limit.

## Dependency policy

- Core deps (`numpy`, `pandas`, `scipy`) only. Everything else is an extra.
- Adapters use **soft imports**: try-import the library, raise an
  `ImportError` with the exact `pip install conformal-ts[<extra>]` command
  if missing.
- Never add a new core dependency without explicit user approval.

## Testing conventions

- Every method in `methods/` has a corresponding `tests/unit/test_<method>.py`.
- Every method test must include:
  - **Coverage test**: on synthetic data with known noise, empirical coverage
    matches `1 - alpha` within a tolerance documented in the test.
  - **Shape test**: outputs have the documented shapes for several
    `(n_samples, horizon)` combinations including `horizon=1`.
  - **Determinism test**: same input + seed → same output.
  - **Capability error test**: a method that needs quantiles/bootstrap raises
    `UnsupportedCapability` when given an inadequate adapter.
- Use `numpy.random.default_rng(seed)` everywhere. Never `np.random.seed`.
- The `CallableAdapter` is the test workhorse — it's the simplest adapter
  and lets every method be tested without external dependencies.

## Adding a new method (procedure)

1. Read `src/conformal_ts/base.py` and `methods/split.py` (the simplest reference).
2. Create `methods/<name>.py` with a class inheriting `ConformalMethod`.
3. Set `REQUIRED_CAPABILITIES` and `IS_ONLINE` class attributes.
4. Implement `_default_score`, `calibrate`, `predict`, and `update` (if online).
5. Tests in `tests/unit/test_<name>.py` covering: coverage, shape (with
   `n_series` in {1, 5} and `horizon` in {1, 12}), determinism, and capability errors.

## Adding a new adapter (procedure)

1. Read `adapters/callable.py` (simplest reference).
2. Create `adapters/<library>.py` inheriting `ForecasterAdapter` plus the
   capability mixins it supports (`SupportsRefit`, `SupportsQuantiles`,
   `SupportsBootstrap`).
3. Implement every abstract method from each parent class.
4. Soft-import the underlying library at the top of the file. If missing,
   raise `ImportError` with the exact `pip install conformal-ts[<extra>]` command.
5. Tests in `tests/integration/test_<library>_adapter.py`.
6. Add an extras entry in `pyproject.toml` and document it in the README.

## What NOT to do

- Don't import pandas inside `methods/` or `nonconformity/`.
- Don't make `predict()` raise capability errors. Capability checks happen at construction.
- Don't add framework-specific types to the public API.
- Don't write a method that only works for `horizon=1`. Multi-horizon is mandatory.
- Don't add a top-level dependency without approval.
- Don't replace numpy operations with pandas inside hot loops.

## Running things

```bash
uv sync --all-extras --group dev   # install everything for development
uv run pytest tests/unit            # fast tests
uv run pytest tests/integration     # slow tests, needs extras
uv run ruff check .
uv run mypy src/
uv build                            # build wheel + sdist
```

## When in doubt

Read `base.py`. Read an existing implementation in the same folder. Match its
style. If the contract in `base.py` doesn't fit your method, that's a design
question — flag it, don't work around it.
