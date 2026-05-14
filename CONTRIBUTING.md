# Contributing to conformal-ts

`conformal-ts` is a Python library for adaptive conformal prediction on time
series forecasts. This document is for anyone opening a pull request — bug
fixes, new methods, new adapters, docs, or examples. It covers how to set up
the project, the conventions you need to follow, and what to expect during
review.

## Quick start

```bash
git clone https://github.com/marcopeix/conformal-ts
cd conformal-ts
uv sync --all-extras --all-groups
uv run pre-commit install
uv run pytest -v
```

Branch from `main`, push to your fork, and open a PR against `main`. The CI
matrix runs on Python 3.10–3.13 against `tests/unit`.

## What contributions are welcome

- **Bug fixes** — in any file. Always add a regression test.
- **New conformal methods** — go in `src/conformal_ts/methods/` with a unit
  test in `tests/unit/`.
- **New adapters** — go in `src/conformal_ts/adapters/` with an integration
  test in `tests/integration/` and a new extras entry in `pyproject.toml`.
- **New nonconformity scores** — go in `src/conformal_ts/nonconformity/`.
- **New aggregators or quantile-regression backends** — go in
  `src/conformal_ts/aggregators/` and `src/conformal_ts/quantile_regressors/`
  respectively. Both are pluggable ABCs designed for extension.
- **Documentation and examples** — docstring fixes, notebook examples under
  `examples/`, README clarifications.
- **Performance work** — benchmarked improvements to hot loops. Include
  before/after numbers in the PR description.

Open an issue first for: large architectural changes, new core dependencies,
or anything that would significantly grow install size. These are not
rejections — they need a design conversation before code.

## Project layout

```
src/conformal_ts/
├── base.py              # ForecasterAdapter, ScoreFunction, ConformalMethod ABCs; type aliases
├── capabilities.py      # Capability mixins (SupportsRefit, SupportsQuantiles, ...)
├── methods/             # CP methods (Split, CQR, ACI, AgACI, NexCP, SPCI)
├── adapters/            # Forecaster adapters (Callable, StatsForecast, MLForecast,
│                        #   NeuralForecast, Darts, sktime, TSFM)
├── nonconformity/       # Score functions (AbsoluteResidual, SignedResidual, QuantileScore)
├── aggregators/         # Online aggregators (EWA) used by AgACI
├── quantile_regressors/ # Quantile regression backends (QRF) used by SPCI
├── diagnostics/         # Coverage, scoring, conditional, reports
├── datasets/            # Small built-in datasets for tests and examples
└── utils/               # Validation, plotting, type helpers

tests/unit/              # Fast tests, no optional dependencies required
tests/integration/       # Slower tests across adapters; gated on extras
examples/                # Runnable notebooks
```

## Conventions that matter

### Array shapes

The leading axis is always `n_series`. The canonical aliases live in
`src/conformal_ts/base.py`:

- `Series` — `(n_series, T)`. A panel of time series.
- `Forecast` — `(n_series, n_samples, horizon)`. Always 3D, even when
  `n_series=1` or `horizon=1`.
- `Interval` — `(n_series, n_samples, horizon, 2)`. The last axis is
  `(lower, upper)`.

```python
# A PredictionResult from a calibrated method
result = method.predict(history)              # history: (n_series, T)
result.point.shape    # (n_series, 1, horizon)
result.interval.shape # (n_series, 1, horizon, 2)
```

Document the shape of every array in every public docstring. Reviewers will
ask if you don't.

### Trailing underscore on fitted state

Following sklearn's convention, attributes set in `__init__` have no trailing
underscore; attributes set in `calibrate()` or `update()` do. A
contributor reading a method should be able to tell what is configuration
versus fitted state at a glance:

```python
self.alpha            # configuration, set in __init__
self.gamma            # configuration
self.alpha_t_         # fitted state, set in calibrate()
self.scores_          # fitted state, grows in update()
self.is_calibrated_   # fitted state
```

### Capability mixins

Adapters declare what they can do by inheriting from mixins in
`src/conformal_ts/capabilities.py`: `SupportsRefit`, `SupportsQuantiles`,
`SupportsBootstrap`, `SupportsCrossValidation`,
`SupportsCrossValidationQuantiles`. Methods declare what they need via the
`REQUIRED_CAPABILITIES` class attribute. `ConformalMethod.__init__` checks
required mixins with `isinstance` and raises `UnsupportedCapability` if the
adapter is inadequate.

Never check capabilities inside `predict()`. The check happens once, at
construction.

### Pandas at the boundary only

Pandas is allowed inside `adapters/`, where it converts framework objects
(e.g., StatsForecast DataFrames) to numpy panels. It is not allowed inside
`methods/` or `nonconformity/`. Those layers work in numpy.

### Online lifecycle

Online methods (`IS_ONLINE = True`) implement a non-trivial `update()`. The
contract is: after `calibrate()`, callers alternate `predict(history)` and
`update(prediction, truth)`. The caller passes back the `point` from the
`PredictionResult` they received — no hidden state coupling between `predict`
and `update`. See `src/conformal_ts/methods/aci.py` for the canonical
example.

### Dependencies

Core dependencies are `numpy`, `pandas`, `scipy`. Anything else is an extra
in `pyproject.toml` (`nixtla`, `darts`, `sktime`, `spci`, `plotting`).
Adapters soft-import their underlying library and raise an `ImportError` with
the exact install command if it's missing:

```python
try:
    from statsforecast import StatsForecast
except ImportError as _err:
    raise ImportError(
        "StatsForecastAdapter requires the 'statsforecast' package. "
        "Install it with: pip install conformal-ts[nixtla]"
    ) from _err
```

Do not add a new top-level dependency without opening an issue first.

## Adding things

### Adding a new conformal method

1. Read `src/conformal_ts/base.py` for the `ConformalMethod` contract.
2. Read `src/conformal_ts/methods/split.py` (offline reference) and
   `src/conformal_ts/methods/aci.py` (online reference).
3. Create `src/conformal_ts/methods/<name>.py` with a class inheriting
   `ConformalMethod`.
4. Set `REQUIRED_CAPABILITIES: tuple[type, ...] = (...)` and `IS_ONLINE: bool`
   as class attributes.
5. Implement `_default_score`, `calibrate`, `predict`, and (if online)
   `update`.
6. Add the class to `src/conformal_ts/methods/__init__.py`.
7. Add `tests/unit/test_<name>.py` mirroring the structure of
   `tests/unit/test_aci.py`: coverage, shape, online adaptation,
   lifecycle errors, invalid parameters, capability requirements.

### Adding a new adapter

1. Read `src/conformal_ts/adapters/callable.py` (the minimal reference) and
   `src/conformal_ts/adapters/statsforecast.py` (a real-library reference).
2. Create `src/conformal_ts/adapters/<library>.py` inheriting
   `ForecasterAdapter` plus whichever capability mixins it supports.
3. Soft-import the underlying library at the top of the file and raise an
   `ImportError` with the install command on failure.
4. Implement every abstract method from each parent class. Convert
   framework-specific types (DataFrames, TimeSeries, etc.) to numpy at the
   boundary.
5. Add an entry under `[project.optional-dependencies]` in `pyproject.toml`
   and reference it from the install message.
6. Add `tests/integration/test_<library>_adapter.py`. Gate the module on
   `pytest.importorskip("<library>")` so unit-test runs without extras still
   pass.

### Adding a new nonconformity score

1. Read `src/conformal_ts/nonconformity/absolute.py` (the simplest reference).
2. Create `src/conformal_ts/nonconformity/<name>.py` with a class inheriting
   `ScoreFunction`.
3. Implement `score`, `invert`, and `fit` (no-op if there's no internal
   state to learn).
4. Export the class from `src/conformal_ts/nonconformity/__init__.py`.
5. Add a unit test alongside the existing `test_absolute_residual.py` and
   `test_quantile_score.py`.

## Code style and tooling

- Python 3.10+. Use modern syntax: `X | None`, `list[int]`, PEP 695 where
  natural.
- `from __future__ import annotations` at the top of every module.
- Type-annotate all public functions. Use `numpy.typing.NDArray`.
- NumPy-style docstrings (Parameters / Returns / Raises). Document array
  shapes.
- 100-character line limit.
- Tools and commands:
  - `uv run ruff check .` — lint
  - `uv run ruff format .` — format (or `--check` for CI parity)
  - `uv run mypy src/` — strict type-checking on `src/`; tests are looser
  - `uv run pre-commit install` — wire up the pre-commit hooks (ruff,
    ruff-format, trailing whitespace, end-of-file, YAML/TOML syntax)

## Tests

- Every method in `methods/` has a matching `tests/unit/test_<method>.py`.
- Every method test includes:
  - **Coverage** on synthetic data: empirical coverage matches `1 - alpha`
    within a documented tolerance.
  - **Shape**: outputs match the documented shapes for `n_series` in `{1, 3}`
    and `horizon` in `{1, 6}`.
  - **Determinism**: same input plus same seed produces the same output.
  - **Capability errors**: a method that needs `SupportsQuantiles` /
    `SupportsBootstrap` raises `UnsupportedCapability` when given an
    inadequate adapter at construction time.
- Use `numpy.random.default_rng(seed)` everywhere. Never `np.random.seed`.
- Unit tests use `CallableAdapter` (and its CV variant in
  `tests/_online_helpers.py`) so they run without optional dependencies.
- Integration tests gate on `pytest.importorskip("<library>")` for the
  relevant backend.

Reference: `tests/unit/test_aci.py` for the typical online-method pattern.

Run the suites with:

```bash
uv run pytest tests/unit          # fast, no extras required
uv run pytest tests/integration   # slower, needs extras
```

## Using AI tools to write code

> Using AI assistants (Claude, Copilot, Cursor, etc.) to write contributions
> is welcome and normal. The expectation is that you understand and verify
> the code before submitting it.
>
> - Run the tests yourself locally. Don't rely on the assistant's claim that
>   they pass.
> - Read what the assistant produced and confirm it does what you intend.
>   AI-generated code can look reasonable while subtly mishandling edge
>   cases — especially shape conventions and the online predict/update
>   lifecycle.
> - If the assistant suggests changes outside the scope of your PR
>   (drive-by refactors, unrelated formatting), revert them. PRs that bundle
>   unrelated changes are hard to review.
> - You are responsible for the correctness, licensing, and design of your
>   contribution regardless of how it was produced.
>
> The maintainers also use AI tools. The standard is the same on both sides:
> the code works and you can defend it in review.

## Reporting bugs and requesting features

Bug reports go through `.github/ISSUE_TEMPLATE/bug_report.md`. Include a
minimal reproducer, the `conformal-ts` version, your Python version, and the
OS. Feature requests use `.github/ISSUE_TEMPLATE/feature_request.md`.

For non-trivial feature ideas (new method, new adapter, API change), open an
issue before a PR. A short design discussion up front avoids rework later.

## Pull request expectations

The PR template in `.github/PULL_REQUEST_TEMPLATE.md` carries the checklist.
Three things to highlight:

- Keep each PR focused on one change. Split unrelated work into separate PRs.
- Update `CLAUDE.md` if you change project-wide conventions or add a new
  category of files (a new top-level subdirectory under `src/`, a new kind of
  abstract base class, a new testing pattern).
- CI must be green before review. The pipeline runs `ruff check`,
  `ruff format --check`, and `uv run pytest tests/unit` on Python
  3.10–3.13.

## License

Contributions are licensed under the BSD 3-Clause License (see `LICENSE`).
By submitting a PR, you agree your contribution can be released under that
license.
