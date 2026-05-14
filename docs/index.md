# conformal-ts

Adaptive conformal prediction for time series forecasting.

`conformal-ts` provides distribution-free prediction intervals for any
forecasting model — Nixtla, Darts, sktime, or your own — with a focus on
methods designed for non-exchangeable, sequential data.

## What's here

- **[Quickstart](quickstart.md)** — produce calibrated intervals in 5 minutes.
- **[Concepts](concepts.md)** — why conformal prediction matters for time series.
- **[Methods](methods/index.md)** — Split, CQR, ACI, AgACI, NexCP, SPCI.
- **[Adapters](adapters/index.md)** — connect any forecasting library.
- **[API Reference](api/index.md)** — auto-generated from source.

## Installation

```bash
pip install conformal-ts
```

See [Installation](installation.md) for optional extras (Nixtla, Darts, sktime,
SPCI quantile regressor, plotting).

## Status

Early development. APIs are unstable until v0.2.0. Bug reports and
contributions welcome via the [GitHub repository](https://github.com/marcopeix/conformal-ts).
