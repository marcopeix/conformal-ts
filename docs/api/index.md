# API Reference

Auto-generated reference for the public API. For tutorials and concept-level
documentation, see the [Methods](../methods/index.md) and
[Adapters](../adapters/index.md) sections.

## Methods

::: conformal_ts.methods.split.SplitConformal
::: conformal_ts.methods.cqr.ConformalizedQuantileRegression
::: conformal_ts.methods.aci.AdaptiveConformalInference
::: conformal_ts.methods.agaci.AggregatedAdaptiveConformalInference
::: conformal_ts.methods.nexcp.NonexchangeableConformalPrediction
::: conformal_ts.methods.spci.SequentialPredictiveConformalInference

## Adapters

::: conformal_ts.adapters.callable.CallableAdapter
::: conformal_ts.adapters.statsforecast.StatsForecastAdapter
::: conformal_ts.adapters.mlforecast.MLForecastAdapter
::: conformal_ts.adapters.neuralforecast.NeuralForecastAdapter

## Nonconformity Scores

::: conformal_ts.nonconformity.absolute.AbsoluteResidual
::: conformal_ts.nonconformity.quantile.QuantileScore
::: conformal_ts.nonconformity.signed.SignedResidual

## Aggregators

::: conformal_ts.aggregators.base.OnlineAggregator
::: conformal_ts.aggregators.ewa.EWA

## Quantile Regressors

::: conformal_ts.quantile_regressors.base.QuantileRegressor
::: conformal_ts.quantile_regressors.qrf.QRFQuantileRegressor

## Capabilities

::: conformal_ts.capabilities.SupportsRefit
::: conformal_ts.capabilities.SupportsQuantiles
::: conformal_ts.capabilities.SupportsBootstrap
::: conformal_ts.capabilities.SupportsCrossValidation
::: conformal_ts.capabilities.SupportsCrossValidationQuantiles

## Diagnostics

::: conformal_ts.diagnostics.coverage
::: conformal_ts.diagnostics.scoring
::: conformal_ts.diagnostics.conditional
::: conformal_ts.diagnostics.method_diagnostics
::: conformal_ts.diagnostics.reports

## Core Contracts

::: conformal_ts.base.ConformalMethod
::: conformal_ts.base.ForecasterAdapter
::: conformal_ts.base.ScoreFunction
::: conformal_ts.base.CalibrationResult
::: conformal_ts.base.PredictionResult
