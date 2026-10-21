# Requirements

tail-lib is a library for measuring market risk, and for backtesting and
comparing the models that measure it.

Requirements are numbered so changes and reviews can cite them (e.g. "R4").

## Stated requirements

### Risk measures

- **R1** — Compute univariate risk measures: volatility, value-at-risk (VaR) and
  expected shortfall (ES).
- **R2** — Multivariate models will be added later. Design choices made now
  must leave room for them.

### Backtesting and comparison

- **R3** — Backtest VaR models.
- **R4** — Benchmark models against each other using their backtest results.
- **R5** — Apply several models to the same portfolio and compare their
  results.

### Performance

- **R6** — Running many calculations at once (many models, many series) is a
  core use case and must be fast.

### Integration

- **R7** — The library works on its own, and also inside the core developer's
  data service.
- **R8** — Callers pass model inputs in as arrays. Building and formatting
  those arrays is the caller's job (for example, the data service's). The
  library validates the inputs it receives and does nothing more with them.
- **R8a** — Dates come from the caller. When dates don't matter to the caller,
  the library offers helpers that fill them in from a start date or an end
  date. These helpers are built on a calendar that decides which dates are
  valid, and users can customize it (e.g. weekends, holidays). The default
  calendar is Monday to Friday, without holidays.
- **R9** — Users can write model specs in a serialized data format.

### Maintainability

- **R10** — The core developer must always fully understand how the code is
  orchestrated and which methodology each model uses. Code and docs serve this
  first.
- **R11** — The library is fully typed, so the types tell users exactly what
  each model and function needs. Use loose types (`Any`, `...`) only where
  NumPy's typing can't express the precise type, and explain that limit in a
  comment.

## Derived requirements (proposed, not yet confirmed)

These follow from the stated requirements, and the core developer needs to
confirm them.

- **D1** — *No look-ahead.* A forecast for date `t` uses only data from before
  `t`. Backtesting (R3) is meaningless without this.
- **D2** — *Uniform forecast output.* Every VaR/ES model produces one forecast
  per date in the same form, so backtests (R3) and comparisons (R4, R5) treat
  all models the same way.
- **D3** — *Shared work identified by spec.* Two models whose specs share an
  identical component (e.g. the same volatility calibration) compute it only
  once, even if each was built from a separate serialized spec (R6, R9).

## Open questions

- Which serialization format(s) should specs use: JSON, YAML, TOML?
- Which backtests are needed (e.g. Kupiec POF, Christoffersen independence,
  conditional coverage, Basel traffic light), and which ES backtests?
- Which criteria should rank models in benchmarks (e.g. backtest p-values,
  quantile/tick loss, Diebold–Mariano tests)?
- Sign convention: is VaR reported as a return quantile (negative) or as a
  positive loss?
- Forecast horizon: 1 day only, or multi-day with scaling?
- Array convention for R2: today, trailing axes are independent series. Which
  axis will hold the assets of a multivariate model?
