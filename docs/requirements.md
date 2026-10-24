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
- **R14** — VaR is expressed as a return quantile: negative for a loss. A
  return overshoots the VaR when it is below it.

### Backtesting and comparison

- **R3** — Backtest VaR models, in either of two modes:
  - on VaR forecasts computed earlier and passed in as an input;
  - together with the models that produce the forecasts, in one run.
- **R4** — Benchmark models against each other using their backtest results.
- **R5** — Apply several models to the same portfolio and compare their
  results.
- **R15** — Backtests may need their own computation windows, e.g. a Kupiec
  test on rolling 250 days and on rolling 500 days, side by side in one run.
- **R16** — Windows don't start where the series being windowed is still
  `NaN`. A node's warm-up (e.g. an EWMA backcast, a VaR's first window) shows
  up as leading `NaN` in its output, and the first window of the next node
  starts on the first available value. For example, filtered historical
  simulation on 500 days, filtered by an EWMA volatility with a 20-day
  backcast, has its first window on filtered returns 20 to 519.

### Performance

- **R6** — Running many calculations at once (many models, many series) is a
  core use case and must be fast.

### Integration

- **R7** — The library works on its own, and also inside the core developer's
  data service.
- **R17** — The library is used in one-off notebooks and scripts, and as
  part of a service that receives specs as JSON over HTTP.
- **R8** — Callers pass model inputs in as arrays. Building and formatting
  those arrays is the caller's job (for example, the data service's). The
  library validates the inputs it receives and does nothing more with them.
- **R8a** — Dates come from the caller. When dates don't matter to the caller,
  the library offers helpers that fill them in from a start date or an end
  date. These helpers are built on a calendar that decides which dates are
  valid, and users can customize it (e.g. weekends, holidays). The default
  calendar is Monday to Friday, without holidays.
- **R9** — Users can write model specs in a serialized data format.
- **R9b** — An invalid spec is reported precisely: the error says which
  field or value is wrong and what was expected (e.g. a confidence level
  outside (0, 1)), so users know exactly where they went wrong. Checking
  values in the spec as well as in the nodes is fine.
- **R9a** — Users specify models. The library builds the preprocessors those
  models need (e.g. volatility models, calibrations).
- **R12** — After a run, users can serialize its results: model outputs and
  preprocessor outputs.
- **R13** — Data that comes from outside a run (e.g. VaR forecasts for R3's
  first mode, or volatilities from the data service) is declared in the spec
  as a named input, and passed in by name at run time. A missing input is an
  error.

### Maintainability

- **R10** — The core developer must always fully understand how the code is
  orchestrated and which methodology each model uses. Code and docs serve this
  first.
- **R11** — The library is fully typed, so the types tell users exactly what
  each model and function needs. Use loose types (`Any`, `...`) only where
  NumPy's typing can't express the precise type, and explain that limit in a
  comment.

## Derived requirements (proposed, not yet confirmed)

These follow from the stated requirements, or were proposed during design.
The core developer needs to confirm them.

- **D1** — *No look-ahead.* A value on date `t` (a calibration, a volatility,
  a VaR) uses only data up to and including `t`. The VaR on `t` is then the
  forecast for `t + 1`, and backtests (R3) compare the VaR on `t - 1` with the
  return on `t`.
- **D2** — *Uniform forecast output.* Every VaR/ES model produces one forecast
  per date in the same form, so backtests (R3) and comparisons (R4, R5) treat
  all models the same way. Models whose estimate uses a sample of past returns
  (e.g. historical simulation) compute it on a trailing window (D5), so they
  too give one forecast per date.
- **D3** — *Shared preprocessors.* Identical preprocessor specs share one
  computation within a run, even across separately written model specs (R6,
  R9a). Implemented two ways: in Python, nodes given the same instance share
  its computation (e.g. backtests given one `Overshootings`); from specs, one
  `Builder` gives one node per distinct spec value, so identical parts of
  separately written specs are computed once.
- **D4** — *Reusing saved results.* Results saved from earlier runs (R12) can
  be loaded back to skip work. Loading them never changes a result, only the
  time a run takes. Wanted, but the design needs fleshing out before
  implementing.
- **D5** — *Rolling windows.* Rolling calculations go through a `Windows`
  object: the windows over a series, in the library's array layout (time
  first, then the windows as a batch axis, then the series). It carries the
  dates inside each window and the date each result belongs to, the window's
  last date. Results go back onto the series' dates, each date holding the
  latest result available on it.
- **D6** — *Independent schedules.* Each node has its own window and step. A
  node reads its inputs as of its own calculation date, i.e. their latest
  values on that date. A windowed model gets, for each window, the returns in
  the window and its inputs as of the window's last date.
- **D7** — *Re-filtering after recalibration.* When an EWMA decay is
  recalibrated on a training window, the volatility is re-filtered with the
  new decay over that training window, starting from its backcast, and then
  continues with that decay until the next calibration. The training window is
  specified once, on the calibration, and is usually much longer than the
  backcast (e.g. 250 days vs 20). This is how `rugarch` and `arch` produce
  forecasts after a refit.

## Open questions

Work items and design questions in progress are tracked in
`docs/open-points.md`.

- Should specs offer shorthands and defaults, so users only write the models
  and the library fills in their preprocessors (R9a)? Today every node is
  spelled out.
- Which backtests are needed beyond Kupiec POF (e.g. Christoffersen
  independence, conditional coverage, Basel traffic light), and which ES
  backtests?
- Which criteria should rank models in benchmarks (e.g. backtest p-values,
  quantile/tick loss, Diebold–Mariano tests)?
- Forecast horizon: 1 day only, or multi-day with scaling?
- Array convention for R2: today, trailing axes are independent series. Which
  axis will hold the assets of a multivariate model?
