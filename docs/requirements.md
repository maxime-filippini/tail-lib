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
- **R16** — *Superseded by R21–R23 (2026-10-25); the code still follows it
  until the run date range is built.* Windows don't start where the series
  being windowed is still `NaN`. A node's warm-up (e.g. an EWMA backcast, a
  VaR's first window) shows up as leading `NaN` in its output, and the first
  window of the next node starts on the first available value. For example,
  filtered historical simulation on 500 days, filtered by an EWMA volatility
  with a 20-day backcast, has its first window on filtered returns 20 to 519.

### Runs

- **R21** — A run treats all its portfolios uniformly, so its results are
  comparable across portfolios. Its returns are trimmed at the first date
  where every portfolio has a value: a late-starting portfolio makes every
  portfolio of the run start on its first date. Users who want portfolios
  treated differently split them into several runs.
- **R22** — A run computes its VaR models between a lower and an upper date,
  so every model's VaR covers the same dates. Before computing, the run
  checks that every VaR model can be computed over the whole range from the
  trimmed returns, and fails otherwise. VaR models are computed on every
  date of the range, not on a schedule. Backtests run on the VaRs of the
  range.
- **R23** — The dates of the nodes upstream of the VaR models (e.g. an EWMA
  volatility feeding an FHS or a Gaussian model) follow from the VaR range
  and from the warm-up each node needs. Depending on the node, it starts on
  the first date it is needed, or as early as possible: an EWMA volatility's
  recursion starts on the first available date.
- **R24** — Nodes compute on time series, which keep track of the dates.
  Every node's time series output has the same dates and shape as the
  run's returns, throughout the graph (2026-10-26). Backtest results are
  not time series: they have one value per window.

### Simulation

- **R18** — Simulate returns easily, from a basic AR mean with an EWMA or a
  GARCH volatility. EWMA simulations can explode: that is accepted, and left
  to the user to watch for.

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
- **R20** — Model, backtest and run input names are Python identifiers
  (letters, digits and underscores, not starting with a digit). Reports
  can show friendlier names, given separately.
- **R9a** — Users specify models. The library builds the preprocessors those
  models need (e.g. volatility models, calibrations).
- **R12** — After a run, users can serialize its results: model outputs and
  preprocessor outputs.
- **R13** — Data that comes from outside a run (e.g. VaR forecasts for R3's
  first mode, or volatilities from the data service) is declared in the spec
  as a named input, and passed in by name at run time. A missing input is an
  error.

### Reporting

- **R19** — Visualize a run in a report: its node graph and charts of its
  results. A `Report` takes a run and the returns (and run inputs), and
  saves one self-contained HTML file that opens offline. The report has
  tabs: the graph, returns vs VaR, a backtest table, rolling backtests, and
  node outputs. Viewed as a saved file for now; inline in Jupyter and
  served by the data service (R17) may come later.

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
- **D6** — *Independent schedules.* For VaR models, superseded by R22: they
  are computed on every date of the run's range. Calibrations keep their
  step. Each node has its own window and step. A
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

- **D8** — *Simulator details.* Agreed in design, part of R18: the
  simulator returns the true means and volatilities next to the returns,
  dated like the models (D1); the shocks are drawn from a `Distribution`
  with zero mean and unit variance, Gaussian or Student-t for now; GARCH
  starts from its long-run variance (`alpha + beta < 1`), EWMA from a given
  starting volatility, the AR lags from the long-run mean, with an optional
  burn-in; the variance recursions run on the residuals `r - mu`.

Proposed for the context API (2026-10-26), see "Context API" in
`docs/open-points.md`. D9 is built (step 2); D10 and D11 are not yet:

- **D9** — *Nodes identified by value.* Two nodes of the same class with
  equal constructor arguments, input nodes included, are one node, wherever
  they were built. A node's name is not part of it. This replaces the two
  ways of D3: sharing no longer depends on passing one instance or on a
  `Builder`.
- **D10** — *Nodes bind to their context.* A node built inside a
  `with Context():` block belongs to that context, the innermost one of its
  thread or asyncio task, so concurrent requests in a service (R17) never
  mix their nodes. A node built outside any context belongs to none, and can
  still be the input of a node in any context.
- **D11** — *Inputs checked when a node is built.* A node's inputs are its
  constructor arguments that are nodes, and its `compute` takes one keyword
  argument per input, named after it. Building a node checks this, so a
  mismatch fails where the model is declared, not mid-run.

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
