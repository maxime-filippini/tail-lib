# Open points

Pending work and design questions, so a new session can pick them up. Read
`CLAUDE.md`, `docs/requirements.md` and `docs/architecture.md` first.

Working rules, from `CLAUDE.md` and the developer:

- One small change at a time, then stop for review.
- Propose design options before any new abstraction or public API.
- Check each proposal against the decisions recorded here and in
  `docs/architecture.md`, e.g. every backtest runs on every model (R4).
- Commit only when the developer says so. Commits have no co-author trailer.

State as of 2026-10-24: branch `feat/model-api`, not pushed, all checks
green (ty, ruff, 229 tests), nothing uncommitted.

## Next up

1. **Serializing results (R12)**, a design to propose first. See "Design
   questions" below.

## Decisions on finished work

Recorded so later changes don't undo them by accident.

### Rolling backtests (R15, R16)

- A backtest takes an optional `window`: `trailing` windows, or `last`, one
  window on the last dates (e.g. Kupiec on the last 250 days as of the last
  date). Named backtests let several windows coexist, and all the backtests
  of a model share its `Overshootings` (D3).
- Results have one value per window: every `KupiecResult` array has the
  shape (n_windows, *batch), and `end_dates` gives each window's last date.
  A whole-sample test is one window ending on the last date. Rejected:
  results as time series on the returns' dates (mostly `NaN` for a
  whole-sample test), and a separate rolling result type.
- Windows start on the first date where every series has a value (R16,
  `Windows(..., skip_leading_nan=True)`), so no window holds warm-up `NaN`s.
  Backtests and filtered historical simulation use it; other models don't
  yet.
- Rejected for "the last 250 days": reading `[-1]` of a rolling test
  (computes every window), and windows counted back from the last date (past
  windows move when a date is added, so results aren't reproducible, D4).
- Timing: the window of overshootings ending on `t` uses VaR up to `t - 1`
  and returns up to `t` (D1).

### `Last` windows for models

- Historical simulation (filtered or not), the decay calibration and the
  fitted EWMA take
  `Trailing | Last`. With `Last`, a model gives one value, on the last date,
  e.g. today's VaR without computing every window (R6, R7).
- Such a model can't be backtested: its only VaR is the forecast for a date
  after the data. `RunSpec` rejects a run with backtests and a model with a
  `last` window anywhere in its spec, naming the model and the path. So
  `last` models go in their own run, without backtests.
- Every window in a spec needs its `type` (`trailing` or `last`). The
  developer chose strict, as for R9a, over a missing `type` defaulting to
  `trailing`.

### FHS spec (R9)

`FilteredHistoricalSimulationsSpec` nests a `VolatilityFilterSpec` (`vol`,
`mean`) in its `filter` field, mirroring the node's `filter` argument.
Chosen by the developer over flat `vol` and `mean` fields on the FHS spec
(as in `GaussianSpec`), from which the builder would make the filter: that
breaks "a spec's fields mirror its node's arguments". Identical filter
specs give one filter node (`Builder.volatility_filter`, D3), and paths go
through it, e.g. `models.fhs.filter.vol.calibration.window`.

### JSON round trip (R9)

`RunSpec.model_dump_json()` parses back to an equal spec, and spells out
every default. Tested on specs holding every spec class.

### Runs from Python code (R17)

Python code (notebooks, scripts) builds nodes and passes them to `Run`.
Specs are the form that can be deserialized, for the JSON service. They can
be built in Python too, but only as a data format, with its validation,
to be turned into nodes (`Run.from_spec`, `Builder`): they hold no logic.
This replaces an earlier choice of specs as the one way in. Chosen by the
developer:

- Backtests are given to a run as functions from a model's `Overshootings`
  to a test node, e.g. `KupiecTest` itself, or
  `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`. ty checks
  lambdas; it accepts `functools.partial` without checking its arguments.
  ES backtests may later need the model too, not only its overshootings.
  Rejected: a settings class per backtest (duplicates its spec), and
  backtest nodes bound to each model by hand (loses "every backtest on
  every model", R4).
- Nodes list the nodes they read (`dependencies()`), so `Run` can find run
  inputs and `last` windows up front. It also gives names for R12. Rejected:
  no graph walk, which reports missing inputs mid-run and can't reject a
  backtested `last` model.
- `Run.from_spec(spec)` builds the nodes, then makes the same `Run`.
  `RunSpec` keeps its own `last`-window check, for errors with a spec path
  (R9b).
- Names are unique in a run's graph (developer's rule, against silent
  overriding): models and backtests are named by dict keys, and each run
  input name belongs to one node. Models reading the same input share its
  node; two different nodes with one input name are an error, even if they
  agree.
- Not now: a node-built run can't be written back as a spec, which saving
  what ran (R12) and reusing results (D4) may need later.

## Small open points

- Specs from HTTP (R17) are untrusted. To discuss: limits on what one spec
  can ask for (number of models, window lengths), and whether the service
  needs spec errors in a structured form (pydantic's `errors()` gives each
  path and message).
- The Python example in `docs/architecture.md` ("From Python") isn't tested.
- The code examples of the user guide (`docs/guide/`) were run by hand
  when written, but aren't tested. Options: a pytest that runs the
  `python` blocks, or `pymdownx.snippets` including tested files.
- The docs site is built in CI (strict, uploaded as the `docs-site`
  artifact), but not published yet (GitHub Pages or Cloudflare).
- The guide installs from the git repository: tail-lib isn't on PyPI.

- `NaN` in the middle of a window (e.g. a missing return). Proposal: keep
  counting the usable flags (`n_obs < length`).
- Maybe a `to_series()` on backtest results, to put window results on the
  returns' dates.
- Maybe a `load_run_spec(path)` helper (new public API, to propose if
  wanted). `RunSpec.model_validate_json(path.read_text())` works today.
- Integer checks are still written in each class: window length (in both
  `Trailing` and `Last`) and step, backcast window, number of dates and
  burn-in. A `check_at_least(name, value, minimum)` in `_checks.py` would
  write them once.
- `GaussianModel` rejects a zero or negative volatility, the scale of its
  distribution (developer's choice). `VolatilityFilter` and the FHS, which
  divide and rescale by a volatility, don't check it yet.
- `Windows.apply` holds all windows in memory at once. Chunk it for long
  series (R6).
- The tox matrix varies NumPy only. Consider pinning the lowest pydantic
  (2.13) in a test environment.
- The derived requirements D1–D7 are not yet confirmed by the developer.
- The branch hasn't been pushed, and there's no PR to `develop` yet.

## Design questions

### Serializing results (R12) and reusing them (D4)

`RunResult` holds only model VaRs and backtest results. R12 also asks for
preprocessor outputs (volatilities, calibrations). To decide:

- how to name them, e.g. by their path in the spec (`gauss_99.vol`);
- the file format for arrays;
- how to save backtest results, now that they have a windows axis and
  `end_dates`.

Frozen specs compare and hash by value, so D4 could key saved outputs by
spec, and the JSON dump of a spec records exactly what ran.

### Returns as a node (pinned by the developer)

Today every node reads `cache.returns`. Proposal: a `Returns` node with a
shared default instance `RETURNS`, taken by every node that reads the
returns. `ConstantVolatility` is the exception: it only needs the dates and
shape.

The gain is composability, not reuse:

- demeaned or portfolio returns (R5, R2), e.g. to filter back the
  volatilities of a simulation with an AR mean (R18);
- backtesting against a different P&L series.

Alternatives: B, returns as a named input (`run(inputs={"returns": ...})`),
which loses the reference series that inputs are aligned with; C, keep it
as it is. The developer asked to revisit this later. `VolatilityFilter`
has a spec, but only nested in the FHS spec (see "FHS spec" above).

Filtered historical simulation, first listed as a gain here, got its own
node instead: `FilteredHistoricalSimulationsModel` takes a `VolatilityFilter`
(`(r[t] - mean[t - 1]) / vol[t - 1]`, zero mean by default), and rescales the
quantile with the filter's mean and volatility. One filter can be shared by
several FHS models, e.g. on 500 and 750 dates.

### Simulator and estimation models (R18)

The shocks are a `Distribution` subclass (`distributions.py`), chosen by
the developer over a `Shocks` union, a `Protocol`, one class with a name
field, and wrapped SciPy distributions (loose types, R11). Distributions
give their quantiles too (`quantile`), which `GaussianModel` reads, so a
Student-t VaR, and ES, can reuse them.

Chosen by the developer (option A): the simulator's processes
(`ArProcess`, `EwmaProcess`, `GarchProcess`) are separate from the model
nodes, and share only the one-step recursions (`ewma_update`,
`garch_update`). Rejected (B): a `simulate` method on the volatility nodes,
whose parameters are nodes, and some of which (`FittedEwmaVolatility`,
`VolatilityInput`, a `Backcast` start) can't simulate.

For later (option C, noted by the developer): shared parameter objects,
e.g. `GarchParams(omega, alpha, beta)`, output by a future GARCH
calibration node and taken by the simulator, so a fit can be simulated
from. To revisit when GARCH estimation lands. The library has no GARCH
volatility model yet, nor an AR mean model.

The simulator's recursions run on the residuals `r - mu`, while the
volatility models run on the raw returns. A model only filters back the
simulated volatilities with a zero mean, until the volatility models can
read demeaned returns: see "Returns as a node".

### Spec shorthands and defaults (R9a)

Also an open question in the requirements. The developer agreed to start
strict, with every node spelled out, and kept it strict for window types.
Candidates: `"decay": 0.94` for a constant decay, and default preprocessors
for a model.

### Per-backtest model filter (declined for now)

Every backtest runs on every model (R4). A filter would let one run hold
both `last` models and backtests, or run different tests on different
models. Declined by the developer: separate runs cover the `last` case. If
it's needed later, the options were: (A) `"models": [...]` on the backtest
spec, which breaks "a spec's fields mirror its node's arguments" and the
sharing of identical backtests; (B) a run-level entry wrapping the test,
`{"test": {...}, "models": [...]}` (recommended for different tests on
different models); (C) one `backtested_models` list on the run. The
`last`-window check in `RunSpec` would then apply only to backtested models.

## Features not started

- **More backtests:** Christoffersen independence and conditional coverage,
  the Basel traffic light, ES backtests (open question). They should take
  the shared `Overshootings` node and an optional `window`, like Kupiec. See
  "Adding a node type" in `docs/architecture.md`.
- **Expected shortfall (R1).**
- **GARCH volatility and AR mean models:** estimation nodes using
  `garch_update`, and an AR mean model. See "Simulator and estimation
  models" for option C.
- **Benchmarking (R4):** ranking criteria are an open question.
- **Multivariate models (R2):** the axis convention is an open question.
