# Open points

Pending work and design questions, so a new session can pick them up. Read
`CLAUDE.md`, `docs/requirements.md` and `docs/architecture.md` first.

Working rules, from `CLAUDE.md` and the developer:

- One small change at a time, then stop for review.
- Propose design options before any new abstraction or public API.
- Check each proposal against the decisions recorded here and in
  `docs/architecture.md`, e.g. every backtest runs on every model (R4).
- Commit only when the developer says so. Commits have no co-author trailer.

State as of 2026-10-23: branch `feat/model-api`, not pushed, all checks
green (ty, ruff, 142 tests), everything committed.

## Next up

1. **Literature reference for historical simulation.**
   `HistoricalSimulationsModel`'s docstring has none (`CLAUDE.md` asks for
   one), e.g. Jorion, *Value at Risk*.
2. **Serializing results (R12)**, a design to propose first. See "Design
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
  Backtests use it; models don't yet.
- Rejected for "the last 250 days": reading `[-1]` of a rolling test
  (computes every window), and windows counted back from the last date (past
  windows move when a date is added, so results aren't reproducible, D4).
- Timing: the window of overshootings ending on `t` uses VaR up to `t - 1`
  and returns up to `t` (D1).

### `Last` windows for models

- Historical simulation, the decay calibration and the fitted EWMA take
  `Trailing | Last`. With `Last`, a model gives one value, on the last date,
  e.g. today's VaR without computing every window (R6, R7).
- Such a model can't be backtested: its only VaR is the forecast for a date
  after the data. `RunSpec` rejects a run with backtests and a model with a
  `last` window anywhere in its spec, naming the model and the path. So
  `last` models go in their own run, without backtests.
- Every window in a spec needs its `type` (`trailing` or `last`). The
  developer chose strict, as for R9a, over a missing `type` defaulting to
  `trailing`.

### JSON round trip (R9)

`RunSpec.model_dump_json()` parses back to an equal spec, and spells out
every default. Tested on specs holding every spec class.

### Runs from Python code (R17)

Notebooks and scripts build spec objects in Python and call `Run`, like the
JSON service: specs are the one way in to a run. Chosen by the developer
over the alternatives, kept here in case a model defined outside the
library has to join a run:

- B: `Run` takes built nodes, and `Run.from_spec` builds them first.
  Backtests stay templates (specs, or a function of the `Overshootings`),
  and the input and `last`-window checks need nodes to list the nodes they
  read. Node-built runs can't be saved as specs.
- C: B, plus nodes that give back their spec, so node-built runs can be
  saved (R12, D4).

## Small open points

- Specs from HTTP (R17) are untrusted. To discuss: limits on what one spec
  can ask for (number of models, window lengths), and whether the service
  needs spec errors in a structured form (pydantic's `errors()` gives each
  path and message).
- The Python example in `docs/architecture.md` ("From Python") isn't tested.

- `NaN` in the middle of a window (e.g. a missing return). Proposal: keep
  counting the usable flags (`n_obs < length`).
- Maybe a `to_series()` on backtest results, to put window results on the
  returns' dates.
- `Builder.backtest` used directly in Python doesn't reject a model with a
  `last` window; only `RunSpec` does.
- Maybe a `load_run_spec(path)` helper (new public API, to propose if
  wanted). `RunSpec.model_validate_json(path.read_text())` works today.
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

- filtered HS as HS on `VolatilityFilter` output, whose first window then
  starts after the volatility warm-up (R16);
- demeaned or portfolio returns (R5, R2);
- backtesting against a different P&L series.

Alternatives: B, returns as a named input (`run(inputs={"returns": ...})`),
which loses the reference series that inputs are aligned with; C, keep it
as it is. The developer asked to revisit this later. `VolatilityFilter` has
no spec until this is settled.

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
- **Benchmarking (R4):** ranking criteria are an open question.
- **Multivariate models (R2):** the axis convention is an open question.
