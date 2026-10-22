# Open points

Pending work and design questions, so a new session can pick them up. Read
`CLAUDE.md`, `docs/requirements.md` and `docs/architecture.md` first.

Working rules, from `CLAUDE.md` and the developer:

- One small change at a time, then stop for review.
- Propose design options before any new abstraction or public API.
- Commit only when the developer says so. Commits have no co-author trailer.

State as of 2026-10-22: branch `feat/model-api`, not pushed, all checks
green (ty, ruff, 115 tests).

## Next up

### 1. Rolling backtests (R15)

The developer wants backtests on their own windows, e.g. Kupiec on rolling
250 and 500 days, side by side in one run:

```json
"backtests": {
  "kupiec_250": {"type": "kupiec", "window": {"length": 250}},
  "kupiec_500": {"type": "kupiec", "window": {"length": 500}}
}
```

Agreed so far: the setting goes on the backtest spec, and named backtests
let both versions coexist. Both should share the model's `Overshootings`.

To design with the developer: what a rolling test returns. Today
`KupiecResult` has one value per series. A rolling test has one per window.
Options to present:

- `KupiecResult` fields become time series on the window end dates, held
  until the next window like `Windows.to_series` (D5). A whole-sample test
  is then the special case of one window.
- A separate rolling result type next to `KupiecResult`. That's simpler per
  type, but `BacktestResult` becomes a union.

Check the timing: the window of overshootings ending on `t` uses VaR up to
`t - 1` and returns up to `t` (D1).

### 2. JSON round trip (R9, spec increment 3)

Parsing already works. What's left is a test that `RunSpec.model_dump_json()`
parses back to an equal spec (lists come back as tuples), and maybe a small
`load_run_spec(path)` helper. Note: `RunSpec` holds dicts, so it is frozen
but not hashable, which is fine since only sub-specs are table keys.

## Design questions, not started

### Returns as a node (pinned by the developer)

Today every node reads `cache.returns`. Proposal: a `Returns` node with a
shared default instance `RETURNS`, taken by every node that reads the
returns. `ConstantVolatility` is the exception: it only needs the dates and
shape.

The gain is composability, not reuse:

- filtered HS as HS on `VolatilityFilter` output;
- demeaned or portfolio returns (R5, R2);
- backtesting against a different P&L series.

Alternatives: B, returns as a named input (`run(inputs={"returns": ...})`),
which loses the reference series that inputs are aligned with; C, keep it
as it is. The developer asked to revisit this later. `VolatilityFilter` has
no spec until this is settled.

### Serializing results (R12) and reusing them (D4)

`RunResult` holds only model VaRs and backtest results. R12 also asks for
preprocessor outputs (volatilities, calibrations). To decide:

- how to name them, e.g. by their path in the spec (`gauss_99.vol`);
- the file format for arrays.

Frozen specs compare and hash by value, so D4 could key saved outputs by
spec.

### Spec shorthands and defaults (R9a)

Also an open question in the requirements. The developer agreed to start
strict, with every node spelled out. Candidates: `"decay": 0.94` for a
constant decay, and default preprocessors for a model.

### Per-backtest model filter

Today every backtest runs on every model. A backtest spec could later take
`"models": [...]`. Leave it out until it's needed.

## Features not started

- **More backtests:** Christoffersen independence and conditional coverage,
  the Basel traffic light, ES backtests (open question). They should take
  the shared `Overshootings` node. See "Adding a node type" in
  `docs/architecture.md`.
- **Expected shortfall (R1).**
- **Benchmarking (R4):** ranking criteria are an open question.
- **Multivariate models (R2):** the axis convention is an open question.

## Smaller items

- `HistoricalSimulationsModel`'s docstring has no literature reference
  (`CLAUDE.md` asks for one), e.g. Jorion, *Value at Risk*.
- `Windows.apply` holds all windows in memory at once. Chunk it for long
  series (R6).
- The tox matrix varies NumPy only. Consider pinning the lowest pydantic
  (2.13) in a test environment.
- The derived requirements D1–D7 are not yet confirmed by the developer.
- The branch hasn't been pushed, and there's no PR to `develop` yet.
