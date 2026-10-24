# Architecture

How tail-lib is organized, and how a call flows from a spec to the numbers
(R10). Requirement IDs refer to `docs/requirements.md`.

## Conventions

- **Arrays** (R8): time on the first axis, then any number of independent
  series. A `TimeSeries` pairs the values with their dates.
- **Timing** (D1): a value on date `t` uses data up to and including `t`.
  The VaR on `t` is the forecast for `t + 1`.
- **Sign** (R14): VaR is a return quantile, negative for a loss. The return
  of date `t` *overshoots* the VaR when `r[t] < VaR[t - 1]`. The code and
  docs say "overshootings", never "exceedances".
- **Confidence level**: models take `cf`. In a backtest, the model's level
  is called `cf_var`.

## Modules

| Module | Contents |
|---|---|
| `time_series.py`, `calendar.py` | `TimeSeries`, and the calendar behind its date helpers (R8a). |
| `windows.py` | `Trailing` and `Last` window specs, and `Windows`: the windows over a series, `apply` and `to_series` (D5). With `skip_leading_nan=True`, the windows start on the first date where every series has a value (R16). |
| `graph.py` | `Node`, `Preprocessor`, `Cache`, `Pool`: the computation graph. |
| `models/mean.py` | `MeanModel`, `ZeroMean`, `ConstantMean`. |
| `models/ewma.py` | EWMA start specs (`Backcast`, `StartingVol`), the recursion (`ewma_update`, `ewma_variance`), decay nodes (`ConstantDecay`, `EwmaDecayCalibration`) and the decay fit. |
| `models/volatility.py` | `VolatilityModel`, `VolatilityInput`, `ConstantVolatility`, `EwmaVolatility`, `FittedEwmaVolatility` (D7), `VolatilityFilter`. |
| `models/value_at_risk.py` | `ValueAtRiskModel`, `ValueAtRiskInput`, `HistoricalSimulationsModel` (D2), `GaussianModel`. |
| `backtests/overshootings.py` | `overshootings` (function) and `Overshootings` (node). |
| `backtests/kupiec.py` | `kupiec_pof` (function), `KupiecResult`, `KupiecTest` (node). |
| `specs.py` | Pydantic specs mirroring each node, with value ranges (R9, R9b). |
| `build.py` | `Builder`: specs to nodes, one node per distinct spec (R9a, D3). |
| `run.py` | `RunSpec`, `Run`, `RunResult`: models and backtests in one run. |

## The node graph (`graph.py`)

Every computation is a `Node[T]`: a preprocessor (a model input such as a
volatility), a VaR model, or a backtest.

- `evaluate(cache)` gathers the node's inputs, i.e. other nodes' outputs,
  with `cache.get(node)`, then calls `compute`.
- `compute(...)` is pure: it uses only its arguments and the node's own
  constructor settings (see `CLAUDE.md`).
- `Cache(returns, inputs)` belongs to one returns series. It computes each
  node at most once, identified by instance, and detects cycles. It also
  holds the named run inputs (R13): `cache.input(name)` checks the input is
  aligned with the returns, and raises `KeyError` if it is missing.
- `Pool(nodes)` computes named nodes on one cache. Pools evaluated on the
  same cache share their common nodes.
- `node.run(returns, inputs)` and `pool.run(returns, inputs)` do the same with
  a fresh cache.

Estimable parameters (a volatility, a decay) are input nodes. Every variant
of such an input is a node, e.g. `ConstantDecay` next to
`EwmaDecayCalibration`. Settings always set by hand (`cf`, windows, fit
bounds) are constructor arguments.

Sharing in Python is explicit: pass one instance to several nodes. For
example, backtests take an `Overshootings` node, not a model, so all the
backtests of a model can share one:

```python
ovs = Overshootings(model)
Pool({"kupiec": KupiecTest(ovs)}).run(returns)
```

## From a spec to results

Specs are the way in to a run, both serialized (R9) and from Python code
(R17, see "From Python" below). The serialized format is Python dicts or
JSON, parsed by pydantic. YAML could later be a loader producing dicts.

1. **Parse.** `RunSpec.model_validate_json(text)` (or `model_validate(dict)`)
   gives frozen spec objects. In each slot, a `type` tag picks the spec class
   (a discriminated union per role: `ValueAtRiskSpec`, `VolatilitySpec`,
   ...). Field names, types and value ranges are all checked here, and an
   error gives the path to the faulty value, e.g.
   `vol.ewma.decay.value.list.1` (R9b). `Values[T]` is a single value or a
   non-empty list (one value per series). `spec.model_dump_json()` writes a
   spec back, with every default spelled out (e.g. `"step": 1`,
   `"window": null`), and parses back to an equal spec.
2. **Build.** `Run(spec)` uses a `Builder`, which turns each spec into its
   node with a plain `match`, one typed method per role (`var_model`,
   `volatility`, `mean`, `constant_decay`, `calibration`, `overshootings`,
   `backtest`). It keeps a table from spec to node, and frozen specs compare
   by value, so identical specs, even written separately, give one node
   (D3). Each model's backtests share one `Overshootings`. The builder also
   records the run inputs that input specs declare, and rejects a name
   declared twice differently.
3. **Check inputs.** `run.run(returns, inputs)` checks that every input in
   `run.input_names` is given before computing anything (R13). Inputs no
   node reads are ignored.
4. **Compute.** One `Cache`: the models' pool, then each model's pool of
   backtests. The result is a `RunResult`: `models[name]` is the VaR series,
   `backtests[model][backtest]` is the backtest result.

Example:

```json
{
  "models": {
    "gauss_99": {"type": "gaussian", "cf": 0.99,
                 "vol": {"type": "fitted_ewma",
                         "calibration": {"window": {"type": "trailing",
                                                    "length": 250}}}},
    "hs_99": {"type": "historical", "cf": 0.99,
              "window": {"type": "trailing", "length": 250}},
    "ext_99": {"type": "input", "name": "var_ext", "cf": 0.99}
  },
  "backtests": {
    "kupiec": {"type": "kupiec"},
    "kupiec_250": {"type": "kupiec",
                   "window": {"type": "trailing", "length": 250}},
    "kupiec_last": {"type": "kupiec",
                    "window": {"type": "last", "length": 250}}
  }
}
```

```python
run = Run(RunSpec.model_validate_json(text))
result = run.run(returns, inputs={"var_ext": precomputed_var})
result.backtests["hs_99"]["kupiec"].p_value[0]
```

A backtest result has one value per window and series: its arrays have the
shape (n_windows, *batch), and `end_dates` gives the date each window's
result belongs to, its last date. A test on the whole sample is one window,
ending on the last date, hence `p_value[0]`. With a `window`, the test runs
on trailing windows of the overshootings, which start on the first flag
(R15, R16): `kupiec_250` has one result per date from 250 days after it.
A `last` window is one window on the last dates: `kupiec_last` tests the
last 250 days only, as of the last date. Every window, a model's or a
backtest's, is `trailing` or `last`, and needs its `type`. A model with a
`last` window, or with an input that has one, gives one value, on the last
date, e.g. today's VaR without computing every window. It can't be
backtested: that value is the forecast for a date after the data. So
`RunSpec` rejects a run with backtests and such a model, giving the path to
the window, e.g. `models.today.vol.calibration.window` (`find_last_window`
in `specs.py`).

Every backtest runs on every model (R4). Backtests are named, so two
versions of one test can sit side by side.

## From Python

Notebooks and scripts (R17) build the same spec objects in code, then call
`Run`, the same entry point as JSON. A service receiving JSON over HTTP
calls `RunSpec.model_validate_json` instead. Both go through the same checks
and the same build.

```python
spec = RunSpec(
    models={
        f"hs_{n}": HistoricalSimulationsSpec(cf=0.99, window=TrailingSpec(length=n))
        for n in (250, 500)
    },
    backtests={"kupiec": KupiecSpec()},
)
result = Run(spec).run(returns)
```

- The class picks the spec, so the `type` tags can be left out. A list of
  values (one per series) is a tuple.
- Values are checked when each spec is built, so an error's path starts at
  that spec, e.g. `value.list.1` for `ConstantDecaySpec(value=(0.94, 1.5))`.
- ty reports a missing field or a spec in the wrong slot, e.g. a window
  given as `vol`. Value ranges are only checked at run time.
- Identical specs share one node (D3), and `spec.model_dump_json()` records
  what ran.

Nodes can also be built directly and computed with `Pool` or `node.run`
(see "The node graph"), e.g. in tests. That path has no `Run`: the caller
wires each backtest to its model, missing inputs are only reported when
read, and there is no spec to save. Runs take specs only; a `Run` built
from nodes was considered and set aside until a model defined outside the
library needs to join a run.

## Adding a node type

1. Write the node: `evaluate` gathers inputs from the cache, `compute` is
   pure, and the docstring states the formula, the timing and a reference.
2. Add its spec to `specs.py`, with value ranges matching the node's checks,
   and add the spec to its role's union.
3. Add a `case` to the role's `_build_...` method in `build.py`. Because of
   `assert_never`, ty reports a spec type with no case.
4. Test that the spec builds the same numbers as the node written in Python.

For a new role (e.g. a second backtest), turn the role's alias into a tagged
union (`BacktestSpec`, and `BacktestResult` in `build.py`).
