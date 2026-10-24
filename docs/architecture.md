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
- **Value checks** are written once, in `_checks.py` (e.g.
  `check_probability` for every confidence level, `check_unit_interval` for
  every decay factor), and called by each class that takes the value. Specs
  check value ranges too, for errors with a path (R9b).

## Modules

| Module | Contents |
|---|---|
| `time_series.py`, `calendar.py` | `TimeSeries`, and the calendar behind its date helpers (R8a). |
| `windows.py` | `Trailing` and `Last` window specs, and `Windows`: the windows over a series, `apply` and `to_series` (D5). With `skip_leading_nan=True`, the windows start on the first date where every series has a value (R16). |
| `graph.py` | `Node`, `Preprocessor`, `Cache`, `Pool`: the computation graph. |
| `models/mean.py` | `MeanModel`, `ZeroMean`, `ConstantMean`. |
| `models/ewma.py` | EWMA start specs (`Backcast`, `StartingVol`), the recursion (`ewma_update`, `ewma_variance`), decay nodes (`ConstantDecay`, `EwmaDecayCalibration`) and the decay fit. |
| `models/garch.py` | The GARCH(1, 1) recursion (`garch_update`), shared by the simulator and a future GARCH volatility model. |
| `models/volatility.py` | `VolatilityModel`, `VolatilityInput`, `ConstantVolatility`, `EwmaVolatility`, `FittedEwmaVolatility` (D7), `VolatilityFilter`. |
| `models/value_at_risk.py` | `ValueAtRiskModel`, `ValueAtRiskInput`, `HistoricalSimulationsModel` (D2), `FilteredHistoricalSimulationsModel`, `GaussianModel`. |
| `backtests/overshootings.py` | `overshootings` (function) and `Overshootings` (node). |
| `backtests/kupiec.py` | `kupiec_pof` (function), `KupiecResult`, `KupiecTest` (node). |
| `distributions.py` | `Distribution`, the base class of distributions with zero mean and unit variance, and `Gaussian`, `StudentT`. Each draws values (`draw`) and gives its quantiles (`quantile`): the simulator draws its shocks from them, and `GaussianModel` reads the Gaussian quantile. |
| `simulation.py` | `Simulator` and its processes `ArProcess`, `EwmaProcess`, `GarchProcess`; the output `Simulation` (R18). |
| `specs.py` | Pydantic specs mirroring each node, with value ranges (R9, R9b). |
| `build.py` | `Builder`: specs to nodes, one node per distinct spec (R9a, D3). |
| `run.py` | `RunSpec`, `Run` (from nodes, or from a spec with `Run.from_spec`), `BacktestFunction`, `RunResult`: models and backtests in one run. |

## User documentation

`zensical.toml` builds a [Zensical](https://zensical.org) site from `docs/`
(`uv run --group docs zensical serve`). `docs/guide/` is the user guide,
written by hand; `docs/reference/` is the API reference, generated from the
docstrings by mkdocstrings (`::: tail_lib.module` lines). The developer docs
(this file, the requirements, the open points) are under "Development". A
change to the public API, a spec field or a default also updates the guide:
`docs/guide/specs.md` has a table of every spec.

## The node graph (`graph.py`)

Every computation is a `Node[T]`: a preprocessor (a model input such as a
volatility), a VaR model, or a backtest.

- `evaluate(cache)` gathers the node's inputs, i.e. other nodes' outputs,
  with `cache.get(node)`, then calls `compute`.
- `compute(...)` is pure: it uses only its arguments and the node's own
  constructor settings (see `CLAUDE.md`).
- `dependencies()` lists the nodes `evaluate` reads, keyed by the
  constructor argument each comes from, e.g. `{"vol_model": ...,
  "mean_model": ...}`. So a graph can be walked without computing it: `Run`
  walks it to find its run inputs and `last` windows. The
  returns and the named run inputs are not nodes, so they are not listed.
  `tests/test_dependencies.py` checks, for one node of each library type,
  that it matches what `evaluate` asks the cache for.
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

Specs are the way in to a run that can be serialized (R9), e.g. for the
service receiving JSON (R17). Python code builds nodes instead, see "From
Python" below. The serialized format is Python dicts or JSON, parsed by
pydantic. YAML could later be a loader producing dicts.

1. **Parse.** `RunSpec.model_validate_json(text)` (or `model_validate(dict)`)
   gives frozen spec objects. In each slot, a `type` tag picks the spec class
   (a discriminated union per role: `ValueAtRiskSpec`, `VolatilitySpec`,
   ...). Field names, types and value ranges are all checked here, and an
   error gives the path to the faulty value, e.g.
   `vol.ewma.decay.value.list.1` (R9b). `Values[T]` is a single value or a
   non-empty list (one value per series). `spec.model_dump_json()` writes a
   spec back, with every default spelled out (e.g. `"step": 1`,
   `"window": null`), and parses back to an equal spec.
2. **Build.** `Run.from_spec(spec)` uses a `Builder`, which turns each spec
   into its node with a plain `match`, one typed method per role
   (`var_model`, `volatility`, `mean`, `constant_decay`, `calibration`,
   `volatility_filter`, `backtest`). It keeps a table from spec to node, and frozen specs compare
   by value, so identical specs, even written separately, give one node
   (D3). The builder also rejects an input name declared twice differently.
   It then makes a `Run` of the model nodes, with one backtest function per
   backtest spec: from a model's `Overshootings` to the test node
   (`Builder.backtest`).
3. **Walk the graph.** `Run(models, backtests)` walks each model's
   `dependencies()` once. It collects `run.input_names` from the input
   nodes, each name belonging to one node (two different nodes reading one
   name are an error, so none silently reads another's input) and, if there
   are backtests, rejects a model with a `last` window
   anywhere in its graph, giving the path through the graph, e.g.
   `models.today.vol_model.calibration.window`. Each distinct model node
   gets one `Overshootings`, shared by all its backtests.
4. **Check inputs.** `run.run(returns, inputs)` checks that every input in
   `run.input_names` is given before computing anything (R13). Inputs no
   node reads are ignored.
5. **Compute.** One `Cache`: the models' pool, then each model's pool of
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
    "fhs_99": {"type": "filtered_historical", "cf": 0.99,
               "filter": {"vol": {"type": "ewma",
                                  "decay": {"type": "constant",
                                            "value": 0.94}}},
               "window": {"type": "trailing", "length": 500}},
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
run = Run.from_spec(RunSpec.model_validate_json(text))
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
in `specs.py`). `Run` checks the same on its nodes, for runs built in
Python.

Every backtest runs on every model (R4). Backtests are named, so two
versions of one test can sit side by side.

## From Python

Notebooks and scripts (R17) build nodes and give them to `Run`. A service
receiving JSON over HTTP parses a `RunSpec` and calls `Run.from_spec`,
which builds the nodes with a `Builder`, then makes the same `Run`. Both
then go through the same graph walk and the same computation.

```python
vol = EwmaVolatility(ConstantDecay(0.94))
vol_filter = VolatilityFilter(vol)
models = {
    f"fhs_{n}": FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=n)
    )
    for n in (500, 750)
} | {"gauss": GaussianModel(cf=0.99, vol_model=vol)}
run = Run(
    models,
    backtests={
        "kupiec": KupiecTest,
        "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
    },
)
result = run.run(returns)
```

- Sharing is by instance (D3): `vol` and `vol_filter` above are computed
  once, for all the models that read them. From specs, the `Builder` gives
  identical specs one node, to the same effect.
- Each backtest is a function from a model's `Overshootings` to the test
  node: the class itself (`KupiecTest`) for its defaults, or a lambda for
  its settings. Every backtest runs on every model (R4), and the backtests
  of a model share its `Overshootings`. ty checks lambdas; it accepts
  `functools.partial` without checking its arguments.
- Values are checked when each node is built, e.g. `ConstantDecay(1.5)`
  raises. ty reports a missing argument or a node in the wrong slot, e.g. a
  mean model given as `vol_model`.
- `Run` walks the graph when it is built (step 3 of "From a spec to
  results"), so missing inputs are reported before computing, and backtests
  on a model with a `last` window are rejected, with the path to the window.
- A run built from nodes has no spec, so it can't be written back as JSON.

Specs can be built in Python too, but only as data: a validated
description of a run, turned into nodes by `Run.from_spec`, e.g. to record
what ran with `spec.model_dump_json()`. They hold no logic of their own;
the computation is in the nodes. The class picks the spec, so the `type`
tags can be left out, and a list of values (one per series) is a tuple.
Values are checked when each spec is built, so an error's path starts at
that spec, e.g. `value.list.1` for `ConstantDecaySpec(value=(0.94, 1.5))`.

Nodes can also be computed without a run, with `Pool` or `node.run` (see
"The node graph"), e.g. in tests: then each backtest is wired to its model
by hand, and missing inputs are only reported when read.

`examples/backtest_hs_fhs.py` runs HS, FHS and a Gaussian model built this
way, with a rolling Kupiec test.

## Simulating returns (R18)

`simulation.py` is outside the node graph: a simulation reads no returns, and
its parameters are set by hand, so its parts are plain frozen dataclasses.
The shocks are any `Distribution` (`distributions.py`).

```python
sim = Simulator(
    mean=ArProcess(const=0.0, ar=(0.05,)),
    vol=GarchProcess(omega=2e-6, alpha=0.08, beta=0.90),
    shocks=StudentT(df=5),
).simulate(2520, first_date=date(2015, 1, 1), batch_shape=(100,), rng=42)
sim.returns, sim.means, sim.vols  # TimeSeries of shape (2520, 100)
```

`GarchProcess.from_long_run_variance(alpha=..., beta=...,
long_run_variance=...)` builds the same process from its long-run variance,
with `omega = long_run_variance * (1 - alpha - beta)`.

`Simulator.simulate` is one loop over the dates, vectorized over the batch.
On date `t`: `eps[t] = sigma[t] * z[t]`, `r[t] = mu[t] + eps[t]`, then the
volatility process updates the variance with `eps[t]` and the AR process
updates the mean with `r[t]`. `means` and `vols` hold, on `t`, the mean and
volatility of `t + 1` (D1, D8), so they compare directly with a model's
output.

The processes call the same one-step recursions as the models
(`ewma_update`, `garch_update`), so the formulas are written once. A test
checks that `EwmaVolatility` with the true decay and starting volatility
gives back the simulated volatilities exactly. This only holds with a zero
mean: the simulator's recursion runs on residuals, `EwmaVolatility` on the
raw returns (see "Returns as a node" in `docs/open-points.md`).

## Adding a node type

1. Write the node: `evaluate` gathers inputs from the cache, `compute` is
   pure, `dependencies` lists the nodes `evaluate` reads, and the docstring
   states the formula, the timing and a reference. Add one instance to
   `_one_node_of_each_type` in `tests/test_dependencies.py`: the test fails
   until you do.
2. Add its spec to `specs.py`, with value ranges matching the node's checks,
   and add the spec to its role's union.
3. Add a `case` to the role's `_build_...` method in `build.py`. Because of
   `assert_never`, ty reports a spec type with no case.
4. Test that the spec builds the same numbers as the node written in Python.

For a new role (e.g. a second backtest), turn the role's alias into a tagged
union (`BacktestSpec`, and `BacktestResult` in `build.py`).
