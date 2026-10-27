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
- **Names** (R20): model, backtest and run input names are Python
  identifiers, checked by `check_name` in `Run`, in `RunInput`, and in
  specs (`Name`). So the dotted paths through a run's graph, e.g.
  `models.gauss.vol`, are unambiguous. A model and a backtest can't
  share a name. Display names are `labels` on `Run` and `RunSpec`, checked
  once by `_check_names_and_labels` in `run.py`; `run.labels` fills in
  every name, unlabelled ones as they are.

## Modules

| Module | Contents |
|---|---|
| `time_series.py`, `calendar.py` | `TimeSeries`, and the calendar behind its date helpers (R8a). |
| `windows.py` | `Trailing` and `Last` window specs, and `Windows`: the windows over a series, `apply` and `to_series` (D5). With `skip_leading_nan=True`, the windows start on the first date where every series has a value (R16). |
| `graph.py` | `Node`, `Preprocessor`, `Backtest`, `RunInput`, `Cache`, `Pool`: the computation graph. |
| `models/mean.py` | `MeanModel`, `ZeroMean`, `ConstantMean`. |
| `models/ewma.py` | EWMA start specs (`Backcast`, `StartingVol`), the recursion (`ewma_update`, `ewma_variance`), decay nodes (`ConstantDecay`, `EwmaDecayCalibration`), which output `DecayFactors` (the decay on each date, its refit dates and the first date of each fit), and the decay fit. |
| `models/garch.py` | The GARCH(1, 1) recursion (`garch_update`), shared by the simulator and a future GARCH volatility model. |
| `models/volatility.py` | `VolatilityModel`, `VolatilityInput`, `ConstantVolatility`, `EwmaVolatility` (constant or calibrated decay, D7), `VolatilityFilter`. |
| `models/value_at_risk.py` | `ValueAtRiskModel`, `ValueAtRiskInput`, `HistoricalSimulationsModel` (D2), `FilteredHistoricalSimulationsModel`, `GaussianModel`. |
| `backtests/overshootings.py` | `overshootings` (function) and `Overshootings` (node). |
| `backtests/kupiec.py` | `kupiec_pof` (function), `KupiecResult`, `KupiecTest` (node). |
| `report/data.py` | `ReportData`, the JSON contract with the report page, and `build_report_data` (R19). |
| `report/report.py`, `report/index.html` | `Report`: computes a run and saves its data in the report page, package data built from `report-app/` (R19). |
| `distributions.py` | `Distribution`, the base class of distributions with zero mean and unit variance, and `Gaussian`, `StudentT`. Each draws values (`draw`) and gives its quantiles (`quantile`): the simulator draws its shocks from them, and `GaussianModel` reads the Gaussian quantile. |
| `simulation.py` | `Simulator` and its processes `ArProcess`, `EwmaProcess`, `GarchProcess`; the output `Simulation` (R18). |
| `specs.py` | Pydantic specs mirroring each node, with value ranges (R9, R9b). |
| `build.py` | Plain functions from specs to nodes (R9a), shared by value (D9). The legacy `Builder` stays until step 5c-3. |
| `run.py` | `RunSpec`, `Run` (from nodes, or from a spec with `Run.from_spec`), `BacktestFunction`, `RunResult`: models and backtests in one run. |
| `context.py` | `Context`, `Results`: models declared in a `with` block or built by `Context.from_spec`, their backtests, and their results (D10). Next to `Run` until step 5c-3 of the migration. |

## User documentation

`zensical.toml` builds a [Zensical](https://zensical.org) site from `docs/`
(`uv run --group docs zensical serve`). `docs/guide/` is the user guide,
written by hand; `docs/reference/` is the API reference, generated from the
docstrings by mkdocstrings (`::: tail_lib.module` lines). The developer docs
(this file, the requirements, the open points) are under "Development". A
change to the public API, a spec field or a default also updates the guide:
`docs/guide/specs.md` has a table of every spec.

## The node graph (`graph.py`)

This section, "From a spec to results" and "From Python" describe the code
as it is. A migration to a context API is under way, step by step: nodes
identified by value (done), a pure `compute` with no `evaluate` (done),
and a `Context` in place of `Run`, `Cache` and `Pool`. See "Context API" in
`docs/open-points.md`.

Every computation is a `Node[T]`: a preprocessor (a model input such as a
volatility, a `Preprocessor`), a decay model (a `DecayModel`, giving
`DecayFactors`), a volatility filter (a `VolatilityFilter`, giving
`FilteredReturns`), a VaR model (a `ValueAtRiskModel`), or a backtest (a
`Backtest`). Nodes are frozen
dataclasses,
identified by value (D9): two nodes of the same class with equal fields,
their input nodes included, are one node, e.g. two `ZeroMean()` built
separately. The fields are the constructor arguments. Arrays given to a node
(a value per series) are stored as nested tuples (`frozen_floats`), so they
can be hashed; `np.asarray` gives the array back.

- Any node takes an optional keyword `name`, a Python identifier (R20),
  `None` by default. It is not part of the node's identity
  (`compare=False`), nor a setting, nor in its `repr`: two equal nodes with
  different names are one node, computed once. A context (step 5 of the
  migration) will register its VaR models by name.

- `compute(returns, /, **inputs)` is pure: it uses only its arguments and
  the node's own settings (see `CLAUDE.md`). After the returns, it takes one
  keyword-only argument per dependency, named after its field, holding that
  dependency's output, e.g. `GaussianModel.compute(returns, /, *, vol,
  mean)`. Building a node checks this signature (D11, `_check_compute`), so
  a mismatch fails where the node is declared; a node's `__post_init__`
  calls `super().__post_init__()` for it.
- Run inputs (R13) are `RunInput` nodes, `VolatilityInput` and
  `ValueAtRiskInput`, each with a `name`, checked to be a Python
  identifier. The cache never computes them: their output is the run input
  of their name. Their `compute` refuses to run. Their name is both the
  node's name and the input it reads, and, unlike other nodes' names, it is
  part of their identity and a setting. `RunInput` redefines the field
  (required, positional), so its subclasses list it first among their
  bases: a dataclass takes its fields from its bases in reverse order, and
  the model base class would otherwise bring back `Node.name`.
- `dependencies()` lists the nodes whose outputs `compute` takes: the fields holding
  nodes, keyed by field, e.g. `{"vol": ..., "mean": ...}`. It
  is written once, in `Node`. So a graph can be walked without computing
  it: `Run` walks it to find its run inputs and `last` windows. The returns
  and the named run inputs are not nodes, so they are not listed. No node
  overrides it: a node needing more than its inputs' outputs gets it inside
  them, e.g. the FHS reads the mean and volatility from its filter's
  `FilteredReturns`. `tests/test_dependencies.py` checks, for one node of
  each library type, that it matches what the cache reads for it.
- `settings()` lists the node's other fields, with the values the node
  uses, defaults included, e.g. `{"cf": 0.99, "window": Trailing(250),
  "interpolation": "linear"}`. It is also written once, in `Node`. The
  values stay Python objects (windows, EWMA starts, tuples), typed by the
  `Setting` union; turning them into JSON is the report's job (R19). With
  `dependencies()`, it covers every constructor argument. A field the node
  sets itself (`init=False`), e.g. the `cf_var` a Kupiec test copies from
  its overshootings, is neither.
- `Cache(returns, inputs)` belongs to one returns series. `cache.get(node)`
  computes each node at most once, identified by value: it gets the
  output of each of the node's `dependencies()`, then calls
  `node.compute(returns, **outputs)`, by field name. It detects cycles. It
  also holds the named run inputs (R13): `cache.input(name)` checks the
  input is aligned with the returns, and raises `KeyError` if it is
  missing. `cache.get` gives a `RunInput` node `cache.input(node.name)`
  instead of computing it.
- `Pool(nodes)` computes named nodes on one cache. Pools evaluated on the
  same cache share their common nodes.
- `node.run(returns, inputs)` and `pool.run(returns, inputs)` do the same with
  a fresh cache.

Estimable parameters (a volatility, a decay) are input nodes. Every variant
of such an input is a node, e.g. `ConstantDecay` next to
`EwmaDecayCalibration`. Settings always set by hand (`cf`, windows, fit
bounds) are constructor arguments.

Sharing follows from value identity: equal nodes are computed once, however
they were built. For example, backtests take an `Overshootings` node, not a
model, so all the backtests of a model share one:

```python
backtests = {
    "kupiec": KupiecTest(Overshootings(model)),
    "kupiec_250": KupiecTest(Overshootings(model), window=Trailing(length=250)),
}
Pool(backtests).run(returns)  # the overshootings of `model` are computed once
```

## Contexts (`context.py`)

The context API's replacement for `Run`, built next to it (step 5a of the
migration). The report, the guide, the example, the regression test and the
node tests use it. `Context.from_spec` also builds one from a spec; `Run`
stays until step 5c-3 removes the old API:

```python
with Context(backtests={"kupiec": KupiecTest}, labels={"gauss": "Gaussian"}) as ctx:
    vol = EwmaVolatility(ConstantDecay(0.94))
    GaussianModel(cf=0.99, vol=vol, name="gauss")
    HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250), name="hs")

results = ctx.run(returns, inputs)
results["gauss"], results[vol], results.backtests["gauss"]["kupiec"]
```

- **Binding (D10).** `graph.current_context` is a `ContextVar` holding the
  innermost open context of the thread or asyncio task. `__enter__` sets
  it, `__exit__` resets it. `Node.__post_init__` ends by calling
  `context.bind(node)` when one is open, so a node binds once its checks
  pass. The variable lives in `graph.py` because `context.py` imports
  `Node`.
- **`bind`** keeps every distinct node, by value, in the order bound. A
  value-at-risk model needs a `name` (else `ValueError`, raised where it is
  built). Names are unique: two different nodes can't share one, and no
  node can take a backtest's name; an equal node can take its name again.
  Named models are the context's `models`.
- **`graph()`** checks the names and labels, the `last` windows when there
  are backtests, and that each run input name belongs to one node, then
  returns a `RunGraph`, as `Run.graph` does. Roots: each model
  (`models.gauss`), each backtest of each model
  (`backtests.gauss.kupiec`), then the bound nodes not reached yet, latest
  first (`nodes.<name>`, or `nodes.<class>`, `nodes.<class>_2`, ... without
  a name). The backtest nodes are built from the backtest functions with
  the context variable unset, so they never bind.
- **`run(returns, inputs)`** reports missing inputs before computing, then
  computes each node once, in one loop over the graph ordered inputs first
  (`_inputs_first`): a `RunInput` gets its input, checked against the
  returns; any other node gets `compute(returns, **outputs)` by field name.
  After each node but a backtest, `_check_output` checks its time series
  (a `TimeSeries`, a `DecayFactors`' `values`, `FilteredReturns`'
  `standardized`) has the dates and shape of the returns (R24), and raises
  `TypeError` for an output holding none. It doesn't use the `Cache`.
- **`Results`** holds every node's output (`outputs`), the named nodes
  (`named`), `models` (name to VaR) and `backtests` (model, then test, to
  result). `results[node]` or `results["name"]` reads one output.

The node tests use `tests.helpers.run_node(node, returns, inputs)`, which
binds a node to its own context and reads its output from `Results`. An
unnamed VaR model is bound through a named copy: names aren't part of value
identity (D9), so the original node still reads the result. Shared-model
and backtest tests declare their nodes in one context. Tests specifically
covering `Run`, `Builder`, `Pool` and `Cache`, including the mutable-node
cycle test, stay until those classes are removed (step 5c-3).

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
2. **Build.** `Context.from_spec(spec)` first rejects an input declared
   twice differently (`build._check_input_specs`), then opens a context
   with the spec's labels and backtest functions. Plain typed functions in
   `build.py` turn each spec into its node with a `match`, recursively
   building its inputs. Model keys become node names. A VaR input has one
   name, so its model key must match its input name; a mismatch is a
   `ValueError` naming `models.<key>`. Equal nodes share computation by
   value (D9): there are no tables of built nodes in this path. Each
   backtest function binds its own spec with `partial(_build_backtest,
   backtest_spec)`, and later takes a model's `Overshootings`.
3. **Walk the graph.** Before returning, `Context.from_spec` closes the
   block and calls `ctx.graph()` to check the names, labels, inputs and
   `last` windows. Each model gets one `Overshootings`, shared by its
   backtests. The `RunGraph` lists every distinct node with all its paths,
   and one edge per dependency (see "Contexts"). Construction computes
   nothing, and restores any surrounding context, even on error (D10).
4. **Check inputs.** `ctx.run(returns, inputs)` checks every input named
   by the graph is given before computing anything (R13). Inputs no node
   reads are ignored.
5. **Compute.** The context computes every node once, inputs first (see
   "Contexts"). The `Results` holds every output, read by node or name,
   with `models[name]` for the VaR and `backtests[model][test]` for each
   backtest result. A report reads these results without computing again.

The legacy `Run.from_spec` still uses `Builder` and its sharing tables;
they are removed together in step 5c-3.

Example:

```json
{
  "models": {
    "gauss_99": {"type": "gaussian", "cf": 0.99,
                 "vol": {"type": "ewma",
                         "decay": {"type": "ewma_calibration",
                                   "window": {"type": "trailing",
                                              "length": 250}}}},
    "hs_99": {"type": "historical", "cf": 0.99,
              "window": {"type": "trailing", "length": 250}},
    "fhs_99": {"type": "filtered_historical", "cf": 0.99,
               "filter": {"vol": {"type": "ewma",
                                  "decay": {"type": "constant",
                                            "value": 0.94}}},
               "window": {"type": "trailing", "length": 500}},
    "var_ext": {"type": "input", "name": "var_ext", "cf": 0.99}
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
ctx = Context.from_spec(RunSpec.model_validate_json(text))
results = ctx.run(returns, inputs={"var_ext": precomputed_var})
results.backtests["hs_99"]["kupiec"].p_value[0]
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
the window, e.g. `models.today.vol.decay.window` (`find_last_window`
in `specs.py`). `Context.graph` checks the same on its nodes, for runs built
in Python.

Every backtest runs on every model (R4). Backtests are named, so two
versions of one test can sit side by side.

## From Python

Notebooks and scripts (R17) build nodes inside a `Context` (see
"Contexts"). A service receiving JSON over HTTP parses a `RunSpec` and
calls `Context.from_spec`, which builds the nodes inside a context with
plain functions from the specs, then checks its graph before returning.

```python
with Context(
    backtests={
        "kupiec": KupiecTest,
        "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
    },
) as ctx:
    vol = EwmaVolatility(ConstantDecay(0.94))
    vol_filter = VolatilityFilter(vol)
    for n in (500, 750):
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=n), name=f"fhs_{n}"
        )
    GaussianModel(cf=0.99, vol=vol, name="gauss")
results = ctx.run(returns)
```

- Sharing is by value (D3, D9): `vol` and `vol_filter` above are computed
  once, for all the models that read them, and would be even if each model
  built its own `EwmaVolatility(ConstantDecay(0.94))`. Nodes built from
  specs share by the same value identity.
- Each backtest is a function from a model's `Overshootings` to the test
  node: the class itself (`KupiecTest`) for its defaults, or a lambda for
  its settings. Every backtest runs on every model (R4), and the backtests
  of a model share its `Overshootings`. ty checks lambdas; it accepts
  `functools.partial` without checking its arguments.
- Values are checked when each node is built, e.g. `ConstantDecay(1.5)`
  raises, and so is a model built in a context without a name. ty reports
  a missing argument or a node in the wrong slot, e.g. a mean model given
  as `vol`.
- `ctx.run` walks the graph before computing, so missing inputs are
  reported first, and backtests on a model with a `last` window are
  rejected, with the path to the window.
- A context built from nodes has no spec, so it can't be written back as
  JSON.

Specs can be built in Python too, but only as data: a validated
description of a run, turned into nodes by `Context.from_spec`, e.g. to
record what ran with `spec.model_dump_json()`. They hold no logic of their
own; the computation is in the nodes. The class picks the spec, so the `type`
tags can be left out, and a list of values (one per series) is a tuple.
Values are checked when each spec is built, so an error's path starts at
that spec, e.g. `value.list.1` for `ConstantDecaySpec(value=(0.94, 1.5))`.

A context without models computes the other nodes built in its block, e.g.
a volatility on its own. `Pool` and `node.run` (see "The node graph") still
do it too, until step 5c-3 removes them.

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

## Run reports (R19)

In progress, see "Run report" in `docs/open-points.md`.

`build_report_data(context, results, title=..., series_labels=...)`
computes nothing: it reads every node of `context.graph()` from
`results.outputs`, and the returns from `results.returns`, into a
`ReportData`: a pydantic model, the one contract with the report page. It
raises `ValueError` if the results lack a node of the graph, e.g. results
of another context. In it:

- a node is identified by its first path, e.g. `models.gauss.vol`;
  `models` maps each model name to its node id, and each backtest keeps the
  id of its node;
- each node has its `kind` (preprocessor, model, overshootings, backtest),
  class, paths, run input, and settings, written as in the specs, e.g.
  `{"type": "trailing", "length": 250, "step": 1}`;
- the batch axes are flattened to one list of series, labelled by
  `series_labels` or by index (`0`, or `(0, 1)` with several axes); every
  series of values is a list over the series, then the dates (or windows);
- `outputs` holds every time-series node's output, by node id; backtests
  results are by model, then backtest name;
- `NaN` is written `null`, dates as ISO strings.

`Report(context, results, title=..., series_labels=...)` calls
`build_report_data`, and keeps the result as `report.data`. So a run is
computed once, then reported (risk-manager review, finding 2):
`results = ctx.run(returns, inputs)`, then `Report(ctx, results)`.
`report.save(path)` reads the page, `report/index.html` (package data,
built ahead of time, so pip installs need no Node), and replaces the
placeholder in its data slot, `<script id="report-data"
type="application/json">__TAIL_LIB_REPORT_DATA__</script>`, with the JSON,
`<` escaped as `\u003c` so no text in the data can close the script.

The page comes from the report app, `report-app/` (Svelte, TypeScript,
Vite; see its README). `npm run build` inlines everything into one
`index.html` (`vite-plugin-singlefile`) and writes it into the package;
the built file is committed. Its types come from `ReportData`'s JSON
Schema (`npm run types`, run before `dev`, `check` and `build`, not
committed), so the page and the Python side check against one contract
(R11). The page reads its data in `src/data.ts`: the slot's JSON
in a saved report, or the sample data (`npm run sample`) in development.
CI type checks and rebuilds the page, and fails if it differs from the
committed one, so the page always matches the app's source.

The graph tab (`src/tabs/GraphTab.svelte`) draws every node once with
ECharts (`src/charts.ts` registers only the parts used). The positions come
from `src/graph/layout.ts`, plain code with no layout library (developer's
choice):

- layer 0 holds the returns and one box per run input, with an arrow to
  each node reading it (the returns, read by most nodes, have none);
- a preprocessor sits one layer below the deepest preprocessor it reads;
- then one layer each for the models, the overshootings and the backtests;
- each layer is centred, and ordered by the mean position of the boxes it
  reads above, in one top-down pass, so arrows cross less.

`src/graph/format.ts` writes the labels (display name for models and
backtests, class, settings on one line) and the hover details (settings in
full, every path). The chart is drawn at scale 1, and a wide graph scrolls
sideways; scrolling zooms and dragging pans.

The returns vs VaR tab (`src/tabs/ReturnsVarTab.svelte`) charts one series,
picked by the selector in the header (shared by the tabs). Each model's VaR
is drawn on the date it forecasts: the VaR of date `t - 1` on date `t`
(D1), next to the return it is tested against, so an overshooting is a
return below the line on the same date (developer's choice). Dots mark the
overshootings, read from each model's `Overshootings` output, so only runs
with backtests have them: the page computes nothing itself. Points are
dated (`[date, value]` on a time axis), and zooming keeps every point, so
the tooltip reads every line at the same date. `src/Chart.svelte` is the
ECharts wrapper the tabs share; `src/models.ts` lists the models with their
overshootings node.

The backtests tab (`src/tabs/BacktestsTab.svelte`) is one table, a row per
model, backtest and series: the window, dates tested, overshootings,
observed and expected rates, statistic and p-value. A backtest on rolling
windows shows its last window, named from its `window` setting (e.g. "250
dates to 2025-10-31 (last of 731 windows)"). The p-value cell is red,
with "rejected", below a level set above the table, 5% by default and not
saved; green otherwise (developer's choices). The series selector shows
only on the tabs that chart one series.

The rolling backtests tab (`src/tabs/RollingTab.svelte`) has one chart per
backtest with several windows, for the selected series: the observed rate
on top, with each model's expected rate dashed, and the p-value below,
with the rejection level dashed in red. Each point is a window, on its
last date. The level lives in `App.svelte`, shared by the backtests and
rolling tabs.

The node outputs tab (`src/tabs/NodeOutputsTab.svelte`) charts the output
of one node, for the selected series, on the dates it holds them (a VaR on
the date it is computed, unlike the returns vs VaR tab). The node is picked
in a list, or by clicking its box in the graph: `GraphTab` passes the id to
`App.svelte`, which opens this tab, or Backtests for a backtest node. Decay
factors, filtered returns and overshooting flags read as plain numbers,
the other outputs as percentages.

## Adding a node type

1. Write the node as a frozen dataclass (`@dataclass(frozen=True)`), its
   fields being its constructor arguments, arrays stored with
   `frozen_floats` in `__post_init__` (which ends with
   `super().__post_init__()`). Write `compute(self, returns, /, *, ...)`,
   pure, with one keyword per input field. The
   docstring states the formula, the timing, the fields (`Attributes:`)
   and a reference. `dependencies` and `settings` come from the fields. Add one instance to
   `_one_node_of_each_type` in `tests/test_dependencies.py`: the test fails
   until you do.
2. Add its spec to `specs.py`, with value ranges matching the node's checks,
   and add the spec to its role's union.
3. Add a `case` to the role's `_build_...` method in `build.py`. Because of
   `assert_never`, ty reports a spec type with no case.
4. Test that the spec builds the same numbers as the node written in Python.
5. Pin its output in `tests/test_regression.py` (add it to `_nodes`, then
   `uv run python -m tests.test_regression`), so later refactors can't
   change its numbers unnoticed.

For a new role (e.g. a second backtest), turn the role's alias into a tagged
union (`BacktestSpec`, and `BacktestResult` in `build.py`).
