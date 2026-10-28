# Runs in Python

In notebooks and scripts, build models as Python objects (nodes) inside a
[`Context`][tail_lib.context.Context] (R17). A context collects the models
built in its `with` block, runs every backtest you give it on each of them,
and computes them on any returns series.

## The steps

```python
from tail_lib.backtests.kupiec import Kupiec
from tail_lib.context import Context
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.time_series import Last
from tail_lib.time_series import Trailing

# 1. The context, with the backtests to run on every model.
with Context() as ctx:
    Kupiec(name="kupiec")
    Kupiec(name="kupiec_250", window=Trailing(length=250))

    # 2. The models, each with a name. `vol` and `vol_filter` are shared:
    # computed once.
    vol = EwmaVolatility(ConstantDecay(0.94))
    vol_filter = VolatilityFilter(vol)
    HistoricalSimulationsModel(cf=0.99, window=Trailing(length=500), name="hs_500")
    FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=500), name="fhs_500"
    )
    FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=750), name="fhs_750"
    )
    GaussianModel(cf=0.99, vol=vol, name="gauss")

# 3. Compute, on any returns series.
results = ctx.run(returns)
```

A context can be computed many times, on different returns: build it once,
then call `ctx.run(...)` for each series.

## Models

Every node built inside the `with` block belongs to the context. Any of the
[value-at-risk models](models.md#value-at-risk-models) fits:
`HistoricalSimulationsModel`, `FilteredHistoricalSimulationsModel`,
`GaussianModel`, or `ValueAtRiskInput` for forecasts computed elsewhere.
`ctx.models` lists them by name.

A model built in a context needs a `name`: building one without raises a
`ValueError` on that line. Outside a context, no name is needed.

Values are checked as each node is built, e.g. `ConstantDecay(1.5)` raises a
`ValueError` straight away. A type checker such as `ty` or `mypy` also
reports a missing argument, or a node in the wrong slot, e.g. a mean model
given as `vol`.

Nodes are identified by value: two nodes of the same class with the same
arguments are one node, computed once, however they were built. A node built
before the block, outside any context, can be used inside it, e.g. a
volatility shared by several contexts.

## Backtests

Every registered backtest runs on every model (R4). Declare the test's
settings and its name inside the context block:

```python
with Context() as ctx:
    Kupiec(name="kupiec_last", window=Last(length=250))
    ...  # the models
```

A [`Kupiec`][tail_lib.backtests.kupiec.Kupiec] declaration is immutable and
independent of any model. Creating one inside a `with Context():` block
registers it on the innermost context, as for model nodes. A name is
required there and is checked when the declaration is built. The name is
registration metadata, so equal settings under different names still
share computation.

Outside a context, a declaration can be unnamed and reused. Pass it through
the constructor's `backtests` mapping, or register it explicitly with
`ctx.add_backtest(name, declaration)`. Mapping keys and explicit names
choose its name in that context:

```python
rolling_test = Kupiec(window=Trailing(250))  # outside any context
with Context(backtests={"kupiec_250": rolling_test}) as ctx:
    ...  # the models
```

Registration records settings only and computes nothing. When `ctx.graph()`
builds the graph, the context binds each declaration to every VaR model,
including models added after registration. The internal executable nodes
read each model's overshootings and confidence level. Their shared
[`Overshootings`][tail_lib.backtests.overshootings.Overshootings] node is
computed once (D9).

A backtest name must be a Python identifier, and cannot belong to a named
node or an already registered backtest. Registering the same declaration
under different names is allowed; equal executable nodes compute once.
Results remain `results.backtests[model][test]`. See
[Backtests](backtests.md) for the tests and their results.

## Names and labels

Names are Python identifiers (letters, digits and underscores, not starting
with a digit). Two different nodes of a context can't share a name, and no
node can take a backtest's name. For display, e.g. in a report, give
friendlier names in `labels`:

```python
with Context(
    backtests={"kupiec_250": Kupiec(window=Trailing(length=250))},
    labels={"fhs_500": "Filtered HS, 500 days", "kupiec_250": "Kupiec, 250 days"},
) as ctx:
    ...
ctx.labels  # every model and backtest, by name; unlabelled names as they are
```

## Common history

Each execution starts at the first date where every portfolio has a return.
An earlier-starting portfolio loses its earlier observations when grouped
with a later-starting one, so all portfolios use the same history (R21).
To use different histories, put the portfolios in separate runs.

Pass external inputs on the original dates and with the original shape of
the returns. The context checks them before trimming, then slices them
by the same amount as the returns. Missing values in an input do not move
the start. Missing returns inside the remaining history are left in place.
If no date has a return for every portfolio, the run fails before computing.

EWMA initialization and calibration windows use this common history.
Calibrations start with its first complete training window, then repeat
every `step` observations. Trim the history yourself to choose a different
calibration anchor. Each execution finds its own start; the context and
the caller's arrays are unchanged.

## Date range

Give inclusive forecast bounds to a specific execution of a context:

```python
results = ctx.run(returns, lower=returns.dates[-100], upper=returns.dates[-1])
```

Bounds accept Python `datetime.date` or NumPy `datetime64`. If one is
omitted, it defaults to the corresponding endpoint of the common trimmed
history. Bounds must stay inside that history. They need not be observation
dates: a weekend bound includes only the observations inside the interval.
An interval containing no observations is an error.

All nodes compute on the full trimmed history. Calibrations keep their
original refit dates, so changing the range does not change overlapping
VaRs. Every output keeps the dates and shape of `results.returns` (R24).
VaR values outside the selected range are `NaN`; historical preprocessors
remain available through `results[node]`. Reading a model through
`results.models`, its name or its node gives the same masked series.
`results.lower` and `results.upper` hold the first and last selected
observation dates, after resolving the bounds against the history. For
example, a Saturday lower bound can resolve to Monday. Both are `None`
for runs without bounds.

Before computing, the context checks the built-in models' warm-ups against
the first selected forecast date. A model that needs more history fails
with its name and availability, rather than silently giving a shorter
sample. Custom computations can have additional warm-ups: the run also
checks actual VaR coverage after computing each model. Missing forecasts
anywhere inside the selected range are an error. Bounded HS and FHS
require `Trailing(..., step=1)` so there is a fresh VaR on every date.
Calibrations and backtests keep their own steps.

Backtests use flags after the first forecast date through the upper date.
The first flag compares the next observation's return with the first
selected VaR. Rolling windows start with those flags, and whole-sample and
last-window tests end at the last selected date. A single-date run can
produce today's VaR, but cannot run backtests.

Calls without either bound retain their full-history behavior, including
leading warm-up `NaN`s and existing VaR steps. `Last` windows are retained:
they still produce a forecast only on the last date of the supplied
history. Use a single-date range with a trailing model to select a
historical date without changing the calibration anchor. Specs describe
the context; execution bounds are passed to `Context.run`, including for
contexts built with `Context.from_spec`.

## Results

`ctx.run(returns)` returns [`Results`][tail_lib.context.Results]:

`results.models[model]`
:   The VaR of the model, a `TimeSeries` on the dates of `results.returns`,
    the common trimmed history, with its shape. Dates before the model's
    first value are `NaN`. With bounds, dates outside the requested range
    are also `NaN`.

`results.backtests[model][backtest]`
:   The result of a backtest on a model, e.g. a
    [`KupiecResult`][tail_lib.backtests.kupiec.KupiecResult].

`results[node]`, `results["name"]`
:   The output of any node of the context, e.g. `results[vol]`, or of a node
    named in the block, e.g. `results["gauss"]`.

```python
var = results.models["fhs_500"]
var.dates, var.values

test = results.backtests["fhs_500"]["kupiec"]
test.n_overshootings[0], test.p_value[0]  # whole sample: one window

rolling = results.backtests["fhs_500"]["kupiec_250"]
rolling.end_dates, rolling.p_value  # one result per window

results[vol]  # the EWMA volatility, computed once for three models
```

## Run inputs

To bring in data computed elsewhere, use an input node (R13):

- [`ValueAtRiskInput(name, cf=...)`][tail_lib.models.value_at_risk.ValueAtRiskInput]:
  VaR forecasts, e.g. to backtest another system's VaR next to the library's
  models (R3). It is a model, named after its input;
- [`VolatilityInput(name)`][tail_lib.models.volatility.VolatilityInput]:
  volatilities, e.g. from a data service, read by a Gaussian model or a
  filter.

Pass each input to `ctx.run` under its name, as a `TimeSeries` with the dates
and shape of the returns.

```python
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import VolatilityInput

with Context(backtests={"kupiec": Kupiec()}) as ctx:
    ValueAtRiskInput("var_ext", cf=0.99)
    GaussianModel(cf=0.99, vol=VolatilityInput("vol_svc"), name="gauss_svc")

results = ctx.run(returns, inputs={"var_ext": var_ext, "vol_svc": vol_svc})
results.backtests["var_ext"]["kupiec"]  # the external VaR, backtested
```

Inputs follow the library's timing: the value on `t` is the forecast for
`t + 1`. The library can't check this for you.

## Errors reported up front

Most mistakes fail before anything is computed:

| Mistake | Error |
|---|---|
| A backtest without a name, in a context | `ValueError` when it is built. |
| A duplicate backtest registration | `ValueError` when the second declaration is built (or from explicit registration); the original is kept. |
| A model without a name, in a context | `ValueError` on the line building it. |
| Two different nodes with one name, or a node with a backtest's name | `ValueError` on the line building the second. |
| A model with a `Last` window, in a context with backtests | `ValueError` from `ctx.run` (or `ctx.graph()`), with the path to the window, e.g. `models.today.vol.decay.window`. Put such models in their own context, without backtests. |
| A label naming no model or backtest | `ValueError` from `ctx.run` (or `ctx.graph()`). |
| Two different nodes reading the same input name | `ValueError` from `ctx.run` (or `ctx.graph()`). Models that read the same input share one input node. |
| A missing input | `KeyError` from `ctx.run`, before computing, e.g. `Missing inputs ['var_ext']! Inputs given: [].` |
| An input with other dates or another shape than the returns | `ValueError` when the run reads it. |

Two errors only show up while computing: a node whose output doesn't have
the dates and shape of the returns (a bug in a node of your own), and a
window longer than the data it runs on, e.g. `Window of 250 dates is longer
than the 230 dates from the first date where every series has a value!`.
Warm-ups add up: the example above needs at least 1,020 returns, i.e. 20
dates of backcast, 750 for `fhs_750`'s first window, and 250 for the first
`kupiec_250` window.

## Today's VaR

To compute only the latest VaR, give the models a `Last` window, in a
context without backtests:

```python
from tail_lib.time_series import Last

with Context() as today:
    HistoricalSimulationsModel(cf=0.99, window=Last(length=500), name="hs_500")

today.run(returns).models["hs_500"].values[-1]  # every other date is NaN
```

## Without VaR models

A context computes every node built in its block, so it also computes
volatilities, mean models and filters on their own:

```python
from tail_lib.models.mean import ConstantMean

with Context() as inputs_only:
    vol = EwmaVolatility(ConstantDecay(0.94))
    vol_filter = VolatilityFilter(vol, mean=ConstantMean(0.0005))

results = inputs_only.run(returns)  # computes vol once
results[vol], results[vol_filter].standardized
```

## The graph of a context

`ctx.graph()` lists every node of the context once, backtests and
overshootings included, with the paths it is reached by and the edges
between nodes. It computes nothing:

```python
graph = ctx.graph()
for graph_node in graph.nodes:
    type(graph_node.node).__name__, graph_node.paths
    # e.g. ("EwmaVolatility", ("models.fhs_500.filter.vol", "models.fhs_750.filter.vol", ...))

for edge in graph.edges:
    edge.source, edge.name, edge.target  # e.g. vol, "vol", gauss
```

Nodes no model reaches have paths from `nodes`: `nodes.<name>`, or
`nodes.<class>` without a name.

## Reports

A [`Report`][tail_lib.report.report.Report] shows the results of a run as
one HTML file that opens offline (R19). It reads the results without
computing anything again. Its tabs:

- **Graph:** every node of the context once, with its settings; click a box
  to chart its output.
- **Returns vs VaR:** the returns against each model's VaR, on the date it
  forecasts, with the overshootings marked.
- **Backtests:** every backtest of every model, coloured by rejection at a
  level you set.
- **Rolling backtests:** the observed rate and p-value of rolling backtests
  over time.
- **Node outputs:** the output of any node, e.g. a volatility.

```python
from tail_lib.report.report import Report

results = ctx.run(returns, inputs)
report = Report(ctx, results, title="Daily VaR", series_labels=["SPX", "SX5E"])
report.save("report.html")
report.data  # the ReportData shown, e.g. report.data.model_dump_json()
```

The report header shows the selected **Forecast range** separately from
the **Available history** in `results.returns`. With no execution bounds,
it labels the forecast range **Full history**; individual models may still
have warm-up `NaN`s. Bounds are saved as ISO dates in
`report.data.metadata.lower` and `.upper`, so the report reads the dates
from that execution without computing anything again.

## Runs built in Python vs from specs

A context built from nodes has no spec, so it can't be written back as JSON.
To record exactly what ran, write the models as a [spec](specs.md) instead.
`Context.from_spec(spec)` builds the same kind of context from that data,
with the model keys as names, ready for `ctx.run(returns, inputs)`.
