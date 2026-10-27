# Runs in Python

In notebooks and scripts, build models as Python objects (nodes) inside a
[`Context`][tail_lib.context.Context] (R17). A context collects the models
built in its `with` block, runs every backtest you give it on each of them,
and computes them on any returns series.

## The steps

```python
from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.context import Context
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.windows import Trailing

# 1. The context, with the backtests to run on every model.
with Context(
    backtests={
        "kupiec": KupiecTest,
        "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
    },
) as ctx:
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

Every backtest runs on every model (R4). A backtest is given as a function
from a model's overshootings to the test node:

- the class itself, `KupiecTest`, for its default settings (the whole
  sample);
- a lambda, for other settings, e.g.
  `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`.

The context builds one
[`Overshootings`][tail_lib.backtests.overshootings.Overshootings] node per
model, and all the backtests of that model share it. Backtests are named, so
several versions of one test can sit side by side, e.g. on 250 and 500
dates. See [Backtests](backtests.md) for the tests and their results.

## Names and labels

Names are Python identifiers (letters, digits and underscores, not starting
with a digit). Two different nodes of a context can't share a name, and no
node can take a backtest's name. For display, e.g. in a report, give
friendlier names in `labels`:

```python
with Context(
    backtests={"kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250))},
    labels={"fhs_500": "Filtered HS, 500 days", "kupiec_250": "Kupiec, 250 days"},
) as ctx:
    ...
ctx.labels  # every model and backtest, by name; unlabelled names as they are
```

## Results

`ctx.run(returns)` returns [`Results`][tail_lib.context.Results]:

`results.models[model]`
:   The VaR of the model, a `TimeSeries` on the dates of the returns, with
    their shape. Dates before the model's first value are `NaN`.

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

with Context(backtests={"kupiec": KupiecTest}) as ctx:
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
from tail_lib.windows import Last

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

## Runs built in Python vs from specs

A context built from nodes has no spec, so it can't be written back as JSON.
To record exactly what ran, write the models as a [spec](specs.md) instead.
`Context.from_spec(spec)` builds the same kind of context from that data,
with the model keys as names, ready for `ctx.run(returns, inputs)`.
