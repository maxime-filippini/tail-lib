# Runs in Python

In notebooks and scripts, build models as Python objects (nodes) and give
them to a [`Run`][tail_lib.run.Run] (R17). A run computes named models and
named backtests on any returns series.

## The steps

```python
from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.run import Run
from tail_lib.windows import Trailing

# 1. The models, by name. `vol` and `vol_filter` are shared: computed once.
vol = EwmaVolatility(ConstantDecay(0.94))
vol_filter = VolatilityFilter(vol)
models = {
    "hs_500": HistoricalSimulationsModel(cf=0.99, window=Trailing(length=500)),
    "fhs_500": FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=500)
    ),
    "fhs_750": FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=750)
    ),
    "gauss": GaussianModel(cf=0.99, vol=vol),
}

# 2. The run: the models, and the backtests to run on each of them.
run = Run(
    models,
    backtests={
        "kupiec": KupiecTest,
        "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
    },
)

# 3. Compute, on any returns series.
result = run.run(returns)
```

A run can be computed many times, on different returns: build it once, then
call `run.run(...)` for each series.

## Models

`models` maps a name to a value-at-risk model node. Any of the
[value-at-risk models](models.md#value-at-risk-models) fits:
`HistoricalSimulationsModel`, `FilteredHistoricalSimulationsModel`,
`GaussianModel`, or `ValueAtRiskInput` for forecasts computed elsewhere.

Values are checked as each node is built, e.g. `ConstantDecay(1.5)` raises a
`ValueError` straight away. A type checker such as `ty` or `mypy` also
reports a missing argument, or a node in the wrong slot, e.g. a mean model
given as `vol`.

## Backtests

Every backtest runs on every model (R4). A backtest is given as a function
from a model's overshootings to the test node:

- the class itself, `KupiecTest`, for its default settings (the whole
  sample);
- a lambda, for other settings, e.g.
  `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`.

The run builds one [`Overshootings`][tail_lib.backtests.overshootings.Overshootings]
node per model, and all the backtests of that model share it. Backtests are
named, so several versions of one test can sit side by side, e.g. on 250 and
500 dates. See [Backtests](backtests.md) for the tests and their results.

## Names and labels

Model, backtest and input names are Python identifiers (letters, digits and
underscores, not starting with a digit), and a model and a backtest can't
share a name. For display, e.g. in a report, give friendlier names in
`labels`:

```python
run = Run(
    models,
    backtests={"kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250))},
    labels={"fhs_500": "Filtered HS, 500 days", "kupiec_250": "Kupiec, 250 days"},
)
run.labels  # every model and backtest, by name; unlabelled names as they are
```

## Results

`run.run(returns)` returns a [`RunResult`][tail_lib.run.RunResult]:

`result.models[model]`
:   The VaR of the model, a `TimeSeries` on the dates of the returns, with
    their shape. Dates before the model's first value are `NaN`.

`result.backtests[model][backtest]`
:   The result of a backtest on a model, e.g. a
    [`KupiecResult`][tail_lib.backtests.kupiec.KupiecResult].

```python
var = result.models["fhs_500"]
var.dates, var.values

test = result.backtests["fhs_500"]["kupiec"]
test.n_overshootings[0], test.p_value[0]  # whole sample: one window

rolling = result.backtests["fhs_500"]["kupiec_250"]
rolling.end_dates, rolling.p_value  # one result per window
```

## Run inputs

To bring in data computed elsewhere, use an input node (R13):

- [`ValueAtRiskInput(name, cf=...)`][tail_lib.models.value_at_risk.ValueAtRiskInput]:
  VaR forecasts, e.g. to backtest another system's VaR next to the library's
  models (R3);
- [`VolatilityInput(name)`][tail_lib.models.volatility.VolatilityInput]:
  volatilities, e.g. from a data service, read by a Gaussian model or a
  filter.

Pass each input to `run.run` under its name, as a `TimeSeries` with the dates
and shape of the returns. `run.input_names` lists the names the run reads.

```python
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import VolatilityInput

run = Run(
    {
        "external": ValueAtRiskInput("var_ext", cf=0.99),
        "gauss_svc": GaussianModel(cf=0.99, vol=VolatilityInput("vol_svc")),
    },
    backtests={"kupiec": KupiecTest},
)
run.input_names  # frozenset({'var_ext', 'vol_svc'})
result = run.run(returns, inputs={"var_ext": var_ext, "vol_svc": vol_svc})
```

Inputs follow the library's timing: the value on `t` is the forecast for
`t + 1`. The library can't check this for you.

## Errors reported up front

`Run` walks the graph of its models when it is built, and `run.run` checks
the inputs before computing anything. So these mistakes fail fast:

| Mistake | Error |
|---|---|
| No model | `ValueError` when building the run. |
| A model with a `Last` window, in a run with backtests | `ValueError` when building the run, with the path to the window, e.g. `models.today.vol.decay.window`. Put such models in their own run, without backtests. |
| Two different nodes reading the same input name | `ValueError` when building the run. Models that read the same input share one input node. |
| A missing input | `KeyError` from `run.run`, before computing, e.g. `Missing inputs ['var_ext']! Inputs given: [].` |
| An input with other dates or another shape than the returns | `ValueError` when the input is read. |

One error only shows up while computing: a window longer than the data it
runs on, e.g. `Window of 250 dates is longer than the 230 dates from the
first date where every series has a value!`. Warm-ups add up: the example
above needs at least 1,020 returns, i.e. 20 dates of backcast, 750 for
`fhs_750`'s first window, and 250 for the first `kupiec_250` window.

## Today's VaR

To compute only the latest VaR, give the models a `Last` window, and leave
out the backtests:

```python
from tail_lib.windows import Last

today = Run({"hs_500": HistoricalSimulationsModel(cf=0.99, window=Last(length=500))})
today.run(returns).models["hs_500"].values[-1]  # every other date is NaN
```

## Without a run

A single node can be computed on its own with `node.run(returns, inputs)`,
and a set of named nodes with [`Pool`][tail_lib.graph.Pool]. This is the way
to compute volatilities, mean models and filters without a VaR model, as a
`Run` only takes VaR models:

```python
from tail_lib.graph import Pool
from tail_lib.models.mean import ConstantMean

vol.run(returns)  # a TimeSeries
Pool(
    {
        "vol": vol,
        "filtered": vol_filter,
        "mean": ConstantMean(0.0005),
    }
).run(returns)  # computes vol once
```

Here, backtests are wired to their model by hand
(`KupiecTest(Overshootings(model))`), and missing inputs are only reported
when read. There is no spec form for this: a `RunSpec` only holds VaR
models.

## The graph of a run

`run.graph` lists every node of the run once, backtests and overshootings
included, with the paths it is reached by and the edges between nodes:

```python
for graph_node in run.graph.nodes:
    type(graph_node.node).__name__, graph_node.paths
    # e.g. ("EwmaVolatility", ("models.gauss.vol", "models.fhs_500.filter.vol", ...))

for edge in run.graph.edges:
    edge.source, edge.name, edge.target  # e.g. vol, "vol", gauss
```

To read the output of any node after a run, compute the run on a cache you
hold, with `run.evaluate(cache)`; every node of the graph is then in it:

```python
from tail_lib.graph import Cache

cache = Cache(returns, inputs)
result = run.evaluate(cache)
cache.get(vol)  # already computed, so read for free
```

## Reports

A [`Report`][tail_lib.report.report.Report] shows the results of a run as
one HTML file that opens offline (R19). It takes a
[`Context`][tail_lib.context.Context], which is replacing `Run` (this guide
moves to contexts next), and the results of its run, which it reads
without computing anything again. Its tabs:

- **Graph:** every node of the run once, with its settings; click a box to
  chart its output.
- **Returns vs VaR:** the returns against each model's VaR, on the date it
  forecasts, with the overshootings marked.
- **Backtests:** every backtest of every model, coloured by rejection at a
  level you set.
- **Rolling backtests:** the observed rate and p-value of rolling backtests
  over time.
- **Node outputs:** the output of any node, e.g. a volatility.

```python
from tail_lib.context import Context
from tail_lib.report.report import Report

with Context(backtests={"kupiec": KupiecTest}) as ctx:
    GaussianModel(cf=0.99, vol=EwmaVolatility(ConstantDecay(0.94)), name="gauss")

results = ctx.run(returns, inputs)
report = Report(ctx, results, title="Daily VaR", series_labels=["SPX", "SX5E"])
report.save("report.html")
report.data  # the ReportData shown, e.g. report.data.model_dump_json()
```

## Runs built in Python vs from specs

A run built from nodes has no spec, so it can't be written back as JSON. To
record exactly what ran, build the run from a [spec](specs.md) instead: both
ways give the same `Run`, the same checks and the same numbers.
