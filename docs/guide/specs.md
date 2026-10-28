# Runs from specs

A spec describes a run as data: JSON, or Python dicts. It is the way in
for a service that receives runs over HTTP, and the form to keep a
record of what ran. A spec holds no logic: `Context.from_spec` turns it into
the same nodes you would build [in Python](python-runs.md).

## The steps

```python
from tail_lib.context import Context
from tail_lib.run import RunSpec

spec = RunSpec.model_validate_json(text)  # or RunSpec.model_validate(a_dict)
ctx = Context.from_spec(spec)
results = ctx.run(returns, inputs={"var_ext": var_ext})
```

1. **Parse.** [`RunSpec`][tail_lib.run.RunSpec] checks every field name, type
   and value range. An error gives the path to the faulty value.
2. **Build.** `Context.from_spec` builds the nodes inside a context. Equal
   nodes share computation by value, so identical parts of separate models
   are computed once.
3. **Run.** As for any context: see [results](python-runs.md#results).

## A run spec

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

`models`
:   The models by name: at least one.

`backtests`
:   The backtests by name, optional. Every backtest runs on every model.

`labels`
:   Display names by model or backtest name, optional, e.g.
    `{"gauss_99": "Gaussian 99%"}`. Names without one are shown as they are.

Model, backtest and input names are Python identifiers: letters, digits and
underscores, not starting with a digit, e.g. `gauss_99` but not `gauss 99`
or `kupiec-250`. A node and a backtest can't share a name, nor can two
different nodes.

Run inputs aren't listed separately: they are the `input` specs inside the
models, here `var_ext`. Pass them to `ctx.run` by name. A VaR input's model
key must match its input name, since its node has one name. Missing inputs
are reported before computing any node.

## How specs are written

- **Each model spec mirrors a node.** Its fields are the node's constructor
  arguments, and a node input is a nested spec: the `vol` of a Gaussian model
  is a volatility spec. Backtest specs mirror declarations: `KupiecSpec`
  holds the settings of `Kupiec`; the context supplies its model inputs.
- **A `type` tag picks the spec** in each slot, e.g. `"gaussian"` or
  `"historical"` for a model. Tags are unique within a slot, not across
  slots: `"constant"` is a constant volatility in a `vol` slot and a constant
  mean in a `mean` slot.
- **Every node is spelled out.** There are no shorthands: a constant decay is
  `{"type": "constant", "value": 0.94}`, not `0.94`, and every window needs
  its `type`.
- **Fields with a default can be left out.** The tables below give each
  default.
- **Unknown fields are errors**, as are `NaN` and infinite numbers.

### One value or one per series

Fields marked *values* take a single number, or a non-empty list with one
number per series (for returns of shape `(n_dates, n)`):

```json
{"type": "constant", "value": [0.94, 0.97]}
```

## Spec catalogue

The Python spec classes are in [`tail_lib.specs`](../reference/specs.md).
[Models](models.md) describes what each node computes.

### Value-at-risk models: `models.<name>`

| `type` | Node | Fields |
|---|---|---|
| `historical` | `HistoricalSimulationsModel` | `cf`, `window`, `interpolation` = `"linear"` |
| `filtered_historical` | `FilteredHistoricalSimulationsModel` | `cf`, `filter` (a [filter](#volatility-filter-filter)), `window`, `interpolation` = `"linear"` |
| `gaussian` | `GaussianModel` | `cf`, `vol` (a [volatility](#volatilities-vol)), `mean` (a [mean](#means-mean)) = `{"type": "zero"}` |
| `input` | `ValueAtRiskInput` | `name`, `cf` |

- `cf`: the confidence level, in (0, 1).
- `window`: a [window](#windows-window).
- `interpolation`: how the quantile falls between order statistics,
  `"linear"`, `"below"` or `"above"`.
- `name`: the name of the run input.

### Volatilities: `vol`

| `type` | Node | Fields |
|---|---|---|
| `constant` | `ConstantVolatility` | `vol`: *values*, positive |
| `ewma` | `EwmaVolatility` | `decay` (a [decay factor](#decay-factors), constant or calibrated), `start` (an [EWMA start](#ewma-starts-start)) = `{"type": "backcast", "window": 20}`, the calibration's `start` with a calibrated decay |
| `input` | `VolatilityInput` | `name` |

### Volatility filter: `filter`

The `filter` of a filtered historical simulation. It has a single form, so
its `type` (`"volatility_filter"`) can be left out.

| Fields | |
|---|---|
| `vol` | a [volatility](#volatilities-vol) |
| `mean` | a [mean](#means-mean), default `{"type": "zero"}` |

### Means: `mean`

| `type` | Node | Fields |
|---|---|---|
| `zero` | `ZeroMean` | none |
| `constant` | `ConstantMean` | `mean`: *values* |

### Decay factors

An `ewma` volatility takes a constant or a calibrated decay in `decay`, so
the `type` is needed.

| `type` | Node | Fields |
|---|---|---|
| `constant` | `ConstantDecay` | `value`: *values*, in [0, 1] |
| `ewma_calibration` | `EwmaDecayCalibration` | `window`, `bounds` = `[0.8, 0.9999]`, `start` = `{"type": "backcast", "window": 20}`, `n_iter` = `40` |

- `window`: the training windows of the fits, a [window](#windows-window).
- `bounds`: the search interval of the decay, lower below upper, both in
  [0, 1].
- `start`: how each fit's recursion starts, an [EWMA start](#ewma-starts-start).
- `n_iter`: the number of golden-section iterations of each fit, at least 1.

### EWMA starts: `start`

| `type` | Node | Fields |
|---|---|---|
| `backcast` | `Backcast` | `window` = `20`: number of dates, at least 1 |
| `starting_vol` | `StartingVol` | `vol`: *values*, positive |

### Windows: `window`

| `type` | Node | Fields |
|---|---|---|
| `trailing` | `Trailing` | `length`, `step` = `1`: both at least 1 |
| `last` | `Last` | `length`: at least 1 |

### Backtests: `backtests.<name>`

The context converts each spec into a declaration and registers it through
`add_backtest`, as for Python declarations. Wiring happens during the graph
walk; the JSON format is unchanged.

| `type` | Declaration | Fields |
|---|---|---|
| `kupiec` | `Kupiec` | `window` = `null`: a [window](#windows-window), or `null` for the whole sample |

## Errors

An invalid spec raises a pydantic `ValidationError` that names the faulty
value and says what was expected. The path runs through the field
names and the `type` tags:

```text
1 validation error for RunSpec
models.g.gaussian.vol.ewma.decay.value.list.1
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.5, input_type=float]
```

Here, the second decay factor (`list.1`) of the EWMA volatility of model `g`
is above 1. A misspelled tag lists the tags that slot accepts:

```text
1 validation error for RunSpec
models.g
  Input tag 'gausian' found using 'type' does not match any of the expected tags: 'input', 'historical', 'filtered_historical', 'gaussian' [type=union_tag_invalid, ...]
```

`error.errors()` gives each error as a dict, with its path (`loc`) and
message (`msg`), e.g. to send back from a service.

Three more rules are checked on the whole run:

- **No backtests on a `last` model.** A model with a `last` window, or
  reading a node that has one, gives one value, on the last date: there is
  nothing to backtest. A run with backtests rejects it, with the path to the
  window:

    ```text
    Model 'h' has a `last` window, at `models.h.window`, so it gives one forecast and can't be backtested, but the run has backtests ['k']!
    ```

    Put such models in their own run, without backtests. A `last` window on a
    *backtest* is fine: it tests the last dates only.

- **One spec per input name.** `Context.from_spec` raises a `ValueError` if
  two input specs share a name but differ, e.g. in their `cf`.

- **One name for a VaR input.** `Context.from_spec` raises a `ValueError`
  naming the model's path if an input model's key differs from its `name`.
  Use `"var_ext": {"type": "input", "name": "var_ext", "cf": 0.99}`.

## Sharing

Specs are frozen and compare by value. `Context.from_spec` builds equal
nodes for identical parts of separately written models, and the context
computes each once. In the example above, writing the same `filter` in
two `filtered_historical` models on 500 and 750 dates gives one filter,
computed once.

## Writing a spec back

`spec.model_dump_json()` writes a spec as JSON, with every default spelled
out, and parses back to an equal spec. Use it to record exactly what ran.

```python
spec.model_dump_json(indent=2)
RunSpec.model_validate_json(spec.model_dump_json()) == spec  # True
```

## Specs in Python

Spec classes can also be built in Python, as data: a checked description of a
run, to turn into nodes with `Context.from_spec`, or to save. The class picks
the spec, so `type` tags can be left out, and a list of values is a tuple.

```python
from tail_lib.run import RunSpec
from tail_lib.specs import ConstantDecaySpec, EwmaVolatilitySpec, GaussianSpec

spec = RunSpec(
    models={
        "gauss_99": GaussianSpec(
            cf=0.99, vol=EwmaVolatilitySpec(decay=ConstantDecaySpec(value=0.94))
        )
    }
)
spec.model_dump_json()
```

Values are checked as each spec is built, so an error's path starts at that
spec, e.g. `value.list.1` for `ConstantDecaySpec(value=(0.94, 1.5))`.

For code that only runs models, building [nodes](python-runs.md) is simpler.
