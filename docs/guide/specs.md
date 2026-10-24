# Runs from specs

A spec describes a run as data: JSON, or Python dicts (R9). It is the way in
for a service that receives runs over HTTP (R17), and the form to keep a
record of what ran. A spec holds no logic: `Run.from_spec` turns it into the
same nodes you would build [in Python](python-runs.md).

## The steps

```python
from tail_lib.run import Run, RunSpec

spec = RunSpec.model_validate_json(text)  # or RunSpec.model_validate(a_dict)
run = Run.from_spec(spec)
result = run.run(returns, inputs={"var_ext": var_ext})
```

1. **Parse.** [`RunSpec`][tail_lib.run.RunSpec] checks every field name, type
   and value range. An error gives the path to the faulty value (R9b).
2. **Build.** `Run.from_spec` builds one node per distinct spec, so identical
   parts of separate models are computed once (D3).
3. **Run.** As for any run: see [results](python-runs.md#results).

## A run spec

```json
{
  "models": {
    "gauss_99": {"type": "gaussian", "cf": 0.99,
                 "vol": {"type": "fitted_ewma",
                         "calibration": {"type": "ewma_calibration",
                                         "window": {"type": "trailing",
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

`models`
:   The models by name: at least one.

`backtests`
:   The backtests by name, optional. Every backtest runs on every model (R4).

Run inputs aren't listed separately: they are the `input` specs inside the
models, here `var_ext`. Pass them to `run.run` by name; `run.input_names`
lists them.

## How specs are written

- **Each spec mirrors a node.** Its fields are the node's constructor
  arguments, and a node input is a nested spec: the `vol` of a Gaussian model
  is a volatility spec.
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
| `ewma` | `EwmaVolatility` | `decay` (a [constant decay](#decay-factors)), `start` (an [EWMA start](#ewma-starts-start)) = `{"type": "backcast", "window": 20}` |
| `fitted_ewma` | `FittedEwmaVolatility` | `calibration` (a [decay calibration](#decay-factors)) |
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

An `ewma` volatility takes a constant decay in `decay`; a `fitted_ewma`
volatility takes a calibration in `calibration`. Each slot takes a single
spec, so the `type` can be left out.

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

| `type` | Node | Fields |
|---|---|---|
| `kupiec` | `KupiecTest` | `window` = `null`: a [window](#windows-window), or `null` for the whole sample |

## Errors

An invalid spec raises a pydantic `ValidationError` that names the faulty
value and says what was expected (R9b). The path runs through the field
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

Two more rules are checked on the whole run:

- **No backtests on a `last` model.** A model with a `last` window, or
  reading a node that has one, gives one value, on the last date: there is
  nothing to backtest. A run with backtests rejects it, with the path to the
  window:

    ```text
    Model 'h' has a `last` window, at `models.h.window`, so it gives one forecast and can't be backtested, but the run has backtests ['k']!
    ```

    Put such models in their own run, without backtests. A `last` window on a
    *backtest* is fine: it tests the last dates only.

- **One spec per input name.** `Run.from_spec` raises a `ValueError` if two
  input specs share a name but differ, e.g. in their `cf`.

## Sharing

Specs are frozen and compare by value. `Run.from_spec` builds one node per
distinct spec, so identical parts of separately written models are computed
once (D3). In the example above, writing the same `filter` in two
`filtered_historical` models on 500 and 750 dates gives one filter, computed
once.

## Writing a spec back

`spec.model_dump_json()` writes a spec as JSON, with every default spelled
out, and parses back to an equal spec. Use it to record exactly what ran.

```python
spec.model_dump_json(indent=2)
RunSpec.model_validate_json(spec.model_dump_json()) == spec  # True
```

## Specs in Python

Spec classes can also be built in Python, as data: a checked description of a
run, to turn into nodes with `Run.from_spec`, or to save. The class picks the
spec, so `type` tags can be left out, and a list of values is a tuple.

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
