# tail-lib

tail-lib measures market risk (volatility, value-at-risk) and backtests the
models that measure it. You describe models, run them on a returns series, and
get back each model's value-at-risk and its backtest results.

- **Many models in one run.** Historical simulation, filtered historical
  simulation and Gaussian models, side by side on the same returns.
- **Shared work.** A volatility read by three models is computed once.
- **Two ways in.** Build models in Python for notebooks and scripts, or
  write them as JSON specs, e.g. for a service.
- **Vectorized.** Returns can hold many series at once: every model runs on
  all of them together.

## Install

tail-lib needs Python 3.12 or later. It isn't on PyPI yet: install it from
its repository.

```sh
uv add git+https://github.com/maxime-filippini/tail-lib
```

## A first run

Simulate some returns, build two value-at-risk models at 99%, backtest them
with Kupiec's test, and read the results.

```python
import datetime

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.context import Context
from tail_lib.distributions import StudentT
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.simulation import GarchProcess
from tail_lib.simulation import Simulator
from tail_lib.time_series import Trailing

# 1. Returns: 1000 dates of GARCH returns with fat-tailed shocks.
simulator = Simulator(
    vol=GarchProcess.from_long_run_variance(
        alpha=0.08, beta=0.90, long_run_variance=0.01**2
    ),
    shocks=StudentT(df=5.0),
)
returns = simulator.simulate(1000, first_date=datetime.date(2020, 1, 1), rng=1).returns

# 2. Models, in a context: every backtest runs on every model.
with Context() as ctx:
    Kupiec(name="kupiec")
    HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250), name="hs_99")
    GaussianModel(cf=0.99, vol=EwmaVolatility(ConstantDecay(0.94)), name="gauss_99")

# 3. Results.
results = ctx.run(returns)
results.models["hs_99"]  # the VaR, a TimeSeries
results.backtests["hs_99"]["kupiec"].p_value[0]  # 0.853
```

The same run, written as a JSON spec:

```json
{
  "models": {
    "hs_99": {"type": "historical", "cf": 0.99,
              "window": {"type": "trailing", "length": 250}},
    "gauss_99": {"type": "gaussian", "cf": 0.99,
                 "vol": {"type": "ewma",
                         "decay": {"type": "constant", "value": 0.94}}}
  },
  "backtests": {"kupiec": {"type": "kupiec"}}
}
```

```python
from tail_lib.context import Context
from tail_lib.run import RunSpec

ctx = Context.from_spec(RunSpec.model_validate_json(text))
results = ctx.run(returns)
```

## Where to go next

<div class="grid cards" markdown>

- **[Concepts](guide/concepts.md)**: time series, timing, signs, nodes and
  windows. Read this first.
- **[Runs in Python](guide/python-runs.md)**: build models as Python objects
  and run them.
- **[Runs from specs](guide/specs.md)**: the JSON format, and its errors.
- **[Models](guide/models.md)**: every model, with its formula and
  parameters.
- **[Backtests](guide/backtests.md)**: overshootings and Kupiec's test, on
  the whole sample or on rolling windows.
- **[Simulation](guide/simulation.md)**: simulate returns to try models on.
- **[API reference](reference/index.md)**: every class and function, from
  the docstrings.

</div>
