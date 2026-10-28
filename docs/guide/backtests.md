# Backtests

A backtest checks a model's VaR forecasts against the returns that followed.
In a run, every backtest runs on every model, so models compare on
the same tests.

## Overshootings

Backtests start from the model's **overshootings**: the dates whose return
fell below the previous date's VaR.

\[
I_t = \begin{cases}
1 & \text{if } r_t < \mathrm{VaR}_{t-1} \\
0 & \text{otherwise}
\end{cases}
\]

The flags are a `TimeSeries` on the dates of the returns. They are `NaN` on
the first date, and where the return or the previous VaR is `NaN`, e.g.
during a model's warm-up. A return equal to the VaR is not an overshooting.

In a run, each model gets one
[`Overshootings`][tail_lib.backtests.overshootings.Overshootings] node, shared
by all its backtests. To flag a VaR series by hand, use the
[`overshootings(var, returns)`][tail_lib.backtests.overshootings.overshootings]
function.

## Kupiec's test

[`Kupiec`][tail_lib.backtests.kupiec.Kupiec] is Kupiec's (1995)
proportion-of-failures test: does the overshooting rate match the model's
confidence level? Under the model, each date overshoots with probability
\(p = 1 - c\). With \(x\) overshootings out of \(n\) flagged dates, the
likelihood ratio

\[
\mathrm{LR} = -2 \left[ (n - x)\ln(1 - p) + x \ln p
  - (n - x)\ln\left(1 - \tfrac{x}{n}\right) - x \ln \tfrac{x}{n} \right]
\]

follows a \(\chi^2\) distribution with 1 degree of freedom. A small p-value
rejects the model. The test counts overshootings only: it doesn't look at
when they happen.

The confidence level comes from the model (`cf`), so the test has a single
setting, `window`:

| `window` | Tests | Results |
|---|---|---|
| `None` (default) | the whole sample | one window, ending on the last date |
| `Trailing(length, step)` | rolling windows of `length` flags | one result per window |
| `Last(length)` | the last `length` flags only | one window, ending on the last date |

Windows start on the first flag, so no window holds a model's warm-up.

=== "Python"

    ```python
    from tail_lib.backtests.kupiec import Kupiec
    from tail_lib.context import Context
    from tail_lib.time_series import Last, Trailing

    with Context() as ctx:
        Kupiec(name="kupiec")
        Kupiec(name="kupiec_250", window=Trailing(length=250))
        Kupiec(name="kupiec_last", window=Last(length=250))
        ...  # the models
    ```

=== "Spec"

    ```json
    "backtests": {
      "kupiec": {"type": "kupiec"},
      "kupiec_250": {"type": "kupiec",
                     "window": {"type": "trailing", "length": 250}},
      "kupiec_last": {"type": "kupiec",
                      "window": {"type": "last", "length": 250}}
    }
    ```

## Reading a result

A [`KupiecResult`][tail_lib.backtests.kupiec.KupiecResult] has one value per
window and per series: every array has the shape `(n_windows, *batch)`.

| Field | Meaning |
|---|---|
| `end_dates` | The last date of each window, which its result belongs to. |
| `n_obs` | Number of flagged dates (not `NaN`) in the window. |
| `n_overshootings` | Number of overshootings. |
| `expected_rate` | \(1 - c\), the rate under the model. |
| `observed_rate` | `n_overshootings / n_obs`. |
| `statistic` | The likelihood ratio. |
| `p_value` | Probability of a statistic at least as large under the model. |

A whole-sample test is one window, hence the `[0]`:

```python
test = result.backtests["hs_99"]["kupiec"]
test.n_obs[0], test.n_overshootings[0], test.p_value[0]  # 750, 7, 0.853
```

A rolling test has one result per window, from the first full window:

```python
rolling = result.backtests["hs_99"]["kupiec_250"]
rejected = rolling.p_value < 0.05
share_rejected = rejected.mean()
rolling.end_dates[rejected]  # when the model was rejected
```

The window ending on date \(t\) holds the flags of dates
\(t - \text{length} + 1\) to \(t\): it compares the VaR up to \(t - 1\) with
the returns up to \(t\).

## Comparing models on the same windows

Models have different warm-ups, so their rolling tests start on different
dates. To compare them, keep the windows they all have:

```python
import numpy as np

tests = {name: bt["kupiec_250"] for name, bt in result.backtests.items()}
common = sorted(set.intersection(*(set(t.end_dates) for t in tests.values())))
for name, test in tests.items():
    keep = np.isin(test.end_dates, common)
    print(name, (test.p_value[keep] < 0.05).mean())
```

`examples/backtest_hs_fhs.py` in the repository runs this comparison on
historical, filtered historical and Gaussian models.

## Without a run

[`kupiec_pof(overshootings, cf_var, window=None)`][tail_lib.backtests.kupiec.kupiec_pof]
runs the test on flags you already have:

```python
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import overshootings

flags = overshootings(var, returns)
kupiec_pof(flags, cf_var=0.99, window=Trailing(length=250))
```
