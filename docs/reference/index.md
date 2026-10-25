# API reference

Every public class and function, generated from the docstrings. Each model
states its formula, its timing (which dates feed the value on date `t`) and
its parameters.

| Page | Module | Contents |
|---|---|---|
| [Runs](run.md) | `tail_lib.run` | `Run`, `RunSpec`, `RunResult`, `RunGraph`: models and backtests in one run, and its graph. |
| [Specs](specs.md) | `tail_lib.specs` | The spec classes: models written as data. |
| [Value-at-risk models](value-at-risk.md) | `tail_lib.models.value_at_risk` | Historical, filtered historical, Gaussian, input. |
| [Volatility models](volatility.md) | `tail_lib.models.volatility` | Constant, EWMA, fitted EWMA, input, and the volatility filter. |
| [EWMA](ewma.md) | `tail_lib.models.ewma` | Decay factors, their calibration, EWMA starts and recursion. |
| [Mean models](mean.md) | `tail_lib.models.mean` | Zero and constant means. |
| [Backtests](backtests.md) | `tail_lib.backtests` | Overshootings and Kupiec's test. |
| [Windows](windows.md) | `tail_lib.windows` | `Trailing`, `Last` and `Windows`. |
| [Time series and calendars](time-series.md) | `tail_lib.time_series`, `tail_lib.calendar` | `TimeSeries`, `Calendar`. |
| [Simulation](simulation.md) | `tail_lib.simulation`, `tail_lib.distributions` | `Simulator`, its processes, and distributions. |
| [Node graph](graph.md) | `tail_lib.graph`, `tail_lib.build` | `Node`, `Cache`, `Pool`, `Builder`: how runs compute. |
| [Reports](report.md) | `tail_lib.report.data` | `ReportData`, `build_report_data`: a run's graph and results as one JSON document. |
