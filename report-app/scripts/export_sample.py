"""Write sample report data, for working on the page with `npm run dev`.

Run by `npm run sample`. The run has every kind of node, a shared
volatility, two series, a run input, and whole-sample and rolling backtests.
"""

from pathlib import Path

import numpy as np

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.context import Context
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.report.data import build_report_data
from tail_lib.simulation import EwmaProcess
from tail_lib.simulation import Simulator
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Trailing

SAMPLE = Path(__file__).parents[1] / "dev" / "sample-data.json"

if __name__ == "__main__":
    import datetime

    simulator = Simulator(vol=EwmaProcess(decay=np.array([0.94, 0.97]), start_vol=0.01))
    returns = simulator.simulate(
        1000, first_date=datetime.date(2022, 1, 3), batch_shape=(2,), rng=0
    ).returns
    var_ext = TimeSeries(dates=returns.dates, values=np.full((1000, 2), -0.025))

    with Context(
        labels={
            "gauss": "Gaussian, EWMA 0.94",
            "fhs_250": "Filtered HS, 250 days",
            "kupiec_250": "Kupiec, 250 days",
        },
    ) as ctx:
        Kupiec(name="kupiec")
        Kupiec(name="kupiec_250", window=Trailing(length=250))
        vol = EwmaVolatility(ConstantDecay(0.94))
        fitted = EwmaVolatility(EwmaDecayCalibration(Trailing(length=250, step=20)))
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        GaussianModel(cf=0.99, vol=fitted, name="gauss_fitted")
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250), name="hs_250")
        FilteredHistoricalSimulationsModel(
            cf=0.99,
            filter=VolatilityFilter(vol, mean=ConstantMean(0.0002)),
            window=Trailing(length=250),
            name="fhs_250",
        )
        ValueAtRiskInput("var_ext", cf=0.99)

    data = build_report_data(
        ctx,
        ctx.run(
            returns,
            {"var_ext": var_ext},
            lower=returns.dates[300],
            upper=returns.dates[900],
        ),
        title="Sample report",
        series_labels=["Equity", "Rates"],
    )
    SAMPLE.write_text(data.model_dump_json(), encoding="utf-8")
