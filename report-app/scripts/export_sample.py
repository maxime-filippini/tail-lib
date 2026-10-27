"""Write sample report data, for working on the page with `npm run dev`.

Run by `npm run sample`. The run has every kind of node, a shared
volatility, two series, a run input, and whole-sample and rolling backtests.
"""

from pathlib import Path

import numpy as np

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.graph import Cache
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.report.data import build_report_data
from tail_lib.run import Run
from tail_lib.simulation import EwmaProcess
from tail_lib.simulation import Simulator
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing

SAMPLE = Path(__file__).parents[1] / "dev" / "sample-data.json"

if __name__ == "__main__":
    import datetime

    simulator = Simulator(vol=EwmaProcess(decay=np.array([0.94, 0.97]), start_vol=0.01))
    returns = simulator.simulate(
        1000, first_date=datetime.date(2022, 1, 3), batch_shape=(2,), rng=0
    ).returns
    var_ext = TimeSeries(dates=returns.dates, values=np.full((1000, 2), -0.025))

    vol = EwmaVolatility(ConstantDecay(0.94))
    fitted = FittedEwmaVolatility(EwmaDecayCalibration(Trailing(length=250, step=20)))
    vol_filter = VolatilityFilter(vol, mean=ConstantMean(0.0002))
    run = Run(
        {
            "gauss": GaussianModel(cf=0.99, vol=vol),
            "gauss_fitted": GaussianModel(cf=0.99, vol=fitted),
            "hs_250": HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250)),
            "fhs_250": FilteredHistoricalSimulationsModel(
                cf=0.99, filter=vol_filter, window=Trailing(length=250)
            ),
            "external": ValueAtRiskInput("var_ext", cf=0.99),
        },
        {
            "kupiec": KupiecTest,
            "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
        },
        labels={
            "gauss": "Gaussian, EWMA 0.94",
            "fhs_250": "Filtered HS, 250 days",
            "kupiec_250": "Kupiec, 250 days",
        },
    )

    data = build_report_data(
        run,
        Cache(returns, {"var_ext": var_ext}),
        title="Sample report",
        series_labels=["Equity", "Rates"],
    )
    SAMPLE.write_text(data.model_dump_json(), encoding="utf-8")
