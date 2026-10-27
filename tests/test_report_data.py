import json

import numpy as np
import pytest

from tail_lib._typing import FloatArrayAtLeast1D
from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.graph import Cache
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import StartingVol
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.report.data import ReportData
from tail_lib.report.data import build_report_data
from tail_lib.run import Run
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series

_N_DATES = 300
_N_SERIES = 2


def _run() -> Run:
    vol = EwmaVolatility(ConstantDecay(0.94), start=Backcast(window=20))
    return Run(
        {
            "gauss": GaussianModel(cf=0.99, vol=vol),
            "fhs": FilteredHistoricalSimulationsModel(
                cf=0.99, filter=VolatilityFilter(vol), window=Trailing(length=100)
            ),
            "hs": HistoricalSimulationsModel(
                cf=0.975, window=Trailing(length=100, step=5)
            ),
            "ext": ValueAtRiskInput("var_ext", cf=0.99),
        },
        {
            "kupiec": KupiecTest,
            "kupiec_50": lambda ovs: KupiecTest(ovs, window=Trailing(length=50)),
        },
        labels={"gauss": "Gaussian 99%"},
    )


def _data(
    title: str = "tail-lib report", series_labels: list[str] | None = None
) -> ReportData:
    returns = returns_series(n=_N_DATES, n_assets=2)
    var_ext = TimeSeries(dates=returns.dates, values=np.full((_N_DATES, 2), -0.02))
    cache = Cache(returns, {"var_ext": var_ext})
    return build_report_data(_run(), cache, title=title, series_labels=series_labels)


def test_nodes_are_identified_by_their_first_path() -> None:
    data = _data()
    nodes = {node.id: node for node in data.nodes}

    vol = nodes["models.gauss.vol"]
    assert vol.class_name == "EwmaVolatility"
    assert vol.paths[1] == "models.fhs.filter.vol"
    assert data.models == {
        "gauss": "models.gauss",
        "fhs": "models.fhs",
        "hs": "models.hs",
        "ext": "models.ext",
    }
    assert len(nodes) == len(data.nodes)


def test_nodes_have_their_kind_and_input() -> None:
    nodes = {node.id: node for node in _data().nodes}

    assert nodes["models.gauss"].kind == "model"
    assert nodes["models.gauss.vol"].kind == "preprocessor"
    assert nodes["backtests.gauss.kupiec.overshootings"].kind == "overshootings"
    assert nodes["backtests.gauss.kupiec"].kind == "backtest"
    assert nodes["models.ext"].input_name == "var_ext"
    assert nodes["models.gauss"].input_name is None


def test_settings_are_written_like_the_specs() -> None:
    nodes = {node.id: node for node in _data().nodes}

    assert nodes["models.hs"].settings == {
        "cf": 0.975,
        "window": {"type": "trailing", "length": 100, "step": 5},
        "interpolation": "linear",
    }
    assert nodes["models.gauss.vol"].settings == {
        "start": {"type": "backcast", "window": 20}
    }
    assert nodes["models.gauss.vol.decay"].settings == {"value": 0.94}
    assert nodes["backtests.gauss.kupiec"].settings == {"window": None}


def test_edges_link_node_ids() -> None:
    data = _data()
    ids = {node.id for node in data.nodes}

    assert all(edge.source in ids and edge.target in ids for edge in data.edges)
    assert any(
        (edge.source, edge.target, edge.name)
        == ("models.gauss.vol", "models.gauss", "vol")
        for edge in data.edges
    )


def test_outputs_hold_every_time_series_node_by_series_then_date() -> None:
    data = _data()

    time_series_nodes = {node.id for node in data.nodes if node.kind != "backtest"}
    assert set(data.outputs) == time_series_nodes
    vol = data.outputs["models.gauss.vol"]
    assert len(vol) == _N_SERIES
    assert len(vol[0]) == _N_DATES
    # The backcast's warm-up is NaN, written as None.
    assert vol[0][0] is None
    assert vol[0][-1] is not None


def test_returns_dates_and_series_are_flattened() -> None:
    data = _data()
    returns = returns_series(n=_N_DATES, n_assets=2)

    assert data.dates[0] == "2020-01-01"
    assert len(data.dates) == _N_DATES
    assert data.series == ["0", "1"]
    assert data.returns[1] == returns.values[:, 1].tolist()


def test_backtests_hold_their_node_and_windows() -> None:
    data = _data()

    whole = data.backtests["gauss"]["kupiec"]
    rolling = data.backtests["gauss"]["kupiec_50"]
    assert whole.node == "backtests.gauss.kupiec"
    assert len(whole.result.end_dates) == 1
    assert len(rolling.result.end_dates) == len(rolling.result.p_value[0]) > 1
    assert len(rolling.result.p_value) == _N_SERIES
    assert rolling.result.expected_rate == pytest.approx(0.01)


def test_labels_and_metadata() -> None:
    data = _data(title="Daily VaR")

    assert data.labels["gauss"] == "Gaussian 99%"
    assert data.labels["kupiec"] == "kupiec"
    assert data.metadata.title == "Daily VaR"
    assert data.metadata.version


def test_series_labels_can_be_given() -> None:
    assert _data(series_labels=["SPX", "SX5E"]).series == ["SPX", "SX5E"]

    with pytest.raises(ValueError, match="Got 1 series labels for 2 series"):
        _data(series_labels=["SPX"])


@pytest.mark.parametrize(
    ("values", "labels"),
    [
        (np.full((40,), 0.001), ["0"]),
        (np.full((40, 2, 2), 0.001), ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]),
    ],
)
def test_series_are_labelled_by_their_index(
    values: FloatArrayAtLeast1D, labels: list[str]
) -> None:
    returns = TimeSeries.from_first_date(FIRST_DATE, values=values)
    run = Run({"vol": GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))})

    data = build_report_data(run, Cache(returns))

    assert data.series == labels
    assert len(data.returns) == len(labels)


def test_the_json_has_null_for_nan_and_reads_back() -> None:
    data = _data()

    text = data.model_dump_json()

    assert "NaN" not in text
    assert json.loads(text)["outputs"]["models.gauss.vol"][0][0] is None
    assert ReportData.model_validate_json(text) == data


def test_bounds_starts_and_arrays_are_written_as_json() -> None:
    calibration = EwmaDecayCalibration(
        Last(length=100), bounds=(0.9, 0.99), start=StartingVol(np.array([0.01, 0.02]))
    )
    decay = ConstantDecay(np.array([0.94, 0.97]))
    run = Run(
        {
            "fitted": GaussianModel(
                cf=0.99,
                vol=EwmaVolatility(calibration, start=calibration.start),
            ),
            "ewma": GaussianModel(cf=0.99, vol=EwmaVolatility(decay)),
        }
    )

    data = build_report_data(run, Cache(returns_series(n=150, n_assets=2)))
    nodes = {node.id: node for node in data.nodes}

    assert nodes["models.fitted.vol.decay"].settings == {
        "window": {"type": "last", "length": 100},
        "bounds": [0.9, 0.99],
        "start": {"type": "starting_vol", "vol": [0.01, 0.02]},
        "n_iter": 40,
    }
    assert nodes["models.ewma.vol.decay"].settings == {"value": [0.94, 0.97]}
