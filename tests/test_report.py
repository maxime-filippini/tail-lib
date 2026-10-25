import re
from pathlib import Path

import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.report.data import ReportData
from tail_lib.report.report import DATA_PLACEHOLDER
from tail_lib.report.report import Report
from tail_lib.run import Run
from tail_lib.time_series import TimeSeries
from tests.helpers import returns_series

_SLOT = re.compile(
    r'<script id="report-data" type="application/json">(.*?)</script>', re.DOTALL
)


def _run() -> Run:
    return Run(
        {"gauss": GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.01))},
        {"kupiec": KupiecTest},
    )


def _saved_data(path: Path) -> ReportData:
    match = _SLOT.search(path.read_text(encoding="utf-8"))
    assert match is not None
    return ReportData.model_validate_json(match.group(1))


def test_a_saved_report_holds_its_data(tmp_path: Path) -> None:
    report = Report(_run(), returns_series(n=100), title="Daily VaR")
    path = tmp_path / "report.html"

    report.save(path)

    assert _saved_data(path) == report.data
    assert report.data.metadata.title == "Daily VaR"
    assert DATA_PLACEHOLDER not in path.read_text(encoding="utf-8")


def test_text_in_the_data_can_not_close_the_script(tmp_path: Path) -> None:
    title = "</script><script>alert(1)</script>"
    report = Report(_run(), returns_series(n=100), title=title)
    path = tmp_path / "report.html"

    report.save(path)

    assert title not in path.read_text(encoding="utf-8")
    assert _saved_data(path).metadata.title == title


def test_a_saved_report_loads_nothing_from_outside(tmp_path: Path) -> None:
    path = tmp_path / "report.html"

    Report(_run(), returns_series(n=100)).save(path)

    page = path.read_text(encoding="utf-8")
    assert not re.search(r"""\b(src|href)\s*=\s*["']?(https?:)?//""", page)


def test_a_report_takes_run_inputs_and_series_labels() -> None:
    returns = returns_series(n=100, n_assets=2)
    var_ext = TimeSeries(dates=returns.dates, values=np.full((100, 2), -0.02))
    run = Run({"ext": ValueAtRiskInput("var_ext", cf=0.99)})

    report = Report(run, returns, {"var_ext": var_ext}, series_labels=["a", "b"])

    assert report.data.series == ["a", "b"]
    assert report.data.outputs["models.ext"][0][0] == pytest.approx(-0.02)


def test_a_report_needs_the_run_inputs() -> None:
    run = Run({"ext": ValueAtRiskInput("var_ext", cf=0.99)})

    with pytest.raises(KeyError, match="Missing inputs"):
        Report(run, returns_series(n=100))
