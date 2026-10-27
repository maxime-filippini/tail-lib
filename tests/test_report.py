import re
from pathlib import Path

import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.context import Context
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.report.data import ReportData
from tail_lib.report.report import DATA_PLACEHOLDER
from tail_lib.report.report import Report
from tail_lib.time_series import TimeSeries
from tests.helpers import returns_series

_SLOT = re.compile(
    r'<script id="report-data" type="application/json">(.*?)</script>', re.DOTALL
)


def _report(title: str = "tail-lib report") -> Report:
    with Context(backtests={"kupiec": Kupiec()}) as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
    return Report(ctx, ctx.run(returns_series(n=100)), title=title)


def _saved_data(path: Path) -> ReportData:
    match = _SLOT.search(path.read_text(encoding="utf-8"))
    assert match is not None
    return ReportData.model_validate_json(match.group(1))


def test_a_saved_report_holds_its_data(tmp_path: Path) -> None:
    report = _report(title="Daily VaR")
    path = tmp_path / "report.html"

    report.save(path)

    assert _saved_data(path) == report.data
    assert report.data.metadata.title == "Daily VaR"
    assert DATA_PLACEHOLDER not in path.read_text(encoding="utf-8")


def test_text_in_the_data_can_not_close_the_script(tmp_path: Path) -> None:
    title = "</script><script>alert(1)</script>"
    report = _report(title=title)
    path = tmp_path / "report.html"

    report.save(path)

    assert title not in path.read_text(encoding="utf-8")
    assert _saved_data(path).metadata.title == title


def test_a_saved_report_loads_nothing_from_outside(tmp_path: Path) -> None:
    path = tmp_path / "report.html"

    _report().save(path)

    page = path.read_text(encoding="utf-8")
    assert not re.search(r"""\b(src|href)\s*=\s*["']?(https?:)?//""", page)


def test_a_report_takes_run_inputs_and_series_labels() -> None:
    returns = returns_series(n=100, n_assets=2)
    var_ext = TimeSeries(dates=returns.dates, values=np.full((100, 2), -0.02))
    with Context() as ctx:
        ValueAtRiskInput("var_ext", cf=0.99)
    results = ctx.run(returns, {"var_ext": var_ext})

    report = Report(ctx, results, series_labels=["a", "b"])

    assert report.data.series == ["a", "b"]
    assert report.data.outputs["models.var_ext"][0][0] == pytest.approx(-0.02)


def test_a_report_reads_the_results_and_bounds_without_computing(
    tmp_path: Path,
) -> None:
    n_calls = 0

    class _Counting(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:
            nonlocal n_calls
            n_calls += 1
            return super().compute(returns)

    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Counting(0.01), name="gauss")
    returns = returns_series(n=100)
    results = ctx.run(returns, lower=returns.dates[20], upper=returns.dates[60])

    report = Report(ctx, results)
    path = tmp_path / "bounded.html"
    report.save(path)

    assert n_calls == 1
    saved = _saved_data(path)
    assert saved.metadata.lower == str(returns.dates[20])
    assert saved.metadata.upper == str(returns.dates[60])
    assert saved == report.data


def test_a_report_rejects_results_of_another_context() -> None:
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
    with Context() as other:
        GaussianModel(cf=0.95, vol=ConstantVolatility(0.01), name="gauss")

    with pytest.raises(ValueError, match="results don't hold the nodes"):
        Report(ctx, other.run(returns_series(n=100)))
