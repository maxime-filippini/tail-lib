"""A run report: one self-contained HTML file showing a run (R19).

`Report` computes a run on the returns, gathers its `ReportData`, and saves
it in the report page, a single HTML file that opens offline. The page is
package data (`index.html`), built ahead of time, so saving a report needs
no Node or internet access. Its data slot,

    <script id="report-data" type="application/json">...</script>

holds a placeholder, `DATA_PLACEHOLDER`, that `save` replaces with the JSON
of the data. No templating library is involved.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from tail_lib.graph import Cache
from tail_lib.report.data import build_report_data

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from tail_lib.report.data import ReportData
    from tail_lib.run import Run
    from tail_lib.time_series import TimeSeries

DATA_PLACEHOLDER = "__TAIL_LIB_REPORT_DATA__"
"""The text the report page holds in its data slot, replaced by the JSON."""


class Report:
    """The report of a run on one returns series.

    The run is computed when the report is built, on a cache the report
    keeps to itself, so every node's output is available, not only the
    run's result.
    """

    def __init__(
        self,
        run: Run,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
        *,
        title: str = "tail-lib report",
        series_labels: Sequence[str] | None = None,
    ) -> None:
        """Compute the run and gather the data of its report.

        Args:
            run (Run): The run.
            returns (TimeSeries): The returns time series.
            inputs (Mapping[str, TimeSeries] | None): The run inputs, by name,
                as for `Run.run`.
            title (str): The title of the report.
            series_labels (Sequence[str] | None): A label per series of the
                returns, see `build_report_data`.

        Raises:
            KeyError: If an input the run reads is missing.
            ValueError: If there are not as many series labels as series.

        """
        cache = Cache(returns, inputs)
        self._data = build_report_data(
            run, cache, title=title, series_labels=series_labels
        )

    @property
    def data(self) -> ReportData:
        """The data the report shows."""
        return self._data

    def save(self, path: str | Path) -> None:
        """Save the report as one self-contained HTML file.

        Args:
            path (str | Path): The file to write, e.g. `report.html`.

        """
        Path(path).write_text(_html(self._data), encoding="utf-8")


def _html(data: ReportData) -> str:
    """Write the data into the report page.

    Args:
        data (ReportData): The data of the report.

    Returns:
        str: The HTML of the report page, holding the data.

    Raises:
        ValueError: If the page doesn't hold the placeholder exactly once.

    """
    page = resources.files("tail_lib.report").joinpath("index.html").read_text("utf-8")
    if page.count(DATA_PLACEHOLDER) != 1:
        msg = f"The report page has to hold {DATA_PLACEHOLDER} exactly once!"
        raise ValueError(msg)

    # In a <script> element, the text `</script>` would end the element early.
    # `<` is `<` escaped in JSON, so the data reads back the same.
    text = data.model_dump_json().replace("<", "\\u003c")
    return page.replace(DATA_PLACEHOLDER, text)
