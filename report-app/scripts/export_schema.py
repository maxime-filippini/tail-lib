"""Write the JSON Schema of `ReportData`, which the page's types come from.

Run by `npm run types`, which then turns it into `src/report-data.d.ts`, so
the page and the Python side share one contract.
"""

import json
from pathlib import Path

from tail_lib.report.data import ReportData

SCHEMA = Path(__file__).parents[1] / "src" / "report-data.schema.json"

if __name__ == "__main__":
    # The page reads the JSON as written, so the schema of the serialized form.
    schema = ReportData.model_json_schema(mode="serialization")
    SCHEMA.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
