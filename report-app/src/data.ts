// Reads the report data that `Report.save` wrote into the page (R19).
import type { ReportData } from "./report-data";

/**
 * Read the report data from the page's data slot.
 *
 * A saved report holds the JSON of its `ReportData`, an object. A page that
 * wasn't saved by `Report.save` still holds the placeholder: in development
 * (`npm run dev`), the sample data from `npm run sample` is read instead.
 * The placeholder is not written here, so it appears once in the built page.
 */
export async function loadData(): Promise<ReportData | null> {
  const text = document.getElementById("report-data")?.textContent?.trim() ?? "";
  if (text.startsWith("{")) {
    return JSON.parse(text) as ReportData;
  }
  if (import.meta.env.DEV) {
    const response = await fetch("/dev/sample-data.json");
    return response.ok ? ((await response.json()) as ReportData) : null;
  }
  return null;
}
