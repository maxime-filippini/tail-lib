// The models of a report, with what the tabs need to chart them.
import type { ReportData } from "./report-data";

export interface ModelEntry {
  /** The model's name in the run, e.g. `gauss_99`. */
  name: string;
  /** Its display name, see `Run.labels`. */
  label: string;
  /** The id of its node, whose output is its VaR. */
  nodeId: string;
  /** The id of its overshootings node, or `null` in a run without backtests. */
  overshootingsId: string | null;
}

export function modelEntries(data: ReportData): ModelEntry[] {
  const kinds = new Map(data.nodes.map((node) => [node.id, node.kind]));
  return Object.entries(data.models).map(([name, nodeId]) => {
    // An overshootings node reads its model as `var`.
    const edge = data.edges.find(
      (e) =>
        e.source === nodeId && e.name === "var" && kinds.get(e.target) === "overshootings",
    );
    return {
      name,
      label: data.labels[name] ?? name,
      nodeId,
      overshootingsId: edge?.target ?? null,
    };
  });
}

/** A return or rate as a percentage, e.g. `-2.33%`. */
export function percent(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "–" : `${(value * 100).toFixed(digits)}%`;
}
