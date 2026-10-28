// Labels and hover details of the graph's boxes.
import type { NodeData, ReportData } from "../report-data";
import { RETURNS_ID } from "./layout";

/** A setting in a few characters, e.g. `trailing 250` or `[0.94, 0.97]`. */
export function shortSetting(value: unknown): string {
  if (value === null) return "none";
  if (Array.isArray(value)) return `[${value.map(shortSetting).join(", ")}]`;
  if (typeof value === "object") {
    // Windows and EWMA starts, written as in the specs.
    const { type, ...rest } = value as { type?: string } & Record<string, unknown>;
    const fields = Object.entries(rest)
      .filter(([name, field]) => !(name === "step" && field === 1))
      .map(([, field]) => shortSetting(field));
    return [type, ...fields].filter((part) => part !== undefined).join(" ");
  }
  return String(value);
}

/** The settings of a node on one line, e.g. `cf 0.99 · window trailing 250`. */
export function settingsLine(node: NodeData): string {
  return Object.entries(node.settings)
    .map(([name, value]) => `${name} ${shortSetting(value)}`)
    .join(" · ");
}

/**
 * The display name of a model or backtest node, from the run's labels.
 *
 * A model is named by its first path, `models.<name>`; a backtest by
 * `backtests.<model>.<name>`. Other nodes have none.
 */
export function displayName(node: NodeData, data: ReportData): string | null {
  const parts = node.id.split(".");
  const name =
    node.kind === "model" ? parts[1] : node.kind === "backtest" ? parts[2] : undefined;
  return name === undefined ? null : (data.labels[name] ?? name);
}

/** The label of a box: its name or class, then its settings. */
export function boxLabel(id: string, node: NodeData | null, data: ReportData): string {
  if (node === null) {
    return id === RETURNS_ID ? "Returns" : `Input\n${id.slice("input:".length)}`;
  }
  const name = displayName(node, data);
  const title = name === null ? node.class_name : `${name}\n${node.class_name}`;
  const settings = settingsLine(node);
  return settings === "" ? title : `${title}\n${settings}`;
}

function escape(text: string): string {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

/** The hover details of a box, as HTML: id, every path and full settings. */
export function boxTooltip(id: string, node: NodeData | null): string {
  if (node === null) {
    return id === RETURNS_ID
      ? "The returns, read by most nodes."
      : `Run input <b>${escape(id.slice("input:".length))}</b>, passed in by name.`;
  }
  const lines = [`<b>${escape(node.class_name)}</b> (${node.kind})`];
  for (const [name, value] of Object.entries(node.settings)) {
    lines.push(`${escape(name)}: <code>${escape(JSON.stringify(value))}</code>`);
  }
  if (node.input_name !== null) {
    lines.push(`reads run input <b>${escape(node.input_name)}</b>`);
  }
  lines.push("", "Paths:", ...node.paths.map((path) => `<code>${escape(path)}</code>`));
  return lines.join("<br>");
}
