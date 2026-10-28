// Places the nodes of a run's graph in layers, for the graph tab.
//
// Data flows down: the returns and the run inputs at the top, then the
// preprocessors, the models, their overshootings and the backtests. A
// preprocessor sits one layer below the deepest preprocessor it reads, e.g.
// a decay calibration above the volatility that uses it.
import type { NodeData, ReportData } from "../report-data";

/** A box of the graph: a node of the run, the returns, or a run input. */
export interface Box {
  id: string;
  /** The node, or `null` for the returns and the run inputs. */
  node: NodeData | null;
  layer: number;
  /** Position in the layer, in box widths, centred on 0. */
  x: number;
}

/** An arrow of the graph, from the box read to the box reading it. */
export interface Arrow {
  source: string;
  target: string;
  /** The name it is read under, e.g. `vol`. */
  name: string;
}

export interface Layout {
  boxes: Box[];
  arrows: Arrow[];
  nLayers: number;
  /** The number of boxes in the widest layer. */
  width: number;
}

export const RETURNS_ID = "returns";

/** The id of the box of a run input, e.g. `input:var_ext`. */
export function inputId(name: string): string {
  return `input:${name}`;
}

export function layout(data: ReportData): Layout {
  const nodes = new Map(data.nodes.map((node) => [node.id, node]));

  // Each node's preprocessor depth: 0 if it reads no preprocessor, else one
  // more than the deepest preprocessor it reads. The graph has no cycles.
  const sources = new Map<string, string[]>();
  for (const edge of data.edges) {
    sources.set(edge.target, [...(sources.get(edge.target) ?? []), edge.source]);
  }
  const depths = new Map<string, number>();
  const depth = (id: string): number => {
    const known = depths.get(id);
    if (known !== undefined) return known;
    const preprocessors = (sources.get(id) ?? []).filter(
      (source) => nodes.get(source)?.kind === "preprocessor",
    );
    const value = Math.max(-1, ...preprocessors.map(depth)) + 1;
    depths.set(id, value);
    return value;
  };

  // Layer 0 holds the returns and the run inputs, then the preprocessors by
  // depth, then one layer each for the models, overshootings and backtests.
  const preprocessorLayers =
    Math.max(-1, ...data.nodes.filter((n) => n.kind === "preprocessor").map((n) => depth(n.id))) + 1;
  const firstAfterPreprocessors = 1 + preprocessorLayers;
  const layerOf = (node: NodeData): number => {
    switch (node.kind) {
      case "preprocessor":
        return 1 + depth(node.id);
      case "model":
        return firstAfterPreprocessors;
      case "overshootings":
        return firstAfterPreprocessors + 1;
      case "backtest":
        return firstAfterPreprocessors + 2;
    }
  };

  // The returns and the run inputs are boxes too. A run input has an arrow
  // to each node reading it; the returns, read by most nodes, have none.
  const layers: string[][] = [[RETURNS_ID]];
  const layerIndex = new Map<string, number>([[RETURNS_ID, 0]]);
  const arrows: Arrow[] = data.edges.map(({ source, target, name }) => ({ source, target, name }));
  for (const node of data.nodes) {
    if (node.input_name === null) continue;
    const id = inputId(node.input_name);
    if (!layerIndex.has(id)) {
      layerIndex.set(id, 0);
      layers[0]!.push(id);
    }
    arrows.push({ source: id, target: node.id, name: node.input_name });
  }
  for (const node of data.nodes) {
    const layer = layerOf(node);
    layerIndex.set(node.id, layer);
    while (layers.length <= layer) layers.push([]);
    layers[layer]!.push(node.id);
  }

  // Each layer is centred. Order it by the mean position of the boxes it
  // reads in the layers above (one top-down pass), so arrows cross less.
  // Boxes reading nothing above keep their place in the graph's order.
  const xs = new Map<string, number>();
  layers.forEach((ids, layer) => {
    const centred = (index: number): number => index - (ids.length - 1) / 2;
    const key = (id: string, index: number): number => {
      const above = arrows
        .filter((a) => a.target === id && (layerIndex.get(a.source) ?? layer) < layer)
        .map((a) => xs.get(a.source) ?? 0);
      return above.length === 0
        ? centred(index)
        : above.reduce((sum, x) => sum + x, 0) / above.length;
    };
    const keys = new Map(ids.map((id, index) => [id, key(id, index)]));
    ids.sort((a, b) => keys.get(a)! - keys.get(b)!);
    ids.forEach((id, index) => xs.set(id, centred(index)));
  });

  const boxes = layers.flatMap((ids, layer) =>
    ids.map((id) => ({ id, node: nodes.get(id) ?? null, layer, x: xs.get(id)! })),
  );
  return {
    boxes,
    arrows,
    nLayers: layers.length,
    width: Math.max(...layers.map((ids) => ids.length)),
  };
}
