// ECharts, with only the parts the report uses, so the page stays small.
import {
  GraphChart,
  type GraphSeriesOption,
  LineChart,
  type LineSeriesOption,
  ScatterChart,
  type ScatterSeriesOption,
} from "echarts/charts";
import {
  DataZoomComponent,
  type DataZoomComponentOption,
  GridComponent,
  type GridComponentOption,
  LegendComponent,
  type LegendComponentOption,
  TooltipComponent,
  type TooltipComponentOption,
} from "echarts/components";
import { type ComposeOption, use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

use([
  GraphChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export { init } from "echarts/core";
export type { LineSeriesOption, ScatterSeriesOption };
export type ChartOption = ComposeOption<
  | GraphSeriesOption
  | LineSeriesOption
  | ScatterSeriesOption
  | DataZoomComponentOption
  | GridComponentOption
  | LegendComponentOption
  | TooltipComponentOption
>;

/** One colour per model, in order, reused across tabs. */
export const MODEL_COLORS = [
  "#0969da",
  "#1a7f37",
  "#bc4c00",
  "#8250df",
  "#bf3989",
  "#1b7c83",
  "#9a6700",
  "#cf222e",
];
