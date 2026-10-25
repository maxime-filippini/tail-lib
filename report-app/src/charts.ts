// ECharts, with only the parts the report uses, so the page stays small.
import { GraphChart, type GraphSeriesOption } from "echarts/charts";
import { TooltipComponent, type TooltipComponentOption } from "echarts/components";
import { type ComposeOption, use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

use([GraphChart, TooltipComponent, CanvasRenderer]);

export { init } from "echarts/core";
export type ChartOption = ComposeOption<GraphSeriesOption | TooltipComponentOption>;
