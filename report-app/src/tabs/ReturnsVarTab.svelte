<script lang="ts">
  // Returns vs VaR: the returns, each model's VaR and its overshootings, for
  // one series (R19).
  import Chart from "../Chart.svelte";
  import {
    type ChartOption,
    type LineSeriesOption,
    MODEL_COLORS,
    type ScatterSeriesOption,
  } from "../charts";
  import { modelEntries, percent } from "../models";
  import type { ReportData } from "../report-data";

  let { data, series }: { data: ReportData; series: number } = $props();

  const models = $derived(modelEntries(data));
  const hasOvershootings = $derived(models.some((m) => m.overshootingsId !== null));

  const dateIndex = $derived(new Map(data.dates.map((date, t) => [date, t])));

  function escape(text: string): string {
    return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  }

  // One row per line: the return, then each model's VaR, flagged when the
  // return overshoots it. The flags come from the data, not from the dots.
  function tooltip(params: unknown): string {
    const items = (Array.isArray(params) ? params : [params]) as {
      seriesType: string;
      seriesName: string;
      marker: string;
      value: [string, number | null];
    }[];
    const date = items[0]?.value[0] ?? "";
    const t = dateIndex.get(date);
    const overshot = new Set(
      models
        .filter((m) => m.overshootingsId !== null && t !== undefined)
        .filter((m) => data.outputs[m.overshootingsId!]?.[series]?.[t!] === 1)
        .map((m) => m.label),
    );
    const rows = items
      .filter((item) => item.seriesType === "line")
      .map((item) => {
        const flag = overshot.has(item.seriesName) ? " · <b>overshooting</b>" : "";
        return `${item.marker}${escape(item.seriesName)}: ${percent(item.value[1])}${flag}`;
      });
    return [date, ...rows].join("<br>");
  }

  // No point thinning (ECharts' `sampling`): it can drop the extreme
  // returns, which matter most here.
  // Every point carries its date, `[date, value]`, on a time axis, so the
  // lines, the dots and the tooltip always agree on the date.
  function option(): ChartOption {
    const returns = data.returns[series] ?? [];
    const dated = (values: (number | null)[]): [string, number | null][] =>
      values.map((value, t) => [data.dates[t] ?? "", value]);
    const lines: (LineSeriesOption | ScatterSeriesOption)[] = [
      {
        name: "Returns",
        type: "line",
        data: dated(returns),
        symbol: "none",
        lineStyle: { width: 1, color: "#8c959f" },
        itemStyle: { color: "#8c959f" },
        z: 1,
      },
    ];
    models.forEach((model, index) => {
      const color = MODEL_COLORS[index % MODEL_COLORS.length];
      const vars = data.outputs[model.nodeId]?.[series] ?? [];
      // The VaR on date t is the forecast for t + 1 (D1): draw it on the date
      // it forecasts, next to the return it is tested against.
      lines.push({
        name: model.label,
        type: "line",
        data: dated([null, ...vars.slice(0, -1)]),
        symbol: "none",
        connectNulls: false,
        lineStyle: { width: 1.5, color },
        itemStyle: { color },
        z: 2,
      });
      // Same name as the line, so the legend toggles both.
      if (model.overshootingsId !== null) {
        const flags = data.outputs[model.overshootingsId]?.[series] ?? [];
        lines.push({
          name: model.label,
          type: "scatter",
          data: flags.flatMap((flag, t) =>
            flag === 1 ? [[data.dates[t] ?? "", returns[t] ?? null]] : [],
          ),
          symbolSize: 7,
          itemStyle: { color, borderColor: "#ffffff", borderWidth: 1 },
          z: 3,
        });
      }
    });

    return {
      animation: false,
      grid: { left: 70, right: 30, top: 50, bottom: 80 },
      legend: { top: 10, type: "scroll" },
      tooltip: { trigger: "axis", formatter: tooltip },
      xAxis: { type: "time" },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { formatter: (value: number) => percent(value, 1) },
      },
      // Zooming keeps every point (`filterMode: "none"`): filtering them out
      // made the tooltip read the lines at the wrong date.
      dataZoom: [
        { type: "inside", filterMode: "none" },
        { type: "slider", bottom: 20, filterMode: "none" },
      ],
      series: lines,
    };
  }
</script>

<p class="hint">
  Each VaR is drawn on the date it forecasts: an overshooting is a return below the line on
  the same date.
  {#if hasOvershootings}
    Dots mark the overshootings.
  {:else}
    The run has no backtests, so no overshootings are marked.
  {/if}
  Click a model in the legend to hide it; scroll or use the slider to zoom.
</p>
<Chart option={option()} height={520} />

<style>
  .hint {
    color: #59636e;
    font-size: 0.9rem;
  }
</style>
