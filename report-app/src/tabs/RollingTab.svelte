<script lang="ts">
  // Rolling backtests: for each backtest on rolling windows, the observed
  // overshooting rate and the p-value at each window's end date, one line
  // per model, for one series.
  import Chart from "../Chart.svelte";
  import {
    type ChartOption,
    type LineSeriesOption,
    MODEL_COLORS,
  } from "../charts";
  import { modelEntries, percent } from "../models";
  import type { ReportData } from "../report-data";

  let { data, series, level }: { data: ReportData; series: number; level: number } =
    $props();

  const models = $derived(modelEntries(data));

  // The backtests with several windows, by name. Every backtest runs on
  // every model, so the first model's windows tell.
  const rolling = $derived(
    Object.entries(data.backtests[models[0]?.name ?? ""] ?? {})
      .filter(([, backtest]) => backtest.result.end_dates.length > 1)
      .map(([name]) => name),
  );

  function escape(text: string): string {
    return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  }

  function tooltip(params: unknown): string {
    const items = (Array.isArray(params) ? params : [params]) as {
      seriesId: string;
      seriesName: string;
      marker: string;
      value: [string, number | null];
    }[];
    const rows = items
      .filter((item) => item.seriesId.startsWith("rate:") || item.seriesId.startsWith("p:"))
      .map((item) => {
        const value = item.value[1];
        const text = item.seriesId.startsWith("rate:")
          ? `rate ${percent(value)}`
          : `p-value ${value === null ? "–" : value < 0.001 ? "<0.001" : value.toFixed(3)}`;
        return `${item.marker}${escape(item.seriesName)}: ${text}`;
      });
    return [items[0]?.value[0] ?? "", ...rows].join("<br>");
  }

  function option(name: string): ChartOption {
    const lines: LineSeriesOption[] = [];
    let first = "";
    let last = "";
    models.forEach((model, index) => {
      const result = data.backtests[model.name]?.[name]?.result;
      if (result === undefined) return;
      const color = MODEL_COLORS[index % MODEL_COLORS.length];
      const dated = (values: (number | null)[][]): [string, number | null][] =>
        (values[series] ?? []).map((value, w) => [result.end_dates[w] ?? "", value]);
      first = first === "" || result.end_dates[0]! < first ? result.end_dates[0]! : first;
      last = result.end_dates.at(-1)! > last ? result.end_dates.at(-1)! : last;
      const common = { name: model.label, type: "line", symbol: "none" } as const;
      lines.push(
        {
          ...common,
          id: `rate:${model.name}`,
          data: dated(result.observed_rate),
          lineStyle: { width: 1.5, color },
          itemStyle: { color },
        },
        // The rate expected under the model, `1 - cf`, dashed.
        {
          ...common,
          id: `expected:${model.name}`,
          data: [
            [result.end_dates[0] ?? "", result.expected_rate],
            [result.end_dates.at(-1) ?? "", result.expected_rate],
          ],
          lineStyle: { width: 1, color, type: "dashed" },
          itemStyle: { color },
        },
        {
          ...common,
          id: `p:${model.name}`,
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: dated(result.p_value),
          lineStyle: { width: 1.5, color },
          itemStyle: { color },
        },
      );
    });
    lines.push({
      id: "level",
      name: "Rejection level",
      type: "line",
      symbol: "none",
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [
        [first, level / 100],
        [last, level / 100],
      ],
      lineStyle: { width: 1, color: "#cf222e", type: "dashed" },
      itemStyle: { color: "#cf222e" },
    });

    const axisLabel = { formatter: (value: number) => percent(value, 1) };
    return {
      animation: false,
      legend: { top: 0, type: "scroll" },
      tooltip: { trigger: "axis", formatter: tooltip },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: [
        { left: 70, right: 30, top: 40, height: 170 },
        { left: 70, right: 30, top: 260, height: 170 },
      ],
      xAxis: [
        { type: "time", gridIndex: 0, axisLabel: { show: false } },
        { type: "time", gridIndex: 1 },
      ],
      yAxis: [
        { type: "value", gridIndex: 0, name: "Observed rate", axisLabel },
        { type: "value", gridIndex: 1, name: "p-value", min: 0, max: 1 },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1], filterMode: "none" },
        { type: "slider", xAxisIndex: [0, 1], bottom: 10, filterMode: "none" },
      ],
      series: lines,
    };
  }
</script>

{#if rolling.length === 0}
  <p class="hint">The run has no backtests on rolling windows.</p>
{:else}
  <p class="hint">
    Each point is one window, on its last date. Top: the observed overshooting rate, with
    the rate expected under each model dashed. Bottom: the p-value, with the rejection level
    ({level}%, set in Backtests) dashed in red.
  </p>
  {#each rolling as name (name)}
    <h2>{data.labels[name] ?? name}</h2>
    <Chart option={option(name)} height={510} />
  {/each}
{/if}

<style>
  .hint {
    color: #59636e;
    font-size: 0.9rem;
  }
  h2 {
    margin: 1.25rem 0 0.25rem;
    font-size: 1.1rem;
  }
</style>
