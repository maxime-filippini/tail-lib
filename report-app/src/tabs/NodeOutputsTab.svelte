<script lang="ts">
  // Node outputs: the output of one node of the run, for one series, picked
  // here or by clicking its box in the graph (R19).
  import Chart from "../Chart.svelte";
  import type { ChartOption } from "../charts";
  import { boxTooltip, displayName, settingsLine } from "../graph/format";
  import { percent } from "../models";
  import type { NodeData, ReportData } from "../report-data";

  let {
    data,
    series,
    selected = $bindable(),
  }: { data: ReportData; series: number; selected: string | null } = $props();

  // Every node with a time-series output: all but the backtests.
  const nodes = $derived(data.nodes.filter((node) => node.id in data.outputs));
  const node = $derived(
    nodes.find((n) => n.id === selected) ??
      nodes.find((n) => n.kind === "preprocessor") ??
      nodes[0],
  );

  // How a node's values read. Returns, volatilities, means and VaRs are
  // returns, in percent; the others are plain numbers.
  const PLAIN: Record<string, number> = {
    ConstantDecay: 4,
    EwmaDecayCalibration: 4,
    VolatilityFilter: 2,
    Overshootings: 0,
  };
  function format(n: NodeData, value: number | null): string {
    const digits = PLAIN[n.class_name];
    if (digits === undefined) return percent(value);
    return value === null ? "–" : value.toFixed(digits);
  }

  // Under the heading: the class, if the heading is a display name, then
  // the settings.
  function subtitle(n: NodeData): string {
    const parts = displayName(n, data) === null ? [] : [n.class_name];
    const settings = settingsLine(n);
    return [...parts, ...(settings === "" ? [] : [settings])].join(" · ");
  }

  function pickerLabel(n: NodeData): string {
    const name = displayName(n, data);
    return `${name === null ? "" : `${name} · `}${n.class_name} · ${n.id}`;
  }

  function option(n: NodeData): ChartOption {
    const values = data.outputs[n.id]?.[series] ?? [];
    return {
      animation: false,
      grid: { left: 80, right: 30, top: 20, bottom: 80 },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value) => format(n, value as number | null),
      },
      xAxis: { type: "time" },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { formatter: (value: number) => format(n, value) },
      },
      dataZoom: [
        { type: "inside", filterMode: "none" },
        { type: "slider", bottom: 20, filterMode: "none" },
      ],
      series: [
        {
          name: n.class_name,
          type: "line",
          symbol: "none",
          // Overshooting flags are 0 or 1 on each date: drawn as steps.
          step: n.kind === "overshootings" ? "middle" : false,
          data: values.map((value, t) => [data.dates[t] ?? "", value]),
          lineStyle: { width: 1.5, color: "#0969da" },
          itemStyle: { color: "#0969da" },
        },
      ],
    };
  }
</script>

{#if node === undefined}
  <p class="hint">The run has no node outputs.</p>
{:else}
  <label class="picker">
    Node
    <select value={node.id} onchange={(e) => (selected = e.currentTarget.value)}>
      {#each nodes as n (n.id)}
        <option value={n.id}>{pickerLabel(n)}</option>
      {/each}
    </select>
  </label>
  <p class="hint">
    Click a box in Graph to show its output here. The value on each date uses data up to
    and including that date: a VaR here is on the date it is computed, the day before the
    date it forecasts.
  </p>
  <h2>{displayName(node, data) ?? node.class_name}</h2>
  <p class="settings">{subtitle(node)}</p>
  <Chart option={option(node)} height={440} />
  <details>
    <summary>Settings and paths</summary>
    <!-- boxTooltip escapes every value it writes. -->
    <p class="details">{@html boxTooltip(node.id, node)}</p>
  </details>
{/if}

<style>
  .picker {
    display: block;
    margin-top: 1rem;
    font-size: 0.9rem;
  }
  .picker select {
    margin-left: 0.4rem;
    max-width: 100%;
    font: inherit;
  }
  .hint,
  .settings {
    color: #59636e;
    font-size: 0.9rem;
  }
  h2 {
    margin: 1rem 0 0;
    font-size: 1.1rem;
  }
  .settings {
    margin-top: 0.2rem;
  }
  details {
    margin: 0.5rem 0 1.5rem;
    font-size: 0.9rem;
  }
</style>
