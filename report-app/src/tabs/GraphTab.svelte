<script lang="ts">
  // The graph tab: every node of the run once, in layers (R19, D3).
  import { onMount } from "svelte";
  import { type ChartOption, init } from "../charts";
  import { boxLabel, boxTooltip } from "../graph/format";
  import { layout } from "../graph/layout";
  import type { ReportData } from "../report-data";

  // `onselect` gets the id of a clicked node; the returns and run inputs,
  // which aren't nodes, aren't selectable.
  let { data, onselect }: { data: ReportData; onselect: (id: string) => void } = $props();

  // Box size and spacing, in pixels before any zoom.
  const BOX = { width: 200, height: 64 };
  const GAP = { x: 30, y: 60 };

  // Border colour by role; the returns and run inputs are grey.
  const COLORS = {
    data: "#6e7781",
    preprocessor: "#0969da",
    model: "#1a7f37",
    overshootings: "#bc4c00",
    backtest: "#8250df",
  } as const;

  // Margins that keep whole boxes in view: positions are box centres.
  const PAD = { x: BOX.width / 2 + 10, y: BOX.height / 2 + 10 };

  const graph = $derived(layout(data));
  // The chart is sized to the graph at scale 1, as ECharts scales the
  // positions to fill the chart; a wide graph scrolls sideways.
  const size = $derived({
    width: (graph.width - 1) * (BOX.width + GAP.x) + 2 * PAD.x,
    height: (graph.nLayers - 1) * (BOX.height + GAP.y) + 2 * PAD.y,
  });

  function option(): ChartOption {
    return {
      tooltip: {
        confine: true,
        formatter: (params) => {
          const item = Array.isArray(params) ? params[0] : params;
          const value = item?.data as { id?: string; source?: string; target?: string; name?: string };
          if (item?.dataType === "edge") {
            return `<code>${value.source}</code> → <code>${value.target}</code><br>read as <b>${value.name}</b>`;
          }
          const box = graph.boxes.find((b) => b.id === value.id);
          return box === undefined ? "" : boxTooltip(box.id, box.node);
        },
      },
      series: [
        {
          type: "graph",
          layout: "none",
          roam: true,
          left: PAD.x,
          right: PAD.x,
          top: PAD.y,
          bottom: PAD.y,
          // A path with the box's own proportions: ECharts doesn't stretch
          // it, so its border is as thick on every side.
          symbol: `path://M0,0H${BOX.width}V${BOX.height}H0Z`,
          symbolSize: [BOX.width, BOX.height],
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
          lineStyle: { color: "#8c959f", width: 1.2, curveness: 0.15, opacity: 0.8 },
          emphasis: { focus: "adjacency", lineStyle: { width: 2 } },
          label: {
            show: true,
            color: "#1f2328",
            fontSize: 11,
            lineHeight: 15,
            width: BOX.width - 12,
            overflow: "truncate",
          },
          data: graph.boxes.map((box) => {
            const color = box.node === null ? COLORS.data : COLORS[box.node.kind];
            return {
              id: box.id,
              name: box.id,
              x: box.x * (BOX.width + GAP.x),
              y: box.layer * (BOX.height + GAP.y),
              label: { formatter: boxLabel(box.id, box.node, data) },
              itemStyle: { color: "#ffffff", borderColor: color, borderWidth: 2 },
            };
          }),
          edges: graph.arrows.map((arrow) => ({ ...arrow })),
        },
      ],
    };
  }

  let container: HTMLDivElement;

  onMount(() => {
    const chart = init(container);
    chart.setOption(option());
    chart.on("click", (params) => {
      const id = (params.data as { id?: string } | undefined)?.id;
      if (params.dataType === "node" && id !== undefined && graph.boxes.some((b) => b.id === id && b.node !== null)) {
        onselect(id);
      }
    });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  });
</script>

<p class="hint">
  Each node once, data flowing down. Hover a box for its settings and paths, an arrow
  for the name it is read under; click a box to chart its output. Scroll to zoom, drag
  to pan.
</p>
<div class="legend">
  {#each Object.entries(COLORS) as [kind, color] (kind)}
    <span><i style:border-color={color}></i>{kind === "data" ? "returns and inputs" : kind}</span>
  {/each}
</div>
<div class="scroll">
  <div
    class="chart"
    bind:this={container}
    style:min-width="{size.width}px"
    style:height="{size.height}px"
  ></div>
</div>

<style>
  .hint {
    color: #59636e;
    font-size: 0.9rem;
  }
  .legend {
    display: flex;
    gap: 1rem;
    font-size: 0.85rem;
  }
  .legend i {
    display: inline-block;
    width: 0.8rem;
    height: 0.8rem;
    margin-right: 0.3rem;
    border: 2px solid;
    border-radius: 3px;
    vertical-align: -0.15rem;
  }
  .scroll {
    overflow-x: auto;
  }
  .chart {
    width: 100%;
  }
</style>
