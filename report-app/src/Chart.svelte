<script lang="ts">
  // An ECharts chart that redraws when its option changes, and follows its
  // container's size.
  import { onMount } from "svelte";
  import { type ChartOption, init } from "./charts";

  let { option, height }: { option: ChartOption; height: number } = $props();

  let container: HTMLDivElement;
  let chart: ReturnType<typeof init> | undefined = $state();

  onMount(() => {
    chart = init(container);
    const observer = new ResizeObserver(() => chart?.resize());
    observer.observe(container);
    return () => {
      observer.disconnect();
      chart?.dispose();
    };
  });

  $effect(() => {
    // Replace the whole option, so series removed from it disappear.
    chart?.setOption(option, { notMerge: true });
  });
</script>

<div bind:this={container} style:height="{height}px"></div>
