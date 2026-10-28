// Builds the report page into one self-contained HTML file, written into the
// Python package, where `Report.save` reads it.
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [svelte(), viteSingleFile()],
  build: {
    outDir: "../src/tail_lib/report",
    // The output directory holds the Python modules too: keep them.
    emptyOutDir: false,
  },
});
