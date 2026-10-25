import { mount } from "svelte";
import App from "./App.svelte";
import { loadData } from "./data";

const target = document.getElementById("app");
if (target === null) {
  throw new Error("The page has no #app element!");
}
mount(App, { target, props: { data: await loadData() } });
