// Copies the canonical eval results into the app; falls back to the committed sample before evals exist.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const canonical = resolve(here, "../../evals/results/benchmarks.json");
const sample = resolve(here, "../src/data/benchmarks.sample.json");
const target = resolve(here, "../src/data/benchmarks.json");

mkdirSync(dirname(target), { recursive: true });
const source = existsSync(canonical) ? canonical : sample;
if (existsSync(source)) {
  copyFileSync(source, target);
  console.log(`benchmarks: ${source} -> ${target}`);
} else {
  console.warn("benchmarks: no results or sample found; benchmarks page will be empty");
}
