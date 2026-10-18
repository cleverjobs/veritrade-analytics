// Copies the canonical eval results into the app; falls back to the committed sample before evals exist.
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const canonical = resolve(here, "../../evals/results/benchmarks.json");
const sample = resolve(here, "../src/data/benchmarks.sample.json");
const target = resolve(here, "../src/data/benchmarks.json");

function isVersion1(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8")).version === 1;
  } catch {
    return false;
  }
}

let source = sample;
if (existsSync(canonical)) {
  if (isVersion1(canonical)) source = canonical;
  else console.warn(`benchmarks: ${canonical} is not a version 1 file; using the sample`);
}

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`benchmarks: ${source} -> ${target}`);
