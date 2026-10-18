import type { Metadata } from "next";

import { BenchmarksView } from "@/components/benchmarks/BenchmarksView";
import raw from "@/data/benchmarks.json";
import { isBenchmarksV1 } from "@/lib/benchmarks";

export const metadata: Metadata = { title: "Benchmarks" };

// Rendered at build time from the JSON that scripts/copy-benchmarks.mjs puts in src/data.
export default function BenchmarksPage() {
  const data: unknown = raw;
  if (!isBenchmarksV1(data)) {
    return (
      <>
        <h1>Benchmarks</h1>
        <p>
          No compatible benchmark results were bundled. Run the eval suite to produce{" "}
          <code>evals/results/benchmarks.json</code> (version 1) and rebuild.
        </p>
      </>
    );
  }
  return <BenchmarksView data={data} />;
}
