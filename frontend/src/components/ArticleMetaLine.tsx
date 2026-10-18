import type { ArticleMeta } from "@/lib/api";
import { formatInt } from "@/lib/format";
import { sourceName } from "@/lib/sources";

import styles from "./ArticleMetaLine.module.css";

function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

export function ArticleMetaLine({ meta }: { meta: ArticleMeta }) {
  const source = sourceName(meta.source) ?? hostOf(meta.final_url);
  return (
    <section className={styles.meta} aria-label="Article">
      <h2 className={styles.title}>
        <a href={meta.final_url} target="_blank" rel="noopener noreferrer">
          {meta.title || meta.final_url}
        </a>
      </h2>
      <ul className={styles.facts}>
        <li>{source}</li>
        {meta.published_at ? <li>{meta.published_at}</li> : null}
        <li data-testid="article-extraction">
          extracted by <code>{meta.extraction_method}</code> ({meta.parser} / {meta.extractor})
        </li>
        <li>{formatInt(meta.word_count)} words</li>
        <li data-cached={meta.cached}>{meta.cached ? "cached" : "fresh fetch"}</li>
        <li>{meta.backend}</li>
      </ul>
    </section>
  );
}
