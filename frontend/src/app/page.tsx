import { AnalyzeApp } from "@/components/AnalyzeApp";

import styles from "./page.module.css";

export default function AnalyzePage() {
  return (
    <>
      <header className={styles.intro}>
        <h1>Analyze a news article</h1>
        <p>
          Paste a public financial-news URL. The service fetches and extracts the article, then a local model
          returns a Bullish, Bearish or Neutral signal and a short summary.
        </p>
      </header>
      <AnalyzeApp />
    </>
  );
}
