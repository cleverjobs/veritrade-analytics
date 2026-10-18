import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/SiteHeader";

import "./globals.css";
import styles from "./layout.module.css";

export const metadata: Metadata = {
  title: { default: "VeriTrade Analytics", template: "%s | VeriTrade Analytics" },
  description: "Low-latency summary and Bullish/Bearish/Neutral signal for financial news articles.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className={styles.skip} href="#main">
          Skip to content
        </a>
        <SiteHeader />
        <main id="main" className={styles.main}>
          {children}
        </main>
      </body>
    </html>
  );
}
