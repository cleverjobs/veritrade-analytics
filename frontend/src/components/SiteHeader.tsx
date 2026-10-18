"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import styles from "./SiteHeader.module.css";

const LINKS = [
  { href: "/", label: "Analyze" },
  { href: "/benchmarks/", label: "Benchmarks" },
] as const;

const normalize = (path: string) => path.replace(/\/+$/, "") || "/";

export function SiteHeader() {
  const current = normalize(usePathname() ?? "/");
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <Link href="/" className={styles.brand}>
          VeriTrade Analytics
        </Link>
        <nav aria-label="Primary">
          <ul className={styles.links}>
            {LINKS.map(({ href, label }) => (
              <li key={href}>
                <Link
                  href={href}
                  className={styles.link}
                  aria-current={normalize(href) === current ? "page" : undefined}
                >
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
