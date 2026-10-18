import type { ReactNode } from "react";

import styles from "./Benchmarks.module.css";

export interface Column<T> {
  header: string;
  /** Expanded header text for abbreviations, read by screen readers. */
  title?: string;
  numeric?: boolean;
  cell: (row: T) => ReactNode;
}

interface DataTableProps<T> {
  caption: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, index: number) => string;
  highlight?: (row: T) => boolean;
}

export function DataTable<T>({ caption, columns, rows, rowKey, highlight }: DataTableProps<T>) {
  return (
    // Focusable so keyboard users can scroll wide tables horizontally.
    <div className={styles.scroll} role="region" aria-label={caption} tabIndex={0}>
      <table className={styles.table}>
        <caption className={styles.caption}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.header} scope="col" className={col.numeric ? styles.num : undefined}>
                {col.title ? <abbr title={col.title}>{col.header}</abbr> : col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className={styles.emptyCell}>
                No rows yet.
              </td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr key={rowKey(row, i)} data-selected={highlight?.(row) || undefined}>
                {columns.map((col) => (
                  <td key={col.header} className={col.numeric ? styles.num : undefined}>
                    {col.cell(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
