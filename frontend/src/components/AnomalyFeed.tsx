/**
 * Detected anomalies, most recent first.
 *
 * Severity is a *status* encoding, not a series one, so it uses the reserved
 * status palette and always ships with a text label -- never colour alone. Two of
 * the four status steps sit below 3:1 on a light surface, and the icon-plus-label
 * pairing is what makes them safe regardless.
 */

import { Link } from "react-router-dom";
import type { Anomaly, Severity } from "../lib/types";
import { formatDate, formatNumber, formatReturn } from "../lib/format";

const SEVERITY_STYLE: Record<Severity, { dot: string; text: string }> = {
  low: { dot: "bg-hairline-strong", text: "text-ink-muted" },
  medium: { dot: "bg-status-warning", text: "text-ink-secondary" },
  high: { dot: "bg-status-serious", text: "text-ink" },
  extreme: { dot: "bg-status-critical", text: "text-ink" },
};

const TYPE_LABEL: Record<string, string> = {
  return: "Return",
  volume: "Volume",
  volatility: "Volatility",
  gap: "Gap",
};

export function AnomalyFeed({ anomalies }: { anomalies: Anomaly[] }) {
  return (
    <ul className="divide-y divide-hairline">
      {anomalies.map((anomaly) => {
        const style = SEVERITY_STYLE[anomaly.severity];
        return (
          <li key={anomaly.id} className="py-3 first:pt-0 last:pb-0">
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
              <span aria-hidden className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
              <Link
                to={`/companies?symbol=${anomaly.symbol}`}
                className="focusable rounded text-sm font-medium text-ink hover:underline"
              >
                {anomaly.symbol}
              </Link>
              <span className="text-2xs text-ink-secondary">
                {TYPE_LABEL[anomaly.anomaly_type] ?? anomaly.anomaly_type}
              </span>
              {/* The severity word, always present beside the dot. */}
              <span className={`text-2xs uppercase tracking-wide ${style.text}`}>
                {anomaly.severity}
              </span>
              <span className="ml-auto text-2xs tabular text-ink-muted">
                {formatDate(anomaly.trade_date)}
              </span>
            </div>

            <div className="mt-1 flex flex-wrap items-baseline gap-x-3 text-2xs tabular text-ink-secondary">
              <span>
                {anomaly.anomaly_type === "return"
                  ? formatReturn(String(anomaly.score / 100))
                  : `${formatNumber(anomaly.score, 1)}σ`}
              </span>
              <span>confidence {Math.round(anomaly.confidence * 100)}%</span>
              {/* The method is shown because the two detectors are not
                  comparable, and a reader deserves to know which fired. */}
              <span className="text-ink-muted">
                {anomaly.method === "z_score" ? "robust z-score" : "isolation forest"}
              </span>
            </div>

            {anomaly.explanation && (
              <p className="mt-1.5 whitespace-pre-line text-2xs leading-relaxed text-ink-secondary">
                {anomaly.explanation}
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}
