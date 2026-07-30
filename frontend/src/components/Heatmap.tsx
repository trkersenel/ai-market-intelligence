/**
 * Cross-sectional returns as a diverging heatmap.
 *
 * Returns are *polarity* data -- signed, with a meaningful zero -- so the
 * encoding is a diverging scale: two hues that read as opposite, with a neutral
 * grey where the value is nothing.
 *
 * The poles are blue and red, not the conventional green and red. Green/red is
 * exactly the pair red-green colour blindness collapses, which makes the finance
 * convention the worst available choice for the one encoding a market screen
 * depends on most. Every cell also carries its number, so the colour is a second
 * channel rather than the only one.
 */

import { Link } from "react-router-dom";
import { formatPercent } from "../lib/format";

export interface HeatmapCell {
  symbol: string;
  /** Percentage change; null when the session has no comparable prior close. */
  changePercent: number | null;
  slug?: string | null;
}

/**
 * Colour for a cell, stepped from the diverging ramp.
 *
 * Discrete steps rather than a continuous gradient: past roughly seven classes
 * adjacent shades stop being distinguishable, so more resolution would be
 * decorative. Thresholds are in percent and chosen around what a semiconductor
 * session actually looks like -- a 2% move is ordinary here, 6% is not.
 */
function cellStyle(change: number | null): { background: string; ring: string } {
  if (change === null) return { background: "#222221", ring: "#2c2c2a" };

  const magnitude = Math.abs(change);
  const step: 0 | 1 | 2 | 3 = magnitude < 0.5 ? 0 : magnitude < 2 ? 1 : magnitude < 6 ? 2 : 3;

  // Sequential steps within each arm, from the palette's blue and red ramps.
  // A tuple keyed by step, so the lookup is exhaustive and nothing can be
  // undefined -- the reason no assertion is needed here.
  const ARMS = {
    up: { 0: "#383835", 1: "#1c5cab", 2: "#2a78d6", 3: "#3987e5" },
    down: { 0: "#383835", 1: "#8f2626", 2: "#b53030", 3: "#d03b3b" },
  } as const;

  const background = change > 0 ? ARMS.up[step] : ARMS.down[step];
  return { background, ring: background };
}

export function Heatmap({ cells }: { cells: HeatmapCell[] }) {
  return (
    <div>
      {/* Grid gaps give the 2px surface separation between fills that the mark
          spec calls for -- a border drawn around each cell would be heavier. */}
      <ul className="grid grid-cols-2 gap-0.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {cells.map((cell) => {
          const { background } = cellStyle(cell.changePercent);
          const content = (
            <>
              <span className="text-xs font-medium text-ink">{cell.symbol}</span>
              <span className="text-sm tabular text-ink">
                {formatPercent(cell.changePercent)}
              </span>
            </>
          );

          return (
            <li key={cell.symbol}>
              {cell.slug ? (
                <Link
                  to={`/companies/${cell.slug}`}
                  style={{ background }}
                  title={`${cell.symbol} ${formatPercent(cell.changePercent)}`}
                  className="focusable flex h-16 flex-col justify-between rounded p-2 transition-opacity hover:opacity-80"
                >
                  {content}
                </Link>
              ) : (
                <div
                  style={{ background }}
                  title={`${cell.symbol} ${formatPercent(cell.changePercent)}`}
                  className="flex h-16 flex-col justify-between rounded p-2"
                >
                  {content}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <Legend />
    </div>
  );
}

/** A scale legend, required because colour carries magnitude here. */
function Legend() {
  const steps = [
    { background: "#d03b3b", label: "≤ −6%" },
    { background: "#b53030", label: "−6 to −2" },
    { background: "#8f2626", label: "−2 to −0.5" },
    { background: "#383835", label: "flat" },
    { background: "#1c5cab", label: "+0.5 to 2" },
    { background: "#2a78d6", label: "+2 to 6" },
    { background: "#3987e5", label: "≥ +6%" },
  ];

  return (
    <ul className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1">
      {steps.map((step) => (
        <li key={step.label} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-2.5 w-2.5 rounded-sm"
            style={{ background: step.background }}
          />
          <span className="text-2xs tabular text-ink-muted">{step.label}</span>
        </li>
      ))}
    </ul>
  );
}
