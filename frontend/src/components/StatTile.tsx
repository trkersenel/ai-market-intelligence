/**
 * A single number with its label.
 *
 * The data-viz rule this exists to honour: when the story is one number, a chart
 * of one bar is the wrong form. A tile says it directly.
 */

import type { ReactNode } from "react";
import { polarity } from "../lib/format";

const POLARITY_INK = {
  up: "text-up",
  down: "text-down",
  flat: "text-ink",
} as const;

export function StatTile({
  label,
  value,
  delta,
  hint,
  emphasis = false,
}: {
  label: string;
  value: ReactNode;
  /** A signed change, coloured on the diverging scale. */
  delta?: string | number | null;
  hint?: ReactNode;
  emphasis?: boolean;
}) {
  const tone = delta === undefined ? "flat" : polarity(delta);

  return (
    <div className="card">
      <p className="card-title">{label}</p>
      {/* Proportional figures: this is a standalone number, not a column that
          has to align. Tabular figures here would look mechanical. */}
      <p
        className={`mt-1.5 font-semibold tracking-tight ${
          emphasis ? "text-2xl" : "text-xl"
        } ${delta === undefined ? "text-ink" : POLARITY_INK[tone]}`}
      >
        {value}
      </p>
      {hint && <p className="mt-1 text-2xs text-ink-muted">{hint}</p>}
    </div>
  );
}
