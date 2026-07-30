/**
 * The primitive layer.
 *
 * Small, unopinionated pieces every screen composes from. They exist so a
 * panel, a badge or a loading state looks the same everywhere without anyone
 * re-deciding padding and border colour -- and so a change to that decision is
 * one edit rather than a search.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { EMPTY, type Direction } from "@/lib/format";

/* --- Surfaces ----------------------------------------------------------- */

export function Panel({
  children,
  className,
  ...rest
}: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-[var(--radius-card)] border border-[var(--color-line)]",
        "bg-[var(--color-surface)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function PanelHeader({
  title,
  subtitle,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[var(--color-line)] px-4 py-3">
      <div className="min-w-0">
        <h2 className="text-sm font-semibold tracking-tight text-[var(--color-ink)]">{title}</h2>
        {subtitle ? (
          <p className="mt-0.5 truncate text-xs text-[var(--color-subtle)]">{subtitle}</p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

/* --- Direction ---------------------------------------------------------- */

const DIRECTION_CLASS: Record<Direction, string> = {
  up: "text-[var(--color-up)]",
  down: "text-[var(--color-down)]",
  flat: "text-[var(--color-flat)]",
};

/** The arrow that carries direction for anyone who cannot rely on the colour. */
const DIRECTION_GLYPH: Record<Direction, string> = {
  up: "▲",
  down: "▼",
  flat: "—",
};

export function DeltaText({
  direction,
  children,
  className,
  showGlyph = true,
}: {
  direction: Direction;
  children: ReactNode;
  className?: string;
  showGlyph?: boolean;
}) {
  return (
    <span className={cn("tnum inline-flex items-center gap-1", DIRECTION_CLASS[direction], className)}>
      {showGlyph ? (
        <span aria-hidden className="text-[0.7em] leading-none">
          {DIRECTION_GLYPH[direction]}
        </span>
      ) : null}
      {children}
    </span>
  );
}

/* --- Badges ------------------------------------------------------------- */

type BadgeTone = "neutral" | "accent" | "up" | "down" | "warn";

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral: "bg-[var(--color-raised)] text-[var(--color-muted)] border-[var(--color-line)]",
  accent: "bg-[var(--color-accent-soft)] text-[var(--color-accent)] border-transparent",
  up: "bg-[var(--color-up-soft)] text-[var(--color-up)] border-transparent",
  down: "bg-[var(--color-down-soft)] text-[var(--color-down)] border-transparent",
  warn: "bg-[color-mix(in_oklch,var(--color-sev-medium)_16%,transparent)] text-[var(--color-sev-medium)] border-transparent",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5",
        "text-[11px] font-medium leading-none whitespace-nowrap",
        BADGE_TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const SEVERITY_COLOR: Record<string, string> = {
  low: "var(--color-sev-low)",
  medium: "var(--color-sev-medium)",
  high: "var(--color-sev-high)",
  critical: "var(--color-sev-critical)",
};

export function SeverityBadge({ severity }: { severity: string }) {
  const color = SEVERITY_COLOR[severity] ?? "var(--color-muted)";
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[11px] font-medium leading-none capitalize"
      style={{
        color,
        backgroundColor: `color-mix(in oklch, ${color} 15%, transparent)`,
      }}
    >
      <span aria-hidden className="size-1.5 rounded-full" style={{ backgroundColor: color }} />
      {severity}
    </span>
  );
}

/* --- Values ------------------------------------------------------------- */

/**
 * A labelled figure.
 *
 * `value` accepts the already-formatted string, so the em dash for missing data
 * is decided by the formatter rather than re-derived here. When it *is* the
 * dash, the value is dimmed -- absent data should not have the same visual
 * weight as a real number.
 */
export function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
}) {
  const isEmpty = value === EMPTY;
  return (
    <div className={cn("min-w-0", className)}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-[var(--color-subtle)]">
        {label}
      </div>
      <div
        className={cn(
          "tnum mt-1 truncate text-base",
          isEmpty ? "text-[var(--color-subtle)]" : "text-[var(--color-ink)]",
        )}
      >
        {value}
      </div>
      {hint ? <div className="mt-0.5 text-xs text-[var(--color-subtle)]">{hint}</div> : null}
    </div>
  );
}

/* --- States ------------------------------------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-[var(--color-raised)]", className)}
      aria-hidden
    />
  );
}

/**
 * The empty state.
 *
 * Deliberately a first-class component rather than an afterthought: on this
 * platform "no data" is a frequent, *correct* outcome -- an untracked symbol has
 * no anomalies, and a free-tier provider serves no charts. Those must read as
 * answers, not as breakage.
 */
export function EmptyState({
  title,
  description,
  icon,
  action,
}: {
  title: string;
  description?: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon ? <div className="text-[var(--color-subtle)]">{icon}</div> : null}
      <p className="text-sm font-medium text-[var(--color-muted)]">{title}</p>
      {description ? (
        <p className="max-w-sm text-xs leading-relaxed text-[var(--color-subtle)]">{description}</p>
      ) : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

/* --- Controls ----------------------------------------------------------- */

type ButtonVariant = "primary" | "ghost" | "outline";

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--color-accent)] text-[var(--color-on-accent)] hover:opacity-90 border-transparent",
  ghost:
    "bg-transparent text-[var(--color-muted)] hover:bg-[var(--color-raised)] hover:text-[var(--color-ink)] border-transparent",
  outline:
    "bg-[var(--color-surface)] text-[var(--color-ink)] border-[var(--color-line)] hover:border-[var(--color-line-strong)]",
};

export function Button({
  children,
  variant = "outline",
  className,
  ...rest
}: {
  children: ReactNode;
  variant?: ButtonVariant;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5",
        "text-xs font-medium transition-colors",
        "disabled:pointer-events-none disabled:opacity-50",
        BUTTON_VARIANT[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/** A segmented control, used for chart ranges. */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: readonly T[];
  value: T;
  onChange: (next: T) => void;
  label: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] p-0.5"
    >
      {options.map((option) => {
        const active = option === value;
        return (
          <button
            key={option}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option)}
            className={cn(
              "rounded-md px-2 py-1 text-[11px] font-medium transition-colors",
              active
                ? "bg-[var(--color-raised)] text-[var(--color-ink)]"
                : "text-[var(--color-subtle)] hover:text-[var(--color-muted)]",
            )}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}
