/**
 * Loading, error and empty states.
 *
 * Shared because they are the states most dashboards get wrong: a spinner that
 * shifts the layout when data lands, an error that says "something went wrong",
 * and an empty result rendered as a broken-looking chart. Each of the three is a
 * distinct thing to say, so each gets its own component.
 */

import type { ReactNode } from "react";
import { ApiError } from "../lib/api";

/**
 * A skeleton sized to the content it replaces.
 *
 * Matching the final height matters: a spinner that collapses to nothing makes
 * the whole page jump when data arrives, and the reader loses their place.
 */
export function Skeleton({ className = "h-24" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded bg-hairline/60 ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const isApi = error instanceof ApiError;
  const message = isApi ? error.message : "The request could not be completed.";

  return (
    <div role="alert" className="flex flex-col items-start gap-2 py-6">
      <p className="text-sm text-ink">{message}</p>
      {/* The request id is the same one in the response header and the server
          logs, so a pasted error is traceable to the exact request. */}
      {isApi && error.requestId && (
        <p className="text-2xs text-ink-muted tabular">request {error.requestId}</p>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="focusable mt-1 rounded border border-hairline-strong px-2 py-1 text-2xs text-ink-secondary hover:bg-surface-raised"
        >
          Try again
        </button>
      )}
    </div>
  );
}

/**
 * An empty result, stated as a fact with a reason.
 *
 * "No anomalies in this window" is information; a blank panel looks like a bug.
 */
export function EmptyState({ children }: { children: ReactNode }) {
  return <p className="py-6 text-sm text-ink-muted">{children}</p>;
}

/** Wraps a panel in the right state for the query it renders. */
export function Panel({
  title,
  subtitle,
  isLoading,
  error,
  onRetry,
  isEmpty,
  emptyMessage,
  action,
  children,
  skeletonClass,
}: {
  title: string;
  subtitle?: string;
  isLoading?: boolean;
  error?: unknown;
  onRetry?: () => void;
  isEmpty?: boolean;
  emptyMessage?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  skeletonClass?: string;
}) {
  return (
    <section className="card">
      <header className="mb-3 flex items-baseline justify-between gap-3">
        <div>
          <h2 className="card-title">{title}</h2>
          {subtitle && <p className="mt-0.5 text-2xs text-ink-muted">{subtitle}</p>}
        </div>
        {action}
      </header>
      {error ? (
        <ErrorState error={error} onRetry={onRetry} />
      ) : isLoading ? (
        <Skeleton className={skeletonClass ?? "h-40"} />
      ) : isEmpty ? (
        <EmptyState>{emptyMessage ?? "No data for this selection."}</EmptyState>
      ) : (
        children
      )}
    </section>
  );
}
