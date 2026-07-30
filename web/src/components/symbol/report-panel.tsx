"use client";

/**
 * The AI briefing panel.
 *
 * Two things distinguish it from a chat box bolted onto a finance page.
 *
 * **The evidence is one click away.** The model was handed a numbered list of
 * facts and nothing else; that list is shown on demand beside the prose. A
 * reader who doubts a sentence can check it without leaving the panel, which is
 * the difference between an analysis and an assertion.
 *
 * **The wait is explained.** Cold generation runs a 3B model on the user's own
 * machine and takes the better part of a minute. A bare spinner for forty
 * seconds reads as a hang, so the panel says what is happening and where it is
 * happening -- and says it once, since the answer is then cached for the day.
 */

import { useState } from "react";
import { Check, ChevronDown, Cpu, Sparkles } from "lucide-react";
import { useCompanyReport } from "@/lib/api/hooks";
import { ApiError } from "@/lib/api/client";
import { Badge, Button, EmptyState, Panel, PanelHeader } from "@/components/ui/primitives";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/cn";

export function ReportPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error, refetch, isFetching } = useCompanyReport(symbol);
  const [showEvidence, setShowEvidence] = useState(false);

  return (
    <Panel>
      <PanelHeader
        title={
          <span className="inline-flex items-center gap-1.5">
            <Sparkles size={13} className="text-[var(--color-accent)]" aria-hidden />
            AI briefing
          </span>
        }
        subtitle={
          data
            ? `${data.model} · ${data.cached ? "cached" : "generated"} ${formatRelative(data.generated_at)}`
            : "Written from the figures on this page"
        }
        action={
          data ? (
            <Button
              variant="ghost"
              onClick={() => refetch()}
              disabled={isFetching}
              aria-label="Regenerate the briefing"
            >
              {isFetching ? "Writing…" : "Regenerate"}
            </Button>
          ) : undefined
        }
      />

      <div className="p-4">
        {isPending ? (
          <Pending />
        ) : error ? (
          <Failure error={error} />
        ) : (
          <>
            {/* Stated plainly rather than styled away. A list of figures
                presented as an analysis would be the one dishonest thing this
                panel could do. */}
            {!data.generated ? (
              <div className="mb-3">
                <Badge tone="warn">
                  <Cpu size={9} aria-hidden />
                  Not written by a model
                </Badge>
              </div>
            ) : null}

            <div className="space-y-3">
              {data.summary.split(/\n{2,}/).map((paragraph, index) => (
                <p
                  key={index}
                  className="text-xs leading-relaxed text-[var(--color-muted)]"
                >
                  {paragraph}
                </p>
              ))}
            </div>

            {data.evidence.length > 0 ? (
              <div className="mt-4 border-t border-[var(--color-line)] pt-3">
                <button
                  onClick={() => setShowEvidence((open) => !open)}
                  aria-expanded={showEvidence}
                  className="inline-flex items-center gap-1.5 text-[11px] font-medium text-[var(--color-subtle)] transition-colors hover:text-[var(--color-muted)]"
                >
                  <ChevronDown
                    size={12}
                    aria-hidden
                    className={cn("transition-transform", showEvidence && "rotate-180")}
                  />
                  {showEvidence ? "Hide" : "Show"} the {data.evidence.length} facts it was given
                </button>

                {showEvidence ? (
                  <ol className="scroll-thin mt-2.5 max-h-64 space-y-1 overflow-y-auto pr-1">
                    {data.evidence.map((fact, index) => (
                      <li
                        key={fact}
                        className="flex gap-2 text-[11px] leading-relaxed text-[var(--color-subtle)]"
                      >
                        <span className="tnum shrink-0 text-[var(--color-accent)]">
                          [{index + 1}]
                        </span>
                        <span className="tnum">{fact}</span>
                      </li>
                    ))}
                  </ol>
                ) : null}
              </div>
            ) : null}
          </>
        )}
      </div>
    </Panel>
  );
}

function Pending() {
  return (
    <div className="flex items-start gap-3">
      <div
        aria-hidden
        className="mt-0.5 size-3.5 shrink-0 animate-spin rounded-full border-2 border-[var(--color-line-strong)] border-t-[var(--color-accent)]"
      />
      <div className="space-y-1">
        <p className="text-xs text-[var(--color-muted)]">Reading this company&rsquo;s figures…</p>
        <p className="text-[11px] leading-relaxed text-[var(--color-subtle)]">
          A local model is writing the briefing on your machine, which takes up to a minute the
          first time. Nothing leaves the computer, and the result is cached for the rest of the day.
        </p>
      </div>
    </div>
  );
}

function Failure({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.isQuotaExceeded) {
    return (
      <EmptyState
        title="Provider quota reached"
        description="The briefing is built from live figures, and this minute's request budget is spent. Try again shortly."
      />
    );
  }
  return (
    <EmptyState
      title="Could not write the briefing"
      description={error instanceof Error ? error.message : undefined}
    />
  );
}

/** Marks the panel's promise, used in the page header. Exported for reuse. */
export function GroundedBadge() {
  return (
    <Badge tone="accent">
      <Check size={9} aria-hidden />
      Grounded
    </Badge>
  );
}
