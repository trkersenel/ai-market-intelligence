"use client";

/**
 * Impact analysis: who benefits, who is at risk, and the chain behind each.
 *
 * The path text is the panel's whole reason for existing. A ranked list of
 * companies with scores is indistinguishable from a model's guess; the same
 * list with "Microsoft → buys from → NVIDIA → is manufactured by → TSMC" beside
 * each row is something a reader can check and argue with.
 */

import Link from "next/link";
import { useState } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { useImpact } from "@/lib/api/hooks";
import type { ImpactedEntity } from "@/lib/api/types";
import {
  Badge,
  EmptyState,
  Panel,
  PanelHeader,
  SegmentedControl,
  Skeleton,
} from "@/components/ui/primitives";
import { cn } from "@/lib/cn";

const SCENARIOS = ["Expansion", "Contraction"] as const;
type Scenario = (typeof SCENARIOS)[number];

export function ImpactPanel({ identifier, name }: { identifier: string; name: string }) {
  const [scenario, setScenario] = useState<Scenario>("Expansion");
  const magnitude = scenario === "Expansion" ? 1 : -1;
  const { data, isPending, error } = useImpact(identifier, magnitude);

  const primary = scenario === "Expansion" ? data?.winners : data?.losers;
  const secondary = scenario === "Expansion" ? data?.losers : data?.winners;

  return (
    <Panel>
      <PanelHeader
        title="Impact analysis"
        subtitle={
          scenario === "Expansion"
            ? `If ${name} expands, who benefits`
            : `If ${name} contracts, who is at risk`
        }
        action={
          <SegmentedControl
            label="Scenario"
            options={SCENARIOS}
            value={scenario}
            onChange={setScenario}
          />
        }
      />
      <div className="p-4">
        {isPending ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <Skeleton key={index} className="h-11 w-full" />
            ))}
          </div>
        ) : error || !data ? (
          <EmptyState
            title="No impact analysis"
            description={`${name} is not in the ecosystem graph, so there are no relationships to trace.`}
          />
        ) : (
          <div className="space-y-4">
            <ImpactList entries={primary ?? []} />
            {secondary && secondary.length > 0 ? (
              <details className="group">
                <summary className="cursor-pointer text-[11px] font-medium text-[var(--color-subtle)] transition-colors hover:text-[var(--color-muted)]">
                  {scenario === "Expansion" ? "Negatively affected" : "Positively affected"} (
                  {secondary.length})
                </summary>
                <div className="mt-2">
                  <ImpactList entries={secondary} />
                </div>
              </details>
            ) : null}

            <p className="border-t border-[var(--color-line)] pt-3 text-[10px] leading-relaxed text-[var(--color-subtle)]">
              Scores are derived from curated relationships by an inspectable rule, not predicted.
              A company absent from the graph is absent from this list — a coverage limit, not a
              judgement that it is unaffected.
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}

function ImpactList({ entries }: { entries: ImpactedEntity[] }) {
  if (entries.length === 0) {
    return <p className="text-xs text-[var(--color-subtle)]">Nothing reached at this threshold.</p>;
  }

  const strongest = Math.max(...entries.map((entry) => Math.abs(entry.score)));

  return (
    <ul className="space-y-1.5">
      {entries.map((entry) => {
        const positive = entry.score > 0;
        return (
          <li key={entry.entity.slug}>
            <Link
              href={`/ecosystem/${entry.entity.slug}`}
              className="group block rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 transition-colors hover:border-[var(--color-line-strong)]"
            >
              <div className="flex items-center gap-2.5">
                {positive ? (
                  <TrendingUp size={13} className="shrink-0 text-[var(--color-up)]" aria-hidden />
                ) : (
                  <TrendingDown
                    size={13}
                    className="shrink-0 text-[var(--color-down)]"
                    aria-hidden
                  />
                )}
                <span className="truncate text-xs font-medium text-[var(--color-ink)] group-hover:text-[var(--color-accent)]">
                  {entry.entity.name}
                </span>
                {entry.entity.symbol ? <Badge>{entry.entity.symbol}</Badge> : null}

                <span className="ml-auto flex shrink-0 items-center gap-2">
                  {/* A bar rather than the raw number: the score's units are
                      arbitrary, so only its size relative to the others means
                      anything. Showing 1.18 invites reading it as a percentage. */}
                  <span className="h-1 w-16 overflow-hidden rounded-full bg-[var(--color-raised)]">
                    <span
                      className={cn(
                        "block h-full rounded-full",
                        positive ? "bg-[var(--color-up)]" : "bg-[var(--color-down)]",
                      )}
                      style={{ width: `${(Math.abs(entry.score) / strongest) * 100}%` }}
                    />
                  </span>
                  <span className="tnum w-8 text-right text-[10px] text-[var(--color-subtle)]">
                    {entry.confidence.toFixed(2)}
                  </span>
                </span>
              </div>

              {entry.paths[0] ? (
                <p className="mt-1.5 truncate pl-[23px] text-[10px] text-[var(--color-subtle)]">
                  {entry.paths[0].steps.join(" ")}
                </p>
              ) : null}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
