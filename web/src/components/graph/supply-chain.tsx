"use client";

/**
 * The supply chain view.
 *
 * A layered flow rather than a force-directed blob, because a supply chain has
 * an inherent direction and the reader's question is positional: what sits
 * *above* this company, and what sits *below* it. A physics simulation would
 * throw that ordering away and make the answer a matter of where the springs
 * settled.
 *
 * Upstream is what the company depends on; downstream is what depends on it.
 * Working out which side an edge belongs on is not simply arrow direction --
 * `SUPPLIES` points supplier→customer while `CUSTOMER_OF` points the other way,
 * so both must be normalised against the company being viewed.
 */

import Link from "next/link";
import { ArrowDown } from "lucide-react";
import type { RelationshipEdge } from "@/lib/api/types";
import { Badge, EmptyState, Panel, PanelHeader, Skeleton } from "@/components/ui/primitives";
import { cn } from "@/lib/cn";

/** Relations where the arrow points from supplier toward customer. */
const POINTS_DOWNSTREAM = new Set(["supplies", "manufactures", "produces"]);

interface Tier {
  title: string;
  hint: string;
  entries: { other: string; edge: RelationshipEdge }[];
}

function partition(root: string, edges: RelationshipEdge[]): { up: Tier; down: Tier } {
  const up: Tier["entries"] = [];
  const down: Tier["entries"] = [];

  for (const edge of edges) {
    const rootIsSource = edge.source === root;
    const other = rootIsSource ? edge.target : edge.source;
    // Normalise to economic direction. An edge is "upstream of root" when the
    // other end is the one being depended on.
    const otherIsUpstream = POINTS_DOWNSTREAM.has(edge.kind) ? !rootIsSource : rootIsSource;
    (otherIsUpstream ? up : down).push({ other, edge });
  }

  const byWeight = (a: { edge: RelationshipEdge }, b: { edge: RelationshipEdge }) =>
    b.edge.weight - a.edge.weight;

  return {
    up: {
      title: "Upstream",
      hint: "What this company depends on",
      entries: up.sort(byWeight),
    },
    down: {
      title: "Downstream",
      hint: "What depends on this company",
      entries: down.sort(byWeight),
    },
  };
}

export function SupplyChain({
  root,
  rootName,
  edges,
  isPending,
  nameOf,
}: {
  root: string;
  rootName: string;
  edges: RelationshipEdge[];
  isPending: boolean;
  nameOf: (slug: string) => string;
}) {
  if (isPending) {
    return (
      <Panel>
        <PanelHeader title="Supply chain" />
        <div className="space-y-3 p-4">
          {Array.from({ length: 5 }).map((_, index) => (
            <Skeleton key={index} className="h-9 w-full" />
          ))}
        </div>
      </Panel>
    );
  }

  if (edges.length === 0) {
    return (
      <Panel>
        <PanelHeader title="Supply chain" />
        <EmptyState
          title="No supply relationships recorded"
          description="The ecosystem graph is curated from public disclosure and covers the AI infrastructure stack. A company outside it has no edges rather than none existing."
        />
      </Panel>
    );
  }

  const { up, down } = partition(root, edges);

  return (
    <Panel>
      <PanelHeader
        title="Supply chain"
        subtitle={`${edges.length} relationships, curated from public disclosure`}
      />
      <div className="space-y-1 p-4">
        <TierBlock tier={up} nameOf={nameOf} align="up" />

        <div className="flex items-center gap-2 py-2">
          <div className="h-px flex-1 bg-[var(--color-line)]" />
          <span className="rounded-lg border border-[var(--color-accent)] bg-[var(--color-accent-soft)] px-3 py-1.5 text-sm font-semibold text-[var(--color-accent)]">
            {rootName}
          </span>
          <div className="h-px flex-1 bg-[var(--color-line)]" />
        </div>

        <TierBlock tier={down} nameOf={nameOf} align="down" />
      </div>
    </Panel>
  );
}

function TierBlock({
  tier,
  nameOf,
  align,
}: {
  tier: Tier;
  nameOf: (slug: string) => string;
  align: "up" | "down";
}) {
  if (tier.entries.length === 0) {
    return (
      <div className="py-2 text-center text-[11px] text-[var(--color-subtle)]">
        No {tier.title.toLowerCase()} relationships recorded.
      </div>
    );
  }

  return (
    <div className={cn("space-y-1.5", align === "down" && "flex flex-col-reverse space-y-reverse")}>
      <div className="flex items-baseline gap-2 pb-0.5">
        <span className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-subtle)]">
          {tier.title}
        </span>
        <span className="text-[10px] text-[var(--color-subtle)]">{tier.hint}</span>
      </div>

      {tier.entries.map(({ other, edge }) => (
        <ChainRow key={`${edge.source}-${edge.target}-${edge.kind}`} slug={other} edge={edge} nameOf={nameOf} />
      ))}
    </div>
  );
}

function ChainRow({
  slug,
  edge,
  nameOf,
}: {
  slug: string;
  edge: RelationshipEdge;
  nameOf: (slug: string) => string;
}) {
  return (
    <Link
      href={`/ecosystem/${slug}`}
      className="group flex items-center gap-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 transition-colors hover:border-[var(--color-line-strong)]"
    >
      {/* Line weight encodes materiality, which is the thing a reader most
          wants ranked and the thing a plain list cannot show. */}
      <span
        aria-hidden
        className="h-6 w-0.5 shrink-0 rounded-full bg-[var(--color-accent)]"
        style={{ opacity: 0.25 + edge.weight * 0.75 }}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-xs font-medium text-[var(--color-ink)] group-hover:text-[var(--color-accent)]">
            {nameOf(slug)}
          </span>
          <Badge>{edge.kind.replaceAll("_", " ")}</Badge>
        </div>
        {edge.description ? (
          <p className="mt-0.5 truncate text-[11px] text-[var(--color-subtle)]">
            {edge.description}
          </p>
        ) : null}
      </div>
      <EvidenceBadge edge={edge} />
      <ArrowDown size={12} className="shrink-0 text-[var(--color-subtle)]" aria-hidden />
    </Link>
  );
}

/**
 * How the platform knows, shown rather than buried.
 *
 * A diagram whose edges cannot be checked is decoration. A curated edge from a
 * 10-K and one a model proposed from a headline must not look alike.
 */
export function EvidenceBadge({ edge }: { edge: RelationshipEdge }) {
  const inferred = edge.evidence === "inferred";
  return (
    <span
      title={edge.citation ?? `Source: ${edge.evidence.replaceAll("_", " ")}`}
      className={cn(
        "tnum shrink-0 rounded px-1.5 py-0.5 text-[10px]",
        inferred
          ? "bg-[color-mix(in_oklch,var(--color-sev-medium)_16%,transparent)] text-[var(--color-sev-medium)]"
          : "bg-[var(--color-raised)] text-[var(--color-subtle)]",
      )}
    >
      {inferred ? "inferred " : ""}
      {edge.confidence.toFixed(2)}
    </span>
  );
}
