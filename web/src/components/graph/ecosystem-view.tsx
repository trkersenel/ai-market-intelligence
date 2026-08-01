"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowUpRight, Network } from "lucide-react";
import { EcosystemMap, MapLegend } from "@/components/graph/ecosystem-map";
import { ImpactPanel } from "@/components/graph/impact-panel";
import { SupplyChain } from "@/components/graph/supply-chain";
import { ApiError } from "@/lib/api/client";
import { useEcosystem, useSupplyChain } from "@/lib/api/hooks";
import {
  Badge,
  EmptyState,
  Panel,
  PanelHeader,
  SegmentedControl,
  Skeleton,
} from "@/components/ui/primitives";

const DEPTHS = ["1 hop", "2 hops", "3 hops"] as const;
type Depth = (typeof DEPTHS)[number];

export function EcosystemView({ identifier }: { identifier: string }) {
  const [depth, setDepth] = useState<Depth>("1 hop");
  const hops = DEPTHS.indexOf(depth) + 1;

  const ecosystem = useEcosystem(identifier, hops);
  const supplyChain = useSupplyChain(identifier);

  const root = useMemo(
    () => ecosystem.data?.nodes.find((node) => node.slug === ecosystem.data?.root),
    [ecosystem.data],
  );

  // One lookup for the whole page. The supply chain endpoint returns slugs
  // rather than names, and re-fetching each entity to resolve a label would be
  // a request per row.
  const nameOf = useMemo(() => {
    const names = new Map(ecosystem.data?.nodes.map((node) => [node.slug, node.name]) ?? []);
    return (slug: string) => names.get(slug) ?? slug;
  }, [ecosystem.data]);

  if (ecosystem.error instanceof ApiError && ecosystem.error.isNotFound) {
    return (
      <Panel className="mt-8">
        <EmptyState
          icon={<Network size={20} aria-hidden />}
          title={`${identifier} is not in the ecosystem graph`}
          description="The graph is curated from public disclosure and covers the AI infrastructure stack — accelerators, foundry, memory, equipment, power and the labs. Most listed companies sit outside it."
        />
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3 pt-1">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold tracking-tight">
              {root?.name ?? identifier.toUpperCase()}
            </h1>
            {root?.symbol ? (
              <Link href={`/symbol/${root.symbol}`}>
                <Badge tone="accent">
                  {root.symbol}
                  <ArrowUpRight size={9} aria-hidden />
                </Badge>
              </Link>
            ) : null}
            {root?.tags.slice(0, 3).map((tag) => <Badge key={tag}>{tag}</Badge>)}
          </div>
          {root?.summary ? (
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-[var(--color-subtle)]">
              {root.summary}
            </p>
          ) : null}
        </div>
      </div>

      <Panel>
        <PanelHeader
          title="Ecosystem map"
          subtitle={
            ecosystem.data
              ? `${ecosystem.data.nodes.length} entities · ${ecosystem.data.edges.length} relationships`
              : undefined
          }
          action={
            <SegmentedControl label="Traversal depth" options={DEPTHS} value={depth} onChange={setDepth} />
          }
        />
        {ecosystem.isPending || !ecosystem.data ? (
          <Skeleton className="m-4 h-[520px]" />
        ) : (
          <>
            <EcosystemMap graph={ecosystem.data} />
            <MapLegend />
          </>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <SupplyChain
          root={ecosystem.data?.root ?? identifier}
          rootName={root?.name ?? identifier.toUpperCase()}
          edges={supplyChain.data ?? []}
          isPending={supplyChain.isPending}
          nameOf={nameOf}
        />
        <ImpactPanel identifier={identifier} name={root?.name ?? identifier.toUpperCase()} />
      </div>
    </div>
  );
}
