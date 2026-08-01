"use client";

/**
 * The bridge from a company page into the ecosystem graph.
 *
 * Renders nothing when the symbol has no node. Most of the 5,664 browsable
 * listings sit outside the AI infrastructure graph, and a permanent "no
 * ecosystem data" panel on every one of them would be noise -- the absence is
 * the normal case, not a finding.
 */

import Link from "next/link";
import { ArrowUpRight, Network } from "lucide-react";
import { useEcosystem } from "@/lib/api/hooks";
import { Panel } from "@/components/ui/primitives";

export function EcosystemLink({ symbol }: { symbol: string }) {
  const { data, isPending, isError } = useEcosystem(symbol, 1);

  if (isPending || isError || !data) return null;

  const suppliers = data.edges.filter((edge) =>
    ["supplies", "manufactures", "depends_on"].includes(edge.kind),
  ).length;

  return (
    <Panel className="overflow-hidden">
      <Link
        href={`/ecosystem/${data.root}`}
        className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-[var(--color-raised)]"
      >
        <Network size={16} className="shrink-0 text-[var(--color-accent)]" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-[var(--color-ink)]">Ecosystem &amp; supply chain</p>
          <p className="mt-0.5 text-xs text-[var(--color-subtle)]">
            {data.edges.length} mapped relationships, {suppliers} of them supply-side — see who
            depends on this company and who it depends on.
          </p>
        </div>
        <ArrowUpRight size={14} className="shrink-0 text-[var(--color-subtle)]" aria-hidden />
      </Link>
    </Panel>
  );
}
