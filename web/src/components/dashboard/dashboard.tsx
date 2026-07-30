"use client";

/**
 * The dashboard.
 *
 * Answers three questions in order: what is the ecosystem doing right now,
 * what does the platform actually cover, and where can I go next. The watchlist
 * is the tracked set -- the symbols the platform spends quota analysing --
 * because those are the ones where the rest of the product has something to
 * say.
 */

import Link from "next/link";
import { Database, Activity, Clock } from "lucide-react";
import { useQuote, useUniverseStats, useCapabilities } from "@/lib/api/hooks";
import {
  Badge,
  DeltaText,
  Panel,
  PanelHeader,
  Skeleton,
} from "@/components/ui/primitives";
import { directionOf, formatDate, formatPercent, formatPrice } from "@/lib/format";

/**
 * The tracked ecosystem, in the order a reader would want it: the memory names
 * the platform was built around, then the compute side, then the index proxies
 * that give those moves a benchmark.
 */
const WATCHLIST = [
  { symbol: "MU", label: "Micron" },
  { symbol: "NVDA", label: "NVIDIA" },
  { symbol: "AMD", label: "AMD" },
  { symbol: "AVGO", label: "Broadcom" },
  { symbol: "INTC", label: "Intel" },
  { symbol: "TSM", label: "TSMC" },
  { symbol: "ASML", label: "ASML" },
  { symbol: "SMCI", label: "Supermicro" },
  { symbol: "SMH", label: "Semis ETF" },
  { symbol: "SOXX", label: "Semis ETF" },
  { symbol: "VOO", label: "S&P 500" },
  { symbol: "GEV", label: "GE Vernova" },
] as const;

export function Dashboard() {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3 pt-1">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">AI infrastructure</h1>
          <p className="mt-0.5 text-xs text-[var(--color-subtle)]">
            Memory, compute and the fabs behind them.
          </p>
        </div>
        <CoverageSummary />
      </div>

      <Panel>
        <PanelHeader
          title="Tracked symbols"
          subtitle="Live quotes for the set the platform analyses"
        />
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-b-[var(--radius-card)] bg-[var(--color-line)] sm:grid-cols-3 lg:grid-cols-4">
          {WATCHLIST.map((item) => (
            <QuoteTile key={item.symbol} symbol={item.symbol} label={item.label} />
          ))}
        </div>
      </Panel>

      <ProviderPanel />
    </div>
  );
}

/**
 * One symbol's live quote.
 *
 * Each tile owns its own query rather than the parent fetching twelve at once,
 * so a symbol the provider cannot price shows an em dash while the other eleven
 * render normally. The backend coalesces and caches per symbol, so twelve
 * parallel requests do not become twelve upstream calls.
 */
function QuoteTile({ symbol, label }: { symbol: string; label: string }) {
  const { data, isPending, isError } = useQuote(symbol);
  const direction = directionOf(data?.change_percent);

  return (
    <Link
      href={`/symbol/${symbol}`}
      className="group flex flex-col gap-1 bg-[var(--color-surface)] p-3 transition-colors hover:bg-[var(--color-raised)]"
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="tnum text-sm font-semibold text-[var(--color-ink)]">{symbol}</span>
        <span className="truncate text-[10px] text-[var(--color-subtle)]">{label}</span>
      </div>

      {isPending ? (
        <>
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-3 w-14" />
        </>
      ) : isError ? (
        <span className="text-xs text-[var(--color-subtle)]">Unavailable</span>
      ) : (
        <>
          <span className="tnum text-lg leading-tight">
            {formatPrice(data.price, data.currency)}
          </span>
          <DeltaText direction={direction} className="text-xs">
            {formatPercent(data.change_percent, { signed: true })}
          </DeltaText>
        </>
      )}
    </Link>
  );
}

/** How much of the exchange is browsable, and how much is analysed. */
function CoverageSummary() {
  const { data, isPending } = useUniverseStats();

  if (isPending) {
    return <Skeleton className="h-9 w-56" />;
  }
  if (!data) return null;

  return (
    <div className="flex items-center gap-5 text-xs">
      <span className="inline-flex items-center gap-1.5 text-[var(--color-subtle)]">
        <Database size={12} aria-hidden />
        <span className="tnum text-[var(--color-ink)]">{data.listings.toLocaleString()}</span>
        browsable
      </span>
      <span className="inline-flex items-center gap-1.5 text-[var(--color-subtle)]">
        <Activity size={12} aria-hidden />
        <span className="tnum text-[var(--color-ink)]">{data.tracked}</span>
        analysed
      </span>
      {data.last_synced_at ? (
        <span className="hidden items-center gap-1.5 text-[var(--color-subtle)] sm:inline-flex">
          <Clock size={12} aria-hidden />
          synced {formatDate(data.last_synced_at)}
        </span>
      ) : null}
    </div>
  );
}

/**
 * What the configured providers serve.
 *
 * On the surface a diagnostic panel; in practice the honest answer to "why is
 * there no cash flow statement here". A free tier has real edges, and stating
 * them is better than letting a reader conclude the feature is broken.
 */
function ProviderPanel() {
  const { data, isPending } = useCapabilities();

  return (
    <Panel>
      <PanelHeader
        title="Data coverage"
        subtitle="Which provider serves what, on the current configuration"
      />
      <div className="p-4">
        {isPending ? (
          <Skeleton className="h-16 w-full" />
        ) : !data ? (
          <p className="text-xs text-[var(--color-subtle)]">Coverage is unavailable.</p>
        ) : (
          <div className="space-y-3">
            {Object.entries(data.providers).map(([provider, capabilities]) => (
              <div key={provider} className="flex flex-wrap items-baseline gap-2">
                <span className="w-24 shrink-0 text-xs font-medium capitalize text-[var(--color-muted)]">
                  {provider}
                </span>
                <div className="flex flex-wrap gap-1">
                  {capabilities.map((capability) => (
                    <Badge key={capability}>{capability.replaceAll("_", " ")}</Badge>
                  ))}
                </div>
              </div>
            ))}
            <p className="pt-1 text-[11px] leading-relaxed text-[var(--color-subtle)]">
              Anything not listed resolves to a typed “not available” rather than an error, and a
              panel that needs it is not rendered. Adding a provider that serves it makes those
              panels appear with no other change.
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}
