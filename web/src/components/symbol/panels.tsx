"use client";

/**
 * The company page's panels.
 *
 * Each one owns its own query, so a slow or unavailable section never blocks
 * the rest of the page: metrics render while ratings are still in flight, and a
 * capability the free tier does not serve collapses to a short explanation
 * instead of taking the page down with it.
 */

import { ApiError } from "@/lib/api/client";
import {
  useEarnings,
  useInsiders,
  useMetrics,
  useRatings,
  useCompanyNews,
} from "@/lib/api/hooks";
import type { AnalystRating } from "@/lib/api/types";
import {
  Badge,
  DeltaText,
  EmptyState,
  Panel,
  PanelHeader,
  Skeleton,
  Stat,
} from "@/components/ui/primitives";
import {
  directionOf,
  formatCompact,
  formatDate,
  formatNumber,
  formatPercent,
  formatPrice,
  formatRelative,
} from "@/lib/format";

/**
 * Turns a failed query into the right words.
 *
 * A 501 is not an error -- it means no configured provider serves this, which
 * is a permanent, explainable state on a free tier. Saying "something went
 * wrong" there would send the reader looking for a fault that does not exist.
 */
function FailureState({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.isUnsupported) {
    return (
      <EmptyState
        title="Not available on the current data plan"
        description="No configured provider serves this. Adding a provider that does makes the section appear -- no other change is needed."
      />
    );
  }
  if (error instanceof ApiError && error.isQuotaExceeded) {
    return (
      <EmptyState
        title="Provider quota reached"
        description="The free tier's request budget for this minute is spent. This panel will fill in shortly."
      />
    );
  }
  return (
    <EmptyState
      title="Could not load"
      description={error instanceof Error ? error.message : "An unexpected error occurred."}
    />
  );
}

function PanelBody({ children }: { children: React.ReactNode }) {
  return <div className="p-4">{children}</div>;
}

function LoadingGrid({ rows = 6 }: { rows?: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="space-y-1.5">
          <Skeleton className="h-2.5 w-16" />
          <Skeleton className="h-4 w-20" />
        </div>
      ))}
    </div>
  );
}

/* --- Metrics ------------------------------------------------------------ */

export function MetricsPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error } = useMetrics(symbol);

  return (
    <Panel>
      <PanelHeader title="Key metrics" subtitle="Valuation, profitability and growth" />
      <PanelBody>
        {isPending ? (
          <LoadingGrid rows={9} />
        ) : error ? (
          <FailureState error={error} />
        ) : (
          <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3">
            <Stat label="P/E" value={formatNumber(data.pe_ratio)} />
            <Stat label="Forward P/E" value={formatNumber(data.forward_pe)} />
            <Stat label="P/B" value={formatNumber(data.price_to_book)} />
            <Stat label="P/S" value={formatNumber(data.price_to_sales)} />
            <Stat label="EV / EBITDA" value={formatNumber(data.ev_to_ebitda)} />
            <Stat label="Beta" value={formatNumber(data.beta)} />
            {/* Margins arrive as percentage values (72.57 = 72.57%), not
                fractions -- so they are formatted, never rescaled. */}
            <Stat label="Gross margin" value={formatPercent(data.gross_margin, { digits: 1 })} />
            <Stat
              label="Operating margin"
              value={formatPercent(data.operating_margin, { digits: 1 })}
            />
            <Stat label="Net margin" value={formatPercent(data.net_margin, { digits: 1 })} />
            <Stat label="ROE" value={formatPercent(data.return_on_equity, { digits: 1 })} />
            <Stat label="ROA" value={formatPercent(data.return_on_assets, { digits: 1 })} />
            <Stat label="Debt / equity" value={formatNumber(data.debt_to_equity)} />
            <Stat
              label="Revenue growth"
              value={
                <DeltaText direction={directionOf(data.revenue_growth_yoy)}>
                  {formatPercent(data.revenue_growth_yoy, { digits: 1, signed: true })}
                </DeltaText>
              }
              hint="Year over year"
            />
            <Stat
              label="EPS growth"
              value={
                <DeltaText direction={directionOf(data.eps_growth_yoy)}>
                  {formatPercent(data.eps_growth_yoy, { digits: 1, signed: true })}
                </DeltaText>
              }
              hint="Year over year"
            />
            <Stat label="EPS" value={formatNumber(data.eps)} />
          </div>
        )}
      </PanelBody>
    </Panel>
  );
}

/* --- Analyst ratings ---------------------------------------------------- */

const RATING_BANDS = [
  { key: "strong_buy", label: "Strong buy", color: "var(--color-up)" },
  { key: "buy", label: "Buy", color: "color-mix(in oklch, var(--color-up) 70%, var(--color-flat))" },
  { key: "hold", label: "Hold", color: "var(--color-flat)" },
  {
    key: "sell",
    label: "Sell",
    color: "color-mix(in oklch, var(--color-down) 70%, var(--color-flat))",
  },
  { key: "strong_sell", label: "Strong sell", color: "var(--color-down)" },
] as const;

export function RatingsPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error } = useRatings(symbol);
  const latest: AnalystRating | undefined = data?.[0];

  return (
    <Panel>
      <PanelHeader
        title="Analyst ratings"
        subtitle={latest ? `${latest.total} analysts · ${formatDate(latest.period)}` : undefined}
        action={latest?.consensus ? <Badge tone="accent">{latest.consensus}</Badge> : undefined}
      />
      <PanelBody>
        {isPending ? (
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        ) : error ? (
          <FailureState error={error} />
        ) : !latest || latest.total === 0 ? (
          <EmptyState
            title="No analyst coverage"
            description="Common for smaller listings and most ETFs."
          />
        ) : (
          <div className="space-y-3">
            {/* A single stacked bar rather than five: the question a reader has
                is "which way does the street lean", and proportions of one bar
                answer it faster than five to compare. */}
            <div
              className="flex h-2.5 w-full overflow-hidden rounded-full"
              role="img"
              aria-label={`${latest.strong_buy} strong buy, ${latest.buy} buy, ${latest.hold} hold, ${latest.sell} sell, ${latest.strong_sell} strong sell`}
            >
              {RATING_BANDS.map((band) => {
                const count = latest[band.key];
                if (count === 0) return null;
                return (
                  <div
                    key={band.key}
                    style={{
                      width: `${(count / latest.total) * 100}%`,
                      backgroundColor: band.color,
                    }}
                  />
                );
              })}
            </div>

            <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
              {RATING_BANDS.map((band) => (
                <li key={band.key} className="flex items-center gap-2 text-xs">
                  <span
                    aria-hidden
                    className="size-2 shrink-0 rounded-sm"
                    style={{ backgroundColor: band.color }}
                  />
                  <span className="flex-1 truncate text-[var(--color-subtle)]">{band.label}</span>
                  <span className="tnum text-[var(--color-ink)]">{latest[band.key]}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </PanelBody>
    </Panel>
  );
}

/* --- Earnings ----------------------------------------------------------- */

export function EarningsPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error } = useEarnings(symbol);

  return (
    <Panel>
      <PanelHeader title="Earnings" subtitle="Reported EPS against estimate" />
      <PanelBody>
        {isPending ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <Skeleton key={index} className="h-7 w-full" />
            ))}
          </div>
        ) : error ? (
          <FailureState error={error} />
        ) : !data.length ? (
          <EmptyState title="No reported earnings" />
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-wide text-[var(--color-subtle)]">
                <th className="pb-2 font-medium">Quarter</th>
                <th className="pb-2 text-right font-medium">Estimate</th>
                <th className="pb-2 text-right font-medium">Actual</th>
                <th className="pb-2 text-right font-medium">Surprise</th>
              </tr>
            </thead>
            <tbody>
              {data.map((quarter) => (
                <tr key={quarter.fiscal_date} className="border-t border-[var(--color-line)]">
                  <td className="py-2 text-[var(--color-muted)]">
                    {formatDate(quarter.fiscal_date)}
                  </td>
                  <td className="tnum py-2 text-right text-[var(--color-muted)]">
                    {formatNumber(quarter.eps_estimate)}
                  </td>
                  <td className="tnum py-2 text-right text-[var(--color-ink)]">
                    {formatNumber(quarter.eps_actual)}
                  </td>
                  <td className="py-2 text-right">
                    <DeltaText direction={directionOf(quarter.surprise_percent)}>
                      {formatPercent(quarter.surprise_percent, { digits: 1, signed: true })}
                    </DeltaText>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </PanelBody>
    </Panel>
  );
}

/* --- Insiders ----------------------------------------------------------- */

export function InsidersPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error } = useInsiders(symbol);

  return (
    <Panel>
      <PanelHeader title="Insider transactions" subtitle="As reported to the SEC" />
      <PanelBody>
        {isPending ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <Skeleton key={index} className="h-7 w-full" />
            ))}
          </div>
        ) : error ? (
          <FailureState error={error} />
        ) : !data.length ? (
          <EmptyState title="No reported insider activity" />
        ) : (
          <ul className="scroll-thin max-h-72 space-y-1.5 overflow-y-auto">
            {data.map((transaction, index) => {
              // `change` is signed: positive is an acquisition, negative a
              // disposal. It is the only field that says which, so it drives
              // both the label and the colour.
              const direction = directionOf(transaction.change);
              return (
                <li
                  key={`${transaction.name}-${transaction.filing_date}-${index}`}
                  className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-xs hover:bg-[var(--color-raised)]"
                >
                  <span className="min-w-0 flex-1 truncate text-[var(--color-muted)]">
                    {transaction.name}
                  </span>
                  <span className="tnum shrink-0 text-[var(--color-subtle)]">
                    {formatDate(transaction.transaction_date)}
                  </span>
                  <DeltaText direction={direction} className="w-20 shrink-0 justify-end">
                    {formatCompact(transaction.change)}
                  </DeltaText>
                </li>
              );
            })}
          </ul>
        )}
      </PanelBody>
    </Panel>
  );
}

/* --- News --------------------------------------------------------------- */

export function NewsPanel({ symbol }: { symbol: string }) {
  const { data, isPending, error } = useCompanyNews(symbol);

  return (
    <Panel>
      <PanelHeader title="Recent news" subtitle="Last 14 days" />
      <PanelBody>
        {isPending ? (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, index) => (
              <div key={index} className="space-y-1.5">
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-2.5 w-24" />
              </div>
            ))}
          </div>
        ) : error ? (
          <FailureState error={error} />
        ) : !data.length ? (
          <EmptyState title="No recent coverage" description="Nothing published in the last 14 days." />
        ) : (
          <ul className="scroll-thin max-h-[28rem] space-y-3 overflow-y-auto">
            {data.map((item) => (
              <li key={item.url}>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group block rounded-lg px-2 py-1.5 transition-colors hover:bg-[var(--color-raised)]"
                >
                  <p className="text-xs leading-snug text-[var(--color-ink)] group-hover:text-[var(--color-accent)]">
                    {item.headline}
                  </p>
                  <p className="mt-1 flex items-center gap-1.5 text-[10px] text-[var(--color-subtle)]">
                    <span className="truncate">{item.source}</span>
                    <span aria-hidden>·</span>
                    <span className="shrink-0">{formatRelative(item.published_at)}</span>
                  </p>
                </a>
              </li>
            ))}
          </ul>
        )}
      </PanelBody>
    </Panel>
  );
}

/* --- Quote statistics --------------------------------------------------- */

export function QuoteStats({
  open,
  high,
  low,
  previousClose,
  volume,
  averageVolume,
  week52High,
  week52Low,
  marketCap,
  currency,
}: {
  open: string | null;
  high: string | null;
  low: string | null;
  previousClose: string | null;
  volume: number | null;
  averageVolume: number | null;
  week52High: string | null;
  week52Low: string | null;
  marketCap: string | null;
  currency: string;
}) {
  return (
    <Panel>
      <PanelHeader title="Session" />
      <PanelBody>
        <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3">
          <Stat label="Open" value={formatPrice(open, currency)} />
          <Stat label="High" value={formatPrice(high, currency)} />
          <Stat label="Low" value={formatPrice(low, currency)} />
          <Stat label="Prev close" value={formatPrice(previousClose, currency)} />
          <Stat label="Volume" value={formatCompact(volume)} />
          <Stat label="Avg volume" value={formatCompact(averageVolume)} />
          <Stat label="52w high" value={formatPrice(week52High, currency)} />
          <Stat label="52w low" value={formatPrice(week52Low, currency)} />
          <Stat label="Market cap" value={formatCompact(marketCap)} />
        </div>
      </PanelBody>
    </Panel>
  );
}
