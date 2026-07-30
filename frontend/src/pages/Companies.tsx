/**
 * Company explorer: a searchable list, and one company's detail beside it.
 *
 * The selected company lives in the URL rather than in component state, so a
 * chart a user is looking at can be linked to and survives a refresh -- which is
 * what an analyst does with anything worth a second look.
 */

import { useQuery } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useState } from "react";
import { api } from "../lib/api";
import { ErrorState, Panel, Skeleton } from "../components/states";
import { PriceChart } from "../components/PriceChart";
import { MacdPanel, RsiPanel } from "../components/IndicatorPanels";
import { NewsList } from "../components/NewsList";
import { AnomalyFeed } from "../components/AnomalyFeed";
import { StatTile } from "../components/StatTile";
import {
  formatDate,
  formatNumber,
  formatPercent,
  formatPrice,
  formatReturn,
  toNumber,
} from "../lib/format";
import type { CompanyDetail as CompanyDetailType } from "../lib/types";

const RANGES = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
] as const;

export function Companies() {
  const { slug } = useParams<{ slug?: string }>();
  const [params] = useSearchParams();
  const [search, setSearch] = useState("");

  const companies = useQuery({
    queryKey: ["companies", search],
    queryFn: () => api.companies.list(search.length >= 2 ? { search } : {}),
    staleTime: 60_000,
  });

  const selectedSlug = slug ?? companies.data?.[0]?.slug;
  const symbolHint = params.get("symbol");

  return (
    <div className="grid gap-4 lg:grid-cols-[16rem_1fr]">
      <aside className="card h-fit">
        <h2 className="card-title">Companies</h2>
        {/* Filters sit in one row above the content, per the interaction spec. */}
        <label className="mt-2 block">
          <span className="sr-only">Search companies</span>
          <input
            type="search"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
            }}
            placeholder="Search…"
            className="focusable w-full rounded border border-hairline bg-surface-page px-2 py-1.5 text-xs text-ink placeholder:text-ink-muted"
          />
        </label>

        {companies.isLoading ? (
          <div className="mt-3 space-y-1.5">
            {Array.from({ length: 6 }, (_, index) => (
              <Skeleton key={index} className="h-7" />
            ))}
          </div>
        ) : (
          <ul className="mt-3 space-y-0.5">
            {(companies.data ?? []).map((company) => (
              <li key={company.slug}>
                <Link
                  to={`/companies/${company.slug}`}
                  className={`focusable block rounded px-2 py-1.5 text-xs transition-colors ${
                    company.slug === selectedSlug
                      ? "bg-surface-raised text-ink"
                      : "text-ink-secondary hover:text-ink"
                  }`}
                >
                  <span className="block truncate">{company.name}</span>
                  <span className="block truncate text-2xs text-ink-muted">
                    {company.tags.slice(0, 3).join(" · ")}
                  </span>
                </Link>
              </li>
            ))}
            {(companies.data ?? []).length === 0 && (
              <li className="px-2 py-3 text-2xs text-ink-muted">No match.</li>
            )}
          </ul>
        )}
      </aside>

      {selectedSlug ? (
        <CompanyDetail slug={selectedSlug} symbolHint={symbolHint} />
      ) : (
        <div className="card">
          <p className="text-sm text-ink-muted">Select a company.</p>
        </div>
      )}
    </div>
  );
}

function CompanyDetail({ slug, symbolHint }: { slug: string; symbolHint: string | null }) {
  const company = useQuery({
    queryKey: ["company", slug],
    queryFn: () => api.companies.get(slug),
  });

  if (company.isLoading) return <Skeleton className="h-96" />;
  if (company.error) {
    return (
      <div className="card">
        <ErrorState error={company.error} onRetry={() => void company.refetch()} />
      </div>
    );
  }

  const detail = company.data;
  if (!detail) return null;

  // Prefer a symbol named in the URL, so an anomaly link lands on the right
  // listing for a company with several (TSMC has an ADR and a local line).
  const symbol =
    detail.tickers.find((ticker) => ticker.symbol === symbolHint)?.symbol ??
    detail.tickers[0]?.symbol;

  return (
    <div className="space-y-4">
      <CompanyHeader detail={detail} activeSymbol={symbol} />
      {/* The price and indicator panels take `symbol: string`, not an optional
          one. Splitting them out is what removes five non-null assertions:
          a company with no listing simply does not render them, and the
          compiler enforces that rather than a `!` promising it. */}
      {symbol === undefined ? (
        <div className="card">
          <p className="text-sm text-ink-muted">
            This company has no tracked listing, so there are no prices to show.
          </p>
        </div>
      ) : (
        <ListingPanels symbol={symbol} />
      )}
    </div>
  );
}

function CompanyHeader({
  detail,
  activeSymbol,
}: {
  detail: CompanyDetailType;
  activeSymbol: string | undefined;
}) {
  return (
    <section className="card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">{detail.name}</h1>
          <p className="mt-0.5 text-2xs text-ink-muted">
            {[detail.sector, detail.industry, detail.country].filter(Boolean).join(" · ")}
          </p>
        </div>
        <div className="flex flex-wrap gap-1">
          {detail.tickers.map((ticker) => (
            <span
              key={ticker.symbol}
              className={`rounded border px-1.5 py-0.5 text-2xs tabular ${
                ticker.symbol === activeSymbol
                  ? "border-series-1 text-ink"
                  : "border-hairline text-ink-muted"
              }`}
            >
              {ticker.symbol}
            </span>
          ))}
        </div>
      </div>
      {detail.description && (
        <p className="mt-2 max-w-3xl text-xs leading-relaxed text-ink-secondary">
          {detail.description}
        </p>
      )}
      {detail.tags.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1">
          {detail.tags.map((tag) => (
            <li
              key={tag}
              className="rounded bg-surface-raised px-1.5 py-0.5 text-2xs text-ink-secondary"
            >
              {tag.replace(/_/g, " ")}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ListingPanels({ symbol }: { symbol: string }) {
  const [days, setDays] = useState<number>(180);
  const start = new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10);

  const prices = useQuery({
    queryKey: ["prices", symbol, days],
    queryFn: () => api.prices.series(symbol, { start }),
  });

  const indicators = useQuery({
    queryKey: ["indicators", symbol, days],
    queryFn: () => api.indicators.series(symbol, { start }),
  });

  const latest = useQuery({
    queryKey: ["indicators", "latest", symbol],
    queryFn: () => api.indicators.latest(symbol),
    // A listing with no computed features 404s, which is expected before the
    // feature job has run rather than an outage.
    retry: false,
  });

  const news = useQuery({
    queryKey: ["news", symbol],
    queryFn: () => api.news.list({ tickers: symbol, days: 14, limit: 8 }),
  });

  const anomalies = useQuery({
    queryKey: ["anomalies", symbol],
    queryFn: () => api.anomalies.forSymbol(symbol),
  });

  const snapshot = latest.data;
  const rsi = toNumber(snapshot?.rsi_14 ?? null);
  const volatility = toNumber(snapshot?.volatility_20 ?? null);

  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile
          label="Last close"
          value={formatPrice(prices.data?.bars.at(-1)?.close ?? null)}
          hint={formatDate(prices.data?.end ?? null)}
        />
        <StatTile
          label={`${days}-day return`}
          value={formatReturn(prices.data?.period_return ?? null)}
          delta={prices.data?.period_return ?? null}
        />
        <StatTile
          label="RSI (14)"
          value={formatNumber(rsi, 1)}
          hint={
            rsi === null ? undefined : rsi >= 70 ? "overbought" : rsi <= 30 ? "oversold" : "neutral"
          }
        />
        <StatTile
          label="Volatility (20d)"
          value={volatility === null ? "—" : formatPercent(volatility * 100)}
          hint="annualised"
        />
      </div>

      <Panel
        title={`${symbol} adjusted close`}
        subtitle="Split- and dividend-adjusted, so a split is not rendered as a crash"
        isLoading={prices.isLoading}
        error={prices.error}
        onRetry={() => void prices.refetch()}
        isEmpty={(prices.data?.bars.length ?? 0) === 0}
        emptyMessage="No prices stored for this listing."
        skeletonClass="h-72"
        action={
          <div className="flex gap-0.5" role="group" aria-label="Date range">
            {RANGES.map((range) => (
              <button
                key={range.label}
                type="button"
                onClick={() => {
                  setDays(range.days);
                }}
                aria-pressed={days === range.days}
                className={`focusable rounded px-2 py-1 text-2xs tabular transition-colors ${
                  days === range.days
                    ? "bg-surface-raised text-ink"
                    : "text-ink-muted hover:text-ink"
                }`}
              >
                {range.label}
              </button>
            ))}
          </div>
        }
      >
        <PriceChart bars={prices.data?.bars ?? []} />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="RSI (14)"
          subtitle="Wilder's smoothing; 30 and 70 marked"
          isLoading={indicators.isLoading}
          error={indicators.error}
          isEmpty={(indicators.data?.rows.length ?? 0) === 0}
          emptyMessage="No indicators computed yet."
          skeletonClass="h-44"
        >
          <RsiPanel rows={indicators.data?.rows ?? []} />
        </Panel>

        <Panel
          title="MACD (12, 26, 9)"
          subtitle="Own axis: overlaying it on price would need a second y-scale"
          isLoading={indicators.isLoading}
          error={indicators.error}
          isEmpty={(indicators.data?.rows.length ?? 0) === 0}
          emptyMessage="No indicators computed yet."
          skeletonClass="h-44"
        >
          <MacdPanel rows={indicators.data?.rows ?? []} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="News"
          subtitle="Last 14 days, attributed to this listing"
          isLoading={news.isLoading}
          error={news.error}
          onRetry={() => void news.refetch()}
          isEmpty={(news.data?.length ?? 0) === 0}
          emptyMessage="No articles attributed to this listing."
        >
          <NewsList articles={news.data ?? []} />
        </Panel>

        <Panel
          title="Anomalies"
          isLoading={anomalies.isLoading}
          error={anomalies.error}
          onRetry={() => void anomalies.refetch()}
          isEmpty={(anomalies.data?.length ?? 0) === 0}
          emptyMessage="Nothing unusual in this window."
        >
          <AnomalyFeed anomalies={anomalies.data ?? []} />
        </Panel>
      </div>
    </>
  );
}
