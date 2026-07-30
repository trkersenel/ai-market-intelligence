/**
 * Market overview: breadth, a cross-sectional heatmap, and the anomalies feed.
 *
 * Breadth is presented as stat tiles rather than a chart. Four numbers is not a
 * chart's job -- a four-bar plot would encode nothing the numbers do not already
 * say, and the tile puts the value where the eye lands first.
 */

import { useQueries, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Panel } from "../components/states";
import { StatTile } from "../components/StatTile";
import { Heatmap, type HeatmapCell } from "../components/Heatmap";
import { AnomalyFeed } from "../components/AnomalyFeed";
import { NewsList } from "../components/NewsList";
import { formatPercent, toNumber } from "../lib/format";

export function Dashboard() {
  const tickers = useQuery({
    queryKey: ["tickers"],
    queryFn: () => api.tickers.list(),
    staleTime: 5 * 60_000,
  });

  const symbols = (tickers.data ?? []).map((ticker) => ticker.symbol);

  // One request per symbol. The universe is fourteen listings, so a fan-out is
  // cheaper and simpler than a bespoke batch endpoint -- and each quote caches
  // independently, so revisiting the page refetches only what went stale.
  const quotes = useQueries({
    queries: symbols.map((symbol) => ({
      queryKey: ["quote", symbol],
      queryFn: () => api.tickers.quote(symbol),
      staleTime: 60_000,
      // A listing with no stored bars 404s, which is expected rather than an
      // outage; retrying would only delay the page.
      retry: false,
    })),
  });

  const companies = useQuery({
    queryKey: ["companies"],
    queryFn: () => api.companies.list(),
    staleTime: 5 * 60_000,
  });

  const anomalies = useQuery({
    queryKey: ["anomalies", "recent"],
    queryFn: () => api.anomalies.list({ days: 30, limit: 12 }),
  });

  const news = useQuery({
    queryKey: ["news", "recent"],
    queryFn: () => api.news.list({ days: 3, limit: 10 }),
  });

  const slugBySymbol = new Map(
    (companies.data ?? []).flatMap((company) =>
      // The company list does not carry symbols, so the heatmap links by slug
      // only where a company's name plainly maps to one of its listings.
      [[company.slug.toUpperCase(), company.slug] as const],
    ),
  );

  const cells: HeatmapCell[] = quotes.flatMap((query, index) => {
    const symbol = symbols[index];
    if (!symbol || !query.data) return [];
    return [
      {
        symbol,
        changePercent: toNumber(query.data.change_percent),
        slug: slugBySymbol.get(symbol) ?? null,
      },
    ];
  });

  const changes = cells.map((cell) => cell.changePercent).filter((v): v is number => v !== null);
  const advancers = changes.filter((value) => value > 0).length;
  const decliners = changes.filter((value) => value < 0).length;
  const average = changes.length
    ? changes.reduce((sum, value) => sum + value, 0) / changes.length
    : null;
  const leader = cells.reduce<HeatmapCell | null>((best, cell) => {
    if (cell.changePercent === null) return best;
    if (!best || best.changePercent === null) return cell;
    return cell.changePercent > best.changePercent ? cell : best;
  }, null);

  const quotesLoading = tickers.isLoading || quotes.some((query) => query.isLoading);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile
          label="Average move"
          value={quotesLoading ? "…" : formatPercent(average)}
          delta={average}
          hint="across tracked listings"
          emphasis
        />
        <StatTile
          label="Advancing"
          value={quotesLoading ? "…" : advancers}
          hint={`${decliners} declining`}
        />
        <StatTile
          label="Strongest"
          value={quotesLoading || !leader ? "…" : leader.symbol}
          delta={leader?.changePercent}
          hint={leader ? formatPercent(leader.changePercent) : undefined}
        />
        <StatTile
          label="Anomalies"
          value={anomalies.isLoading ? "…" : (anomalies.data?.length ?? 0)}
          hint="last 30 days"
        />
      </div>

      <Panel
        title="Session heatmap"
        subtitle="Change versus the previous close. Blue is up, red is down."
        isLoading={quotesLoading}
        error={tickers.error}
        onRetry={() => void tickers.refetch()}
        isEmpty={cells.length === 0}
        emptyMessage="No quotes stored yet. Run price ingestion to populate them."
        skeletonClass="h-40"
      >
        <Heatmap cells={cells} />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Detected anomalies"
          subtitle="Robust z-score and isolation forest, last 30 days"
          isLoading={anomalies.isLoading}
          error={anomalies.error}
          onRetry={() => void anomalies.refetch()}
          isEmpty={(anomalies.data?.length ?? 0) === 0}
          emptyMessage="Nothing unusual in the last 30 days."
        >
          <AnomalyFeed anomalies={anomalies.data ?? []} />
        </Panel>

        <Panel
          title="Latest news"
          subtitle="Per-ticker financial sources, scored by FinBERT"
          isLoading={news.isLoading}
          error={news.error}
          onRetry={() => void news.refetch()}
          isEmpty={(news.data?.length ?? 0) === 0}
          emptyMessage="No articles in the last three days."
        >
          <NewsList articles={news.data ?? []} />
        </Panel>
      </div>
    </div>
  );
}
