"use client";

/**
 * Query hooks.
 *
 * The stale times are the interesting part. Each matches how fast the
 * underlying thing actually changes, so the app never refetches a company
 * profile to learn that Micron is still in Boise -- and never caches a quote
 * long enough for it to be wrong.
 */

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { api, ApiError } from "./client";
import type {
  AnalystRating,
  Capabilities,
  CandleSeries,
  CompanyProfile,
  CompanyReport,
  Earnings,
  InsiderTransaction,
  KeyMetrics,
  ListingSummary,
  EcosystemGraph,
  GraphStats,
  ImpactAnalysis,
  ProviderNewsItem,
  Quote,
  RelationshipEdge,
  UniverseStats,
} from "./types";

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;

/** The chart windows the backend accepts. Mirrors its `_RANGES` vocabulary. */
export const CHART_RANGES = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX"] as const;
export type ChartRange = (typeof CHART_RANGES)[number];

/** Query keys in one place, so an invalidation cannot miss a variant. */
export const keys = {
  universeSearch: (q: string) => ["universe", "search", q] as const,
  universeStats: () => ["universe", "stats"] as const,
  listing: (symbol: string) => ["universe", "listing", symbol] as const,
  capabilities: () => ["market", "capabilities"] as const,
  quote: (symbol: string) => ["market", "quote", symbol] as const,
  candles: (symbol: string, range: ChartRange) => ["market", "candles", symbol, range] as const,
  profile: (symbol: string) => ["market", "profile", symbol] as const,
  metrics: (symbol: string) => ["market", "metrics", symbol] as const,
  ratings: (symbol: string) => ["market", "ratings", symbol] as const,
  insiders: (symbol: string) => ["market", "insiders", symbol] as const,
  earnings: (symbol: string) => ["market", "earnings", symbol] as const,
  news: (symbol: string) => ["market", "news", symbol] as const,
  report: (symbol: string) => ["market", "report", symbol] as const,
};

/**
 * Retry policy shared by every query.
 *
 * A 501 means no configured provider serves this and none will until the
 * configuration changes; a 404 means the symbol does not exist. Retrying
 * either spends latency to reach the same answer -- and on a quota-limited free
 * tier, retrying a 429 actively makes the situation worse.
 */
function retry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError) {
    if (error.isUnsupported || error.isNotFound || error.isQuotaExceeded) return false;
  }
  return failureCount < 2;
}

type Options<T> = Omit<UseQueryOptions<T, Error>, "queryKey" | "queryFn">;

/* --- Universe ----------------------------------------------------------- */

export function useUniverseSearch(query: string, options?: Options<ListingSummary[]>) {
  return useQuery({
    queryKey: keys.universeSearch(query),
    queryFn: ({ signal }) =>
      api.get<ListingSummary[]>("/universe/search", { params: { q: query, limit: 12 }, signal }),
    // The universe changes at IPO and delisting, so a result set stays good for
    // a long time. This is what makes the palette feel instant on a re-query.
    staleTime: 10 * MINUTE,
    enabled: query.trim().length > 0,
    retry,
    ...options,
  });
}

export function useUniverseStats(options?: Options<UniverseStats>) {
  return useQuery({
    queryKey: keys.universeStats(),
    queryFn: ({ signal }) => api.get<UniverseStats>("/universe/stats", { signal }),
    staleTime: 10 * MINUTE,
    retry,
    ...options,
  });
}

export function useListing(symbol: string, options?: Options<ListingSummary>) {
  return useQuery({
    queryKey: keys.listing(symbol),
    queryFn: ({ signal }) => api.get<ListingSummary>(`/universe/${symbol}`, { signal }),
    staleTime: 10 * MINUTE,
    retry,
    ...options,
  });
}

/* --- Market ------------------------------------------------------------- */

/**
 * What the configured providers can serve.
 *
 * Fetched once and effectively never refetched: it changes only when an API key
 * is added and the server restarts. Pages read it to decide whether to render a
 * section at all, rather than rendering one that resolves to a 501.
 */
export function useCapabilities(options?: Options<Capabilities>) {
  return useQuery({
    queryKey: keys.capabilities(),
    queryFn: ({ signal }) => api.get<Capabilities>("/market/capabilities", { signal }),
    staleTime: Infinity,
    retry,
    ...options,
  });
}

export function useQuote(symbol: string, options?: Options<Quote>) {
  return useQuery({
    queryKey: keys.quote(symbol),
    queryFn: ({ signal }) => api.get<Quote>(`/market/${symbol}/quote`, { signal }),
    // Matches the backend's own quote TTL. Anything shorter would miss that
    // cache and spend quota to receive the identical value.
    staleTime: 15 * SECOND,
    refetchInterval: 30 * SECOND,
    retry,
    ...options,
  });
}

export function useCandles(symbol: string, range: ChartRange, options?: Options<CandleSeries>) {
  return useQuery({
    queryKey: keys.candles(symbol, range),
    queryFn: ({ signal }) =>
      api.get<CandleSeries>(`/market/${symbol}/candles`, {
        params: { chart_range: range },
        signal,
      }),
    staleTime: 15 * MINUTE,
    // Keeps the previous range's bars on screen while the next loads, so
    // switching 1M -> 1Y redraws rather than blanking. A chart that disappears
    // on every range change reads as broken even when it is merely loading.
    placeholderData: (previous) => previous,
    retry,
    ...options,
  });
}

export function useProfile(symbol: string, options?: Options<CompanyProfile>) {
  return useQuery({
    queryKey: keys.profile(symbol),
    queryFn: ({ signal }) => api.get<CompanyProfile>(`/market/${symbol}/profile`, { signal }),
    staleTime: 24 * HOUR,
    retry,
    ...options,
  });
}

export function useMetrics(symbol: string, options?: Options<KeyMetrics>) {
  return useQuery({
    queryKey: keys.metrics(symbol),
    queryFn: ({ signal }) => api.get<KeyMetrics>(`/market/${symbol}/metrics`, { signal }),
    staleTime: HOUR,
    retry,
    ...options,
  });
}

export function useRatings(symbol: string, options?: Options<AnalystRating[]>) {
  return useQuery({
    queryKey: keys.ratings(symbol),
    queryFn: ({ signal }) => api.get<AnalystRating[]>(`/market/${symbol}/ratings`, { signal }),
    staleTime: HOUR,
    retry,
    ...options,
  });
}

export function useInsiders(symbol: string, options?: Options<InsiderTransaction[]>) {
  return useQuery({
    queryKey: keys.insiders(symbol),
    queryFn: ({ signal }) =>
      api.get<InsiderTransaction[]>(`/market/${symbol}/insiders`, {
        params: { limit: 15 },
        signal,
      }),
    staleTime: 6 * HOUR,
    retry,
    ...options,
  });
}

export function useEarnings(symbol: string, options?: Options<Earnings[]>) {
  return useQuery({
    queryKey: keys.earnings(symbol),
    queryFn: ({ signal }) =>
      api.get<Earnings[]>(`/market/${symbol}/earnings`, { params: { limit: 8 }, signal }),
    staleTime: 6 * HOUR,
    retry,
    ...options,
  });
}

export function useCompanyNews(symbol: string, options?: Options<ProviderNewsItem[]>) {
  return useQuery({
    queryKey: keys.news(symbol),
    queryFn: ({ signal }) =>
      api.get<ProviderNewsItem[]>(`/market/${symbol}/news`, { params: { days: 14 }, signal }),
    staleTime: 5 * MINUTE,
    retry,
    ...options,
  });
}

/**
 * The AI briefing.
 *
 * Cold generation runs a local model and takes roughly forty seconds; the
 * backend then caches it for twelve hours, so this is slow once per symbol per
 * day and instant thereafter.
 *
 * That shape drives the settings. There is no timeout to race, no refetch on
 * an interval, and no retry -- a second attempt would queue another forty
 * seconds of inference behind the first for the same answer.
 */
export function useCompanyReport(symbol: string, options?: Options<CompanyReport>) {
  return useQuery({
    queryKey: keys.report(symbol),
    queryFn: ({ signal }) => api.get<CompanyReport>(`/market/${symbol}/report`, { signal }),
    staleTime: 12 * HOUR,
    retry: false,
    ...options,
  });
}

/* --- Knowledge graph ------------------------------------------------------ */

export const graphKeys = {
  stats: () => ["graph", "stats"] as const,
  entities: (tags: string[]) => ["graph", "entities", ...tags] as const,
  ecosystem: (id: string, depth: number) => ["graph", "ecosystem", id, depth] as const,
  supplyChain: (id: string) => ["graph", "supply-chain", id] as const,
  impact: (id: string, magnitude: number) => ["graph", "impact", id, magnitude] as const,
};

/**
 * The graph changes only when the curated seed does, which is a deploy. These
 * are effectively static within a session, so they are fetched once and kept.
 */
const GRAPH_STALE_TIME = 30 * MINUTE;

export function useGraphStats(options?: Options<GraphStats>) {
  return useQuery({
    queryKey: graphKeys.stats(),
    queryFn: ({ signal }) => api.get<GraphStats>("/ecosystem/stats", { signal }),
    staleTime: GRAPH_STALE_TIME,
    retry,
    ...options,
  });
}

export function useEcosystem(
  identifier: string,
  depth: number,
  options?: Options<EcosystemGraph>,
) {
  return useQuery({
    queryKey: graphKeys.ecosystem(identifier, depth),
    queryFn: ({ signal }) =>
      api.get<EcosystemGraph>(`/ecosystem/${identifier}`, { params: { depth }, signal }),
    staleTime: GRAPH_STALE_TIME,
    // Keeps the previous depth on screen while the next loads, so widening the
    // view redraws rather than blanking the canvas.
    placeholderData: (previous) => previous,
    retry,
    ...options,
  });
}

export function useSupplyChain(identifier: string, options?: Options<RelationshipEdge[]>) {
  return useQuery({
    queryKey: graphKeys.supplyChain(identifier),
    queryFn: ({ signal }) =>
      api.get<RelationshipEdge[]>(`/ecosystem/${identifier}/supply-chain`, { signal }),
    staleTime: GRAPH_STALE_TIME,
    retry,
    ...options,
  });
}

export function useImpact(
  identifier: string,
  magnitude: number,
  options?: Options<ImpactAnalysis>,
) {
  return useQuery({
    queryKey: graphKeys.impact(identifier, magnitude),
    queryFn: ({ signal }) =>
      api.get<ImpactAnalysis>(`/ecosystem/${identifier}/impact`, {
        params: { magnitude, limit: 8 },
        signal,
      }),
    staleTime: GRAPH_STALE_TIME,
    placeholderData: (previous) => previous,
    retry,
    ...options,
  });
}
