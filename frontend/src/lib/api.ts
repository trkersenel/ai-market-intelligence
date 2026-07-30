/**
 * Typed API client.
 *
 * One place that knows how to talk to the backend, so no component builds a URL
 * or unwraps an error envelope. Requests are same-origin and proxied in dev,
 * which means no CORS preflight and no absolute base URL baked into the bundle.
 */

import type {
  Anomaly,
  ApiErrorBody,
  ChatAnswer,
  ChatMessage,
  Company,
  CompanyDetail,
  IndicatorSeries,
  IndicatorSnapshot,
  NewsArticle,
  PriceSeries,
  Ticker,
  TickerQuote,
} from "./types";

const BASE = "/api/v1";

/**
 * An API failure carrying the platform's error code and request id.
 *
 * The request id is surfaced to the user on unexpected failures: it is the same
 * id in the response header and the server logs, so a pasted error is traceable
 * to the exact request rather than to "sometime this afternoon".
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId: string | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const { error } = value;
  return (
    typeof error === "object" &&
    error !== null &&
    typeof (error as { code?: unknown }).code === "string" &&
    typeof (error as { message?: unknown }).message === "string"
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Headers built through the Headers API rather than object spread: HeadersInit
  // may be an array of pairs, and spreading that into an object yields indices.
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");

  const response = await fetch(`${BASE}${path}`, { ...init, headers });

  if (!response.ok) {
    // The backend always returns the envelope, but a proxy or gateway failure
    // may not — so parsing it is best-effort and never the reason an error is
    // swallowed.
    let code = "http_error";
    let message = `Request failed with status ${response.status}`;
    let requestId: string | null = response.headers.get("X-Request-ID");
    try {
      // Parsed as `unknown` and narrowed, not asserted: a proxy or gateway
      // failure returns whatever it likes, and casting to the envelope type
      // would be claiming a shape we have not checked.
      const body: unknown = await response.json();
      if (isErrorBody(body)) {
        code = body.error.code;
        message = body.error.message;
        requestId = body.error.request_id ?? requestId;
      }
    } catch {
      /* keep the fallbacks */
    }
    throw new ApiError(response.status, code, message, requestId);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const query = (params: Record<string, string | number | undefined>): string => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.append(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
};

export const api = {
  companies: {
    list: (params: { search?: string; tags?: string } = {}) =>
      request<Company[]>(`/companies${query(params)}`),
    get: (slug: string) => request<CompanyDetail>(`/companies/${slug}`),
  },

  tickers: {
    list: (params: { asset_type?: string } = {}) =>
      request<Ticker[]>(`/tickers${query(params)}`),
    quote: (symbol: string) =>
      request<TickerQuote>(`/prices/${encodeURIComponent(symbol)}/latest`),
  },

  prices: {
    series: (symbol: string, params: { start?: string; end?: string } = {}) =>
      request<PriceSeries>(`/prices/${encodeURIComponent(symbol)}${query(params)}`),
  },

  indicators: {
    series: (symbol: string, params: { start?: string; end?: string } = {}) =>
      request<IndicatorSeries>(
        `/indicators/${encodeURIComponent(symbol)}${query(params)}`,
      ),
    latest: (symbol: string) =>
      request<IndicatorSnapshot>(`/indicators/${encodeURIComponent(symbol)}/latest`),
  },

  anomalies: {
    list: (params: { days?: number; min_severity?: string; limit?: number } = {}) =>
      request<Anomaly[]>(`/anomalies${query(params)}`),
    forSymbol: (symbol: string) =>
      request<Anomaly[]>(`/anomalies/${encodeURIComponent(symbol)}`),
  },

  news: {
    list: (params: { tickers?: string; days?: number; limit?: number } = {}) =>
      request<NewsArticle[]>(`/news${query(params)}`),
  },

  chat: {
    ask: (question: string, conversationId?: string) =>
      request<ChatAnswer>("/chat", {
        method: "POST",
        body: JSON.stringify({
          question,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        }),
      }),
    history: (conversationId: string) =>
      request<ChatMessage[]>(`/chat/${conversationId}`),
  },
};
