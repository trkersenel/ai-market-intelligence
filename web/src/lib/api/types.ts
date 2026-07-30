/**
 * Types mirroring the backend's response schemas.
 *
 * Hand-written rather than generated, deliberately: they are the place to
 * record what the backend's shapes *mean*.
 *
 * **`Money` is a string.** The API serialises `Decimal` to a JSON string so an
 * exchange price arrives exactly as printed, with no binary rounding in the
 * path. Typing these as `number` would compile and then quietly produce
 * "874.66000000000008" on screen. The formatters in `@/lib/format` accept both
 * and coerce once, at the render boundary.
 *
 * **`null` is not zero.** A null here means "no configured provider serves
 * this", which the UI must render as unavailable rather than as 0.00.
 */

/** A monetary or exact-decimal value, serialised as a string by the API. */
export type Money = string;

/** One security in the browsable universe. */
export interface ListingSummary {
  symbol: string;
  name: string;
  exchange: string;
  currency: string;
  security_type: string;
  /**
   * Whether the platform runs analysis on this symbol. The distinction the
   * whole product turns on: a tracked symbol has price history, indicators,
   * anomalies and embedded news behind it; an untracked one can be browsed and
   * quoted on demand but has no analysis. The UI must say so rather than
   * rendering empty panels that read as breakage.
   */
  is_tracked: boolean;
}

export interface UniverseStats {
  listings: number;
  tracked: number;
  last_synced_at: string | null;
}

export type MarketSessionName = "pre" | "regular" | "post" | "closed";

export interface Quote {
  symbol: string;
  timestamp: string;
  /** Absent for a halted issue, or one that has not traded today. */
  price: Money | null;
  open: Money | null;
  high: Money | null;
  low: Money | null;
  previous_close: Money | null;
  volume: number | null;
  average_volume: number | null;
  week_52_high: Money | null;
  week_52_low: Money | null;
  session: MarketSessionName;
  currency: string;
  /** Derived by the backend so every client agrees on what "change" means. */
  change: Money | null;
  change_percent: number | null;
}

export interface Candle {
  timestamp: string;
  open: Money;
  high: Money;
  low: Money;
  close: Money;
  volume: number;
}

export interface CandleSeries {
  symbol: string;
  interval: string;
  adjusted: boolean;
  candles: Candle[];
}

export interface CompanyProfile {
  symbol: string;
  name: string;
  logo_url: string | null;
  exchange: string | null;
  industry: string | null;
  sector: string | null;
  country: string | null;
  website: string | null;
  ipo_date: string | null;
  market_cap: Money | null;
  shares_outstanding: Money | null;
  description: string | null;
  currency: string | null;
}

/**
 * Ratios and margins.
 *
 * Margins and growth rates arrive as **percentage values**, not fractions:
 * `gross_margin: 72.57` means 72.57%. Verified against the live response for
 * MU. Multiplying these by 100 would report a 72% margin as 7,257%.
 */
export interface KeyMetrics {
  symbol: string;
  pe_ratio: number | null;
  forward_pe: number | null;
  peg_ratio: number | null;
  price_to_book: number | null;
  price_to_sales: number | null;
  ev_to_ebitda: number | null;
  gross_margin: number | null;
  operating_margin: number | null;
  net_margin: number | null;
  return_on_equity: number | null;
  return_on_assets: number | null;
  revenue_growth_yoy: number | null;
  eps_growth_yoy: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  eps: number | null;
  dividend_yield: number | null;
  beta: number | null;
  week_52_change: number | null;
}

export interface AnalystRating {
  symbol: string;
  period: string;
  strong_buy: number;
  buy: number;
  hold: number;
  sell: number;
  strong_sell: number;
  total: number;
  consensus: string | null;
}

export interface InsiderTransaction {
  symbol: string;
  name: string;
  transaction_date: string | null;
  shares: number | null;
  price: Money | null;
  change: number | null;
  transaction_code: string | null;
  filing_date: string | null;
}

export interface Earnings {
  symbol: string;
  fiscal_date: string;
  eps_actual: Money | null;
  eps_estimate: Money | null;
  revenue_actual: Money | null;
  revenue_estimate: Money | null;
  report_date: string | null;
  /** Positive is a beat, including when both figures are losses. */
  surprise_percent: number | null;
}

export interface ProviderNewsItem {
  headline: string;
  url: string;
  published_at: string;
  source: string;
  summary: string | null;
  image_url: string | null;
}

/** What each configured provider can serve. */
export interface Capabilities {
  providers: Record<string, string[]>;
  capabilities: string[];
}

/** The platform's error envelope. Every failure has this shape. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    request_id?: string;
    details?: Record<string, unknown>;
  };
}
