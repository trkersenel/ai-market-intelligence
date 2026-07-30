/**
 * Shapes returned by the backend.
 *
 * Hand-written rather than generated from the OpenAPI schema, deliberately: the
 * frontend consumes a small, stable subset, and hand-written types let the money
 * fields stay `string`. Prices and quantities are NUMERIC in PostgreSQL and are
 * serialised as strings precisely so they do not pass through a JavaScript
 * number, where 0.1 + 0.2 !== 0.3. Parsing them to `number` here would undo the
 * exactness the whole backend maintains.
 */

export type Severity = "low" | "medium" | "high" | "extreme";
export type Direction = "up" | "down" | "neutral";
export type Sentiment = "bullish" | "bearish" | "neutral";
export type AssetType = "equity" | "etf" | "index";

export interface Company {
  id: number;
  slug: string;
  name: string;
  sector: string | null;
  industry: string | null;
  country: string | null;
  tags: string[];
  is_tracked: boolean;
}

export interface CompanyDetail extends Company {
  website: string | null;
  description: string | null;
  tickers: Ticker[];
}

export interface Ticker {
  id: number;
  symbol: string;
  display_name: string;
  exchange: string | null;
  currency: string;
  asset_type: AssetType;
  is_active: boolean;
  last_price_date: string | null;
}

export interface PriceBar {
  trade_date: string;
  open: string;
  high: string;
  low: string;
  close: string;
  adjusted_close: string;
  volume: number;
}

export interface PriceSeries {
  symbol: string;
  start: string | null;
  end: string | null;
  bars: PriceBar[];
  count: number;
  period_return: string | null;
}

export interface TickerQuote {
  symbol: string;
  display_name: string;
  trade_date: string;
  close: string;
  previous_close: string | null;
  volume: number;
  change_percent: string | null;
}

export interface IndicatorSnapshot {
  trade_date: string;
  daily_return: string | null;
  weekly_return: string | null;
  monthly_return: string | null;
  sma_20: string | null;
  sma_50: string | null;
  sma_200: string | null;
  rsi_14: string | null;
  macd: string | null;
  macd_signal: string | null;
  macd_histogram: string | null;
  bollinger_upper: string | null;
  bollinger_lower: string | null;
  atr_14: string | null;
  volatility_20: string | null;
  volume_ratio: string | null;
  relative_strength_smh: string | null;
}

export interface IndicatorSeries {
  symbol: string;
  start: string | null;
  end: string | null;
  rows: IndicatorSnapshot[];
  count: number;
}

export interface Anomaly {
  id: number;
  symbol: string;
  trade_date: string;
  anomaly_type: string;
  method: string;
  direction: Direction;
  severity: Severity;
  score: number;
  confidence: number;
  explanation: string | null;
  related_document_ids: string[];
}

export interface NewsArticle {
  id: string | null;
  url: string;
  title: string;
  summary: string | null;
  source: string;
  source_name: string | null;
  published_at: string;
  tickers: string[];
  tags: string[];
  sentiment: Sentiment | null;
  sentiment_confidence: number | null;
}

export interface Citation {
  number: number;
  source_id: string;
  title: string | null;
  url: string | null;
  published_at: string | null;
  matched_by: string[];
  score: number;
}

export interface ChatAnswer {
  conversation_id: string;
  question: string;
  resolved_question: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  retrieved: number;
  refused: boolean;
  extractive: boolean;
  model_name: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  confidence: number | null;
  retrieved_document_ids: string[];
  model_name: string | null;
}

/** The platform's error envelope; every failing response has this shape. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    request_id: string | null;
    details?: Record<string, unknown>;
  };
}
