/**
 * Number and date formatting.
 *
 * Two rules run through all of it.
 *
 * **Absent is not zero.** A missing P/E renders as an em dash, never as 0.00,
 * because a reader who sees 0.00 believes it. The backend is careful to send
 * `null` rather than a placeholder, and discarding that care in the last three
 * lines of the pipeline would waste it.
 *
 * **Money arrives as a string.** The API serialises `Decimal` to JSON strings
 * ("874.66", not 874.66) so an exchange price reaches the browser exactly as
 * the exchange printed it, with no binary rounding anywhere in the path. These
 * formatters therefore accept `string | number` and coerce once, here, at the
 * render boundary -- where a float is finally harmless because the value is
 * about to become pixels.
 */

/** Anything the API might send for a numeric field. */
export type Numeric = string | number | null | undefined;

/** What every formatter shows when a value is genuinely unavailable. */
export const EMPTY = "—";

/**
 * Coerce a wire value to a number, or `null` when there isn't one.
 *
 * The empty string maps to `null` rather than to 0, which is what `Number("")`
 * would give -- a silent and very believable wrong answer.
 */
export function toNumber(value: Numeric): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Format a price with the precision its magnitude warrants. */
export function formatPrice(value: Numeric, currency = "USD"): string {
  const number = toNumber(value);
  if (number === null) return EMPTY;
  // Sub-dollar instruments need more places, or a penny stock renders as "0.00"
  // and appears worthless. Above a dollar, two places is what an exchange
  // quotes and more would be false precision.
  const digits = Math.abs(number) < 1 ? 4 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(number);
}

/**
 * Format a value that is *already* a percentage.
 *
 * The backend sends margins and growth rates as percentage values (72.57 means
 * 72.57%), not as fractions -- verified against the live response for MU. A
 * formatter that multiplied by 100 would report a 72% gross margin as 7,257%.
 */
export function formatPercent(
  value: Numeric,
  { digits = 2, signed = false }: { digits?: number; signed?: boolean } = {},
): string {
  const number = toNumber(value);
  if (number === null) return EMPTY;
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    signDisplay: signed ? "always" : "auto",
  }).format(number);
  return `${formatted}%`;
}

/** Format a large number compactly: 1.2B, 340M, 5.6K. */
export function formatCompact(value: Numeric): string {
  const number = toNumber(value);
  if (number === null) return EMPTY;
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(number);
}

/** Format a plain number with thousands separators. */
export function formatNumber(value: Numeric, digits = 2): string {
  const number = toNumber(value);
  if (number === null) return EMPTY;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(number);
}

/** Which direction a value points, for choosing colour and glyph. */
export type Direction = "up" | "down" | "flat";

export function directionOf(value: Numeric): Direction {
  const number = toNumber(value);
  if (number === null || number === 0) return "flat";
  return number > 0 ? "up" : "down";
}

const DATE_FORMAT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

const TIME_FORMAT = new Intl.DateTimeFormat("en-US", {
  hour: "numeric",
  minute: "2-digit",
});

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? EMPTY : DATE_FORMAT.format(parsed);
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? EMPTY : TIME_FORMAT.format(parsed);
}

/** "3h ago", "2d ago" -- for news, where recency is the point. */
export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return EMPTY;

  const seconds = (Date.now() - parsed.getTime()) / 1000;
  if (seconds < 0) return "just now";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return formatDate(iso);
}
