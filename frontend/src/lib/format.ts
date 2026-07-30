/**
 * Display formatting.
 *
 * Every function takes the API's `string | null` money values rather than
 * numbers. Conversion to a JS number happens here, at the last possible moment
 * and only for display -- so an exact value is never silently rounded on its way
 * through the app.
 */

/** Parse an API decimal string, returning null for absent or unparseable input. */
export const toNumber = (value: string | null | undefined): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const nf = (options: Intl.NumberFormatOptions) => new Intl.NumberFormat("en-US", options);

const price2 = nf({ minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct2 = nf({ minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compact = nf({ notation: "compact", maximumFractionDigits: 1 });

/** A price, always two decimals so a column of them aligns. */
export const formatPrice = (value: string | number | null): string => {
  const parsed = typeof value === "number" ? value : toNumber(value);
  return parsed === null ? "—" : price2.format(parsed);
};

/**
 * A percentage, with an explicit sign.
 *
 * The sign is always shown, including on positives: in a column of returns a
 * bare "1.20" beside a "-3.40" invites misreading, and the plus costs nothing.
 */
export const formatPercent = (value: string | number | null, digits = 2): string => {
  const parsed = typeof value === "number" ? value : toNumber(value);
  if (parsed === null) return "—";
  const formatted = digits === 2 ? pct2.format(parsed) : parsed.toFixed(digits);
  return `${parsed > 0 ? "+" : ""}${formatted}%`;
};

/** A fractional return (0.05) rendered as a percentage (+5.00%). */
export const formatReturn = (value: string | number | null): string => {
  const parsed = typeof value === "number" ? value : toNumber(value);
  return parsed === null ? "—" : formatPercent(parsed * 100);
};

/** A plain number to a fixed precision. */
export const formatNumber = (value: string | number | null, digits = 2): string => {
  const parsed = typeof value === "number" ? value : toNumber(value);
  return parsed === null ? "—" : parsed.toFixed(digits);
};

/** Share volume, compacted (154,353,700 -> 154.4M). */
export const formatVolume = (value: number | null): string =>
  value === null ? "—" : compact.format(value);

/** An ISO date as a short, locale-independent label. */
export const formatDate = (iso: string | null): string => {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
};

/** An axis tick: day and month only, since the year is in the range label. */
export const formatAxisDate = (iso: string): string => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
};

/** A timestamp as a relative age, which is how news is actually read. */
export const formatRelative = (iso: string): string => {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days < 30 ? `${days}d ago` : formatDate(iso);
};

/**
 * Which pole of the diverging scale a value sits on.
 *
 * Returns a token name rather than a colour, so the palette lives in one place
 * and a component never hardcodes a hex.
 */
export const polarity = (value: string | number | null): "up" | "down" | "flat" => {
  const parsed = typeof value === "number" ? value : toNumber(value);
  if (parsed === null || parsed === 0) return "flat";
  return parsed > 0 ? "up" : "down";
};
