/**
 * Adjusted-close line with a crosshair tooltip.
 *
 * Adjusted close, not raw close: a split inside the window would otherwise
 * render as a cliff the reader would read as a crash.
 *
 * One series, so no legend -- the title names it. The endpoint is direct-labelled
 * because that is the value a reader looks for first; every other point is served
 * by the axis and the tooltip, and a number on every point would be unreadable.
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PriceBar } from "../lib/types";
import { formatAxisDate, formatDate, formatPrice, formatVolume, toNumber } from "../lib/format";

const AXIS = "#898781";
const GRID = "#2c2c2a";
const SERIES = "#3987e5";

interface Point {
  date: string;
  close: number;
  volume: number;
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: Point }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;

  return (
    <div className="rounded border border-hairline-strong bg-surface-raised px-2.5 py-2 shadow-lg">
      <p className="text-2xs text-ink-muted">{formatDate(point.date)}</p>
      <p className="mt-0.5 text-sm tabular text-ink">{formatPrice(point.close)}</p>
      <p className="text-2xs tabular text-ink-secondary">
        {formatVolume(point.volume)} shares
      </p>
    </div>
  );
}

export function PriceChart({ bars }: { bars: PriceBar[] }) {
  const data: Point[] = bars.flatMap((bar) => {
    const close = toNumber(bar.adjusted_close);
    return close === null ? [] : [{ date: bar.trade_date, close, volume: bar.volume }];
  });

  // `at(-1)` on a possibly-empty array is `undefined`, so the guard is a real
  // narrowing rather than an assertion that it cannot be.
  const last = data.at(-1);
  if (last === undefined) return null;
  const closes = data.map((point) => point.close);
  // A padded domain rather than starting at zero: for a price series the shape
  // of the movement is the information, and a zero baseline flattens it into a
  // straight line near the top of the plot.
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const pad = (max - min) * 0.08 || max * 0.02;

  return (
    // Height includes the x-axis band, so the axis labels are never clipped into
    // a nested scrollbar.
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 44, bottom: 4, left: 4 }}>
          {/* Horizontal hairlines only: vertical grid on a dense daily series is
              noise, and the crosshair already locates the x position. */}
          <CartesianGrid stroke={GRID} strokeWidth={1} vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatAxisDate}
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            minTickGap={48}
            tickLine={false}
          />
          <YAxis
            domain={[min - pad, max + pad]}
            tickFormatter={(value: number) => formatPrice(value)}
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            width={62}
            tickLine={false}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ stroke: AXIS, strokeWidth: 1 }}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke={SERIES}
            strokeWidth={2}
            dot={false}
            // Mount animation off. Recharts generates the animated path from a
            // snapshot of the previous geometry, so if the container is measured
            // before layout settles the path keeps the stale width -- observed
            // as a line spanning 134px inside an 880px axis. It is also the
            // right call on its own terms: a 1.5s draw-in on a 120-point daily
            // series is decoration, not information.
            isAnimationActive={false}
            // A marker large enough to hit on touch, with a surface ring so it
            // reads as separate from the line it sits on.
            activeDot={{ r: 4, fill: SERIES, stroke: "#1a1a19", strokeWidth: 2 }}
            name="Adjusted close"
          />
          <ReferenceDot
            x={last.date}
            y={last.close}
            r={3.5}
            fill={SERIES}
            stroke="#1a1a19"
            strokeWidth={2}
            label={{
              value: formatPrice(last.close),
              position: "right",
              fill: "#ffffff",
              fontSize: 11,
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
