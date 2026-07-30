"use client";

/**
 * The price chart.
 *
 * An area chart rather than candles by default: at a year of daily bars,
 * candle bodies are a pixel wide and carry no readable information, while the
 * closing line shows the shape of the move at a glance. Candles earn their
 * complexity at intraday resolution, which the free tier's 8-requests-a-minute
 * budget makes an occasional view rather than the default.
 *
 * The series is coloured by its own direction over the window -- green when it
 * ends above where it started -- which is the convention every terminal uses
 * and the first thing a reader checks.
 */

import { useId, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Candle } from "@/lib/api/types";
import { formatDate, formatPrice, formatTime, toNumber } from "@/lib/format";
import type { ChartRange } from "@/lib/api/hooks";

interface Point {
  t: number;
  close: number;
}

export function PriceChart({
  candles,
  range,
  currency = "USD",
}: {
  candles: Candle[];
  range: ChartRange;
  currency?: string;
}) {
  const intraday = range === "1D" || range === "5D";

  const points = useMemo<Point[]>(
    () =>
      candles
        .map((candle) => ({
          t: new Date(candle.timestamp).getTime(),
          close: toNumber(candle.close) ?? Number.NaN,
        }))
        .filter((point) => Number.isFinite(point.close)),
    [candles],
  );

  const first = points.at(0)?.close ?? 0;
  const last = points.at(-1)?.close ?? 0;
  const rising = last >= first;
  const stroke = rising ? "var(--color-up)" : "var(--color-down)";

  // A price axis anchored at zero wastes most of its height on a range no
  // instrument will visit, flattening the very movement the chart exists to
  // show. So the domain hugs the observed range with a little breathing room.
  //
  // The floor is clamped at zero, which is not defensive tidying: MU ran from
  // roughly $60 to $1,324 over a year, and a tenth of *that range* is $126 --
  // more than the low itself. Unclamped, the axis bottomed out at -$5.99 and
  // the chart advertised a negative share price.
  const domain = useMemo<[number, number]>(() => {
    if (points.length === 0) return [0, 1];
    const values = points.map((point) => point.close);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = (high - low || high || 1) * 0.08;
    return [Math.max(0, low - pad), high + pad];
  }, [points]);

  // A gradient id must be unique per chart instance, or two charts on one page
  // both resolve to whichever `<defs>` was parsed last. `useId` rather than a
  // random string: it is stable across re-renders and identical on the server
  // and the client, so the markup does not change between them.
  const gradientId = `price-gradient-${useId()}`;

  return (
    <div className="h-[340px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.22} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid
            stroke="var(--color-line)"
            strokeDasharray="2 4"
            vertical={false}
          />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value: number) =>
              intraday ? formatTime(new Date(value).toISOString()) : formatDate(new Date(value).toISOString())
            }
            tick={{ fill: "var(--color-subtle)", fontSize: 11 }}
            axisLine={{ stroke: "var(--color-line)" }}
            tickLine={false}
            minTickGap={48}
          />
          <YAxis
            domain={domain}
            orientation="right"
            width={68}
            tickFormatter={(value: number) => formatPrice(value, currency)}
            tick={{ fill: "var(--color-subtle)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0].payload as Point;
              const iso = new Date(point.t).toISOString();
              return (
                <div className="rounded-lg border border-[var(--color-line-strong)] bg-[var(--color-surface)] px-2.5 py-1.5 shadow-lg">
                  <div className="text-[10px] text-[var(--color-subtle)]">
                    {intraday ? `${formatDate(iso)} ${formatTime(iso)}` : formatDate(iso)}
                  </div>
                  <div className="tnum text-sm text-[var(--color-ink)]">
                    {formatPrice(point.close, currency)}
                  </div>
                </div>
              );
            }}
            cursor={{ stroke: "var(--color-line-strong)", strokeWidth: 1 }}
          />
          <Area
            type="monotone"
            dataKey="close"
            stroke={stroke}
            strokeWidth={1.75}
            fill={`url(#${gradientId})`}
            // Off deliberately. Recharts builds the mount animation from the
            // geometry it had when the element was created, which for a chart
            // whose data arrives asynchronously is a stale, much smaller box --
            // the line lands compressed into a fraction of the axis and stays
            // there. Verified: a line spanning 134px inside an 880px axis.
            isAnimationActive={false}
            dot={false}
            activeDot={{ r: 3, fill: stroke, strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
