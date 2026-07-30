/**
 * RSI and MACD, each in its own panel.
 *
 * Separate panels rather than overlaid on the price chart, and this is the single
 * most important decision in the file. RSI is bounded 0-100, MACD oscillates
 * around zero in price units, and a close is in dollars: putting any two of them
 * on one plot needs a second y-axis, and the alignment between two y-scales is
 * arbitrary -- so the chart invents a relationship that is not in the data. One
 * axis per plot, always.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { IndicatorSnapshot } from "../lib/types";
import { formatAxisDate, formatDate, formatNumber, toNumber } from "../lib/format";

const AXIS = "#898781";
const GRID = "#2c2c2a";
const SERIES_1 = "#3987e5";
const SERIES_2 = "#d95926";
const UP = "#3987e5";
const DOWN = "#d03b3b";

/** RSI's conventional overbought and oversold levels. */
const RSI_OVERBOUGHT = 70;
const RSI_OVERSOLD = 30;

function TooltipCard({
  date,
  rows,
}: {
  date: string;
  rows: Array<{ label: string; value: string; color?: string }>;
}) {
  return (
    <div className="rounded border border-hairline-strong bg-surface-raised px-2.5 py-2 shadow-lg">
      <p className="text-2xs text-ink-muted">{formatDate(date)}</p>
      {rows.map((row) => (
        <p key={row.label} className="mt-0.5 flex items-center gap-1.5 text-2xs">
          {row.color && (
            <span
              aria-hidden
              className="h-2 w-2 rounded-sm"
              style={{ background: row.color }}
            />
          )}
          {/* The label and value wear text tokens; the swatch carries identity. */}
          <span className="text-ink-secondary">{row.label}</span>
          <span className="ml-auto tabular text-ink">{row.value}</span>
        </p>
      ))}
    </div>
  );
}

export function RsiPanel({ rows }: { rows: IndicatorSnapshot[] }) {
  const data = rows.flatMap((row) => {
    const rsi = toNumber(row.rsi_14);
    return rsi === null ? [] : [{ date: row.trade_date, rsi }];
  });

  if (data.length === 0) return null;

  return (
    <div className="h-44 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatAxisDate}
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            minTickGap={48}
            tickLine={false}
          />
          {/* A fixed 0-100 domain, because RSI's meaning is its absolute level.
              Auto-scaling would make 45 look extreme on a quiet stretch. */}
          <YAxis
            domain={[0, 100]}
            ticks={[0, 30, 50, 70, 100]}
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            width={34}
            tickLine={false}
          />
          {/* Solid hairlines, not dashes: a dashed rule reads as a projection or
              a forecast when it is only a threshold. */}
          <ReferenceLine y={RSI_OVERBOUGHT} stroke="#383835" strokeWidth={1} />
          <ReferenceLine y={RSI_OVERSOLD} stroke="#383835" strokeWidth={1} />
          <Tooltip
            cursor={{ stroke: AXIS, strokeWidth: 1 }}
            content={({ active, payload }) => {
              const point = payload?.[0]?.payload as { date: string; rsi: number } | undefined;
              if (!active || !point) return null;
              const state =
                point.rsi >= RSI_OVERBOUGHT
                  ? "overbought"
                  : point.rsi <= RSI_OVERSOLD
                    ? "oversold"
                    : "neutral";
              return (
                <TooltipCard
                  date={point.date}
                  rows={[
                    { label: "RSI (14)", value: formatNumber(point.rsi, 1), color: SERIES_1 },
                    { label: "State", value: state },
                  ]}
                />
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="rsi"
            stroke={SERIES_1}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            activeDot={{ r: 4, fill: SERIES_1, stroke: "#1a1a19", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MacdPanel({ rows }: { rows: IndicatorSnapshot[] }) {
  const data = rows.flatMap((row) => {
    const macd = toNumber(row.macd);
    const signal = toNumber(row.macd_signal);
    const histogram = toNumber(row.macd_histogram);
    return macd === null || signal === null
      ? []
      : [{ date: row.trade_date, macd, signal, histogram: histogram ?? 0 }];
  });

  if (data.length === 0) return null;

  return (
    <div className="h-44 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatAxisDate}
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            minTickGap={48}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: AXIS, fontSize: 11 }}
            stroke={GRID}
            width={44}
            tickLine={false}
            tickFormatter={(value: number) => formatNumber(value, 1)}
          />
          <ReferenceLine y={0} stroke="#383835" strokeWidth={1} />
          <Tooltip
            cursor={{ fill: "#ffffff", fillOpacity: 0.04 }}
            content={({ active, payload }) => {
              const point = payload?.[0]?.payload as
                | { date: string; macd: number; signal: number; histogram: number }
                | undefined;
              if (!active || !point) return null;
              return (
                <TooltipCard
                  date={point.date}
                  rows={[
                    { label: "MACD", value: formatNumber(point.macd), color: SERIES_1 },
                    { label: "Signal", value: formatNumber(point.signal), color: SERIES_2 },
                    { label: "Histogram", value: formatNumber(point.histogram) },
                  ]}
                />
              );
            }}
          />
          {/* Three series, so a legend is present -- identity is never carried by
              colour alone. */}
          <Legend
            verticalAlign="top"
            align="right"
            height={20}
            iconType="plainline"
            iconSize={10}
            formatter={(value: string) => (
              <span className="text-2xs text-ink-secondary">{value}</span>
            )}
          />
          {/* The histogram is signed, so its bars take the diverging poles. */}
          <Bar
            dataKey="histogram"
            name="Histogram"
            radius={[1, 1, 0, 0]}
            maxBarSize={4}
            isAnimationActive={false}
          >
            {data.map((point) => (
              <Cell
                key={point.date}
                fill={point.histogram >= 0 ? UP : DOWN}
                fillOpacity={0.55}
              />
            ))}
          </Bar>
          <Line
            type="monotone"
            dataKey="macd"
            name="MACD"
            stroke={SERIES_1}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="signal"
            name="Signal"
            stroke={SERIES_2}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
