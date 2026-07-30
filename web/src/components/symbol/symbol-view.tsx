"use client";

import Image from "next/image";
import { useState } from "react";
import { Activity, ExternalLink, Info } from "lucide-react";
import { PriceChart } from "@/components/chart/price-chart";
import {
  EarningsPanel,
  InsidersPanel,
  MetricsPanel,
  NewsPanel,
  QuoteStats,
  RatingsPanel,
} from "@/components/symbol/panels";
import {
  Badge,
  DeltaText,
  EmptyState,
  Panel,
  PanelHeader,
  SegmentedControl,
  Skeleton,
} from "@/components/ui/primitives";
import { ApiError } from "@/lib/api/client";
import {
  CHART_RANGES,
  useCandles,
  useListing,
  useProfile,
  useQuote,
  type ChartRange,
} from "@/lib/api/hooks";
import { directionOf, formatPercent, formatPrice } from "@/lib/format";

export function SymbolView({ symbol }: { symbol: string }) {
  const [range, setRange] = useState<ChartRange>("1Y");

  const listing = useListing(symbol);
  const profile = useProfile(symbol);
  const quote = useQuote(symbol);
  const candles = useCandles(symbol, range);

  const currency = quote.data?.currency ?? profile.data?.currency ?? "USD";

  if (listing.error instanceof ApiError && listing.error.isNotFound) {
    return (
      <Panel className="mt-8">
        <EmptyState
          title={`${symbol} is not in the stored universe`}
          description="Only NASDAQ listings are synced. Press ⌘K to search the ones that are."
        />
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <Header
        symbol={symbol}
        name={profile.data?.name ?? listing.data?.name}
        logo={profile.data?.logo_url}
        industry={profile.data?.industry}
        website={profile.data?.website}
        exchange={listing.data?.exchange}
        isTracked={listing.data?.is_tracked}
        price={quote.data?.price ?? null}
        change={quote.data?.change ?? null}
        changePercent={quote.data?.change_percent ?? null}
        session={quote.data?.session}
        currency={currency}
        loading={quote.isPending || profile.isPending}
      />

      <Panel>
        <PanelHeader
          title="Price"
          subtitle={candles.data ? `${candles.data.candles.length} bars · ${candles.data.interval}` : undefined}
          action={
            <SegmentedControl
              label="Chart range"
              options={CHART_RANGES}
              value={range}
              onChange={setRange}
            />
          }
        />
        <div className="p-2">
          {candles.isPending ? (
            <Skeleton className="h-[340px] w-full" />
          ) : candles.error ? (
            <ChartFailure error={candles.error} />
          ) : candles.data.candles.length === 0 ? (
            <EmptyState
              title="No bars for this range"
              description="The provider returned an empty series. A shorter range may have data."
            />
          ) : (
            <PriceChart candles={candles.data.candles} range={range} currency={currency} />
          )}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <MetricsPanel symbol={symbol} />
          <EarningsPanel symbol={symbol} />
          <InsidersPanel symbol={symbol} />
        </div>

        <div className="space-y-4">
          <QuoteStats
            open={quote.data?.open ?? null}
            high={quote.data?.high ?? null}
            low={quote.data?.low ?? null}
            previousClose={quote.data?.previous_close ?? null}
            volume={quote.data?.volume ?? null}
            averageVolume={quote.data?.average_volume ?? null}
            week52High={quote.data?.week_52_high ?? null}
            week52Low={quote.data?.week_52_low ?? null}
            marketCap={profile.data?.market_cap ?? null}
            currency={currency}
          />
          <RatingsPanel symbol={symbol} />
          <NewsPanel symbol={symbol} />
        </div>
      </div>

      {profile.data?.description ? (
        <Panel>
          <PanelHeader title="About" />
          <p className="p-4 text-xs leading-relaxed text-[var(--color-muted)]">
            {profile.data.description}
          </p>
        </Panel>
      ) : null}
    </div>
  );
}

function ChartFailure({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.isUnsupported) {
    return (
      <EmptyState
        title="Charts need a provider that serves price history"
        description="The current configuration has no candle provider. Adding one makes this chart appear with no other change."
      />
    );
  }
  return (
    <EmptyState
      title="Could not load the chart"
      description={error instanceof Error ? error.message : undefined}
    />
  );
}

function Header({
  symbol,
  name,
  logo,
  industry,
  website,
  exchange,
  isTracked,
  price,
  change,
  changePercent,
  session,
  currency,
  loading,
}: {
  symbol: string;
  name?: string;
  logo?: string | null;
  industry?: string | null;
  website?: string | null;
  exchange?: string;
  isTracked?: boolean;
  price: string | null;
  change: string | null;
  changePercent: number | null;
  session?: string;
  currency: string;
  loading: boolean;
}) {
  const direction = directionOf(changePercent);

  return (
    <div className="flex flex-wrap items-start justify-between gap-4 pt-1">
      <div className="flex min-w-0 items-center gap-3">
        {logo ? (
          // The provider serves official logos over https from its own CDN.
          // `unoptimized` because Next's optimiser would proxy every one of
          // 5,664 possible logos through the server for no benefit -- they are
          // already small, already cached, and already the right size.
          <Image
            src={logo}
            alt=""
            width={40}
            height={40}
            unoptimized
            className="size-10 shrink-0 rounded-lg bg-white object-contain p-1"
          />
        ) : (
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--color-raised)] text-xs font-semibold text-[var(--color-subtle)]">
            {symbol.slice(0, 2)}
          </div>
        )}

        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold tracking-tight">{symbol}</h1>
            {exchange ? <Badge>{exchange}</Badge> : null}
            {isTracked ? (
              <Badge tone="accent">
                <Activity size={9} aria-hidden />
                Analysed
              </Badge>
            ) : (
              <Badge>
                <Info size={9} aria-hidden />
                Browse only
              </Badge>
            )}
          </div>
          <p className="mt-0.5 flex items-center gap-2 text-xs text-[var(--color-subtle)]">
            <span className="truncate">{name ?? "—"}</span>
            {industry ? (
              <>
                <span aria-hidden>·</span>
                <span className="truncate">{industry}</span>
              </>
            ) : null}
            {website ? (
              <a
                href={website}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-0.5 hover:text-[var(--color-accent)]"
              >
                <ExternalLink size={10} aria-hidden />
              </a>
            ) : null}
          </p>
        </div>
      </div>

      <div className="text-right">
        {loading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-7 w-28" />
            <Skeleton className="h-3 w-20" />
          </div>
        ) : (
          <>
            <div className="tnum text-2xl font-semibold tracking-tight">
              {formatPrice(price, currency)}
            </div>
            <div className="mt-0.5 flex items-center justify-end gap-2 text-xs">
              <DeltaText direction={direction}>
                {formatPrice(change, currency)} ({formatPercent(changePercent, { signed: true })})
              </DeltaText>
              {session ? (
                <span className="text-[10px] uppercase tracking-wide text-[var(--color-subtle)]">
                  {session}
                </span>
              ) : null}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
