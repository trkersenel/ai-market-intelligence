import type { Metadata } from "next";
import { SymbolView } from "@/components/symbol/symbol-view";

/**
 * The company page.
 *
 * A thin server component: it resolves the route parameter and hands off. All
 * the data is client-fetched because most of it refreshes on its own schedule
 * (a quote every thirty seconds) and because each panel must be able to fail
 * independently -- which is a client-side concern.
 *
 * Note the `await` on `params`: in Next 16 the route params are a Promise, and
 * synchronous access was removed rather than deprecated.
 */
export async function generateMetadata({
  params,
}: {
  params: Promise<{ symbol: string }>;
}): Promise<Metadata> {
  const { symbol } = await params;
  const upper = symbol.toUpperCase();
  return {
    title: `${upper} · Market Intelligence`,
    description: `Price, fundamentals, analyst coverage and news for ${upper}.`,
  };
}

export default async function SymbolPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  return <SymbolView symbol={symbol.toUpperCase()} />;
}
