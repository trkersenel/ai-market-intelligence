/**
 * A news feed with sentiment.
 *
 * Sentiment is shown as a word plus a coloured dot, never a dot alone. It is a
 * model's opinion, so the label also carries the confidence -- a bullish call at
 * 51% and one at 96% are different claims and should not look identical.
 */

import type { NewsArticle, Sentiment } from "../lib/types";
import { formatRelative } from "../lib/format";

const SENTIMENT_STYLE: Record<Sentiment, { dot: string; text: string; label: string }> = {
  bullish: { dot: "bg-up", text: "text-up", label: "bullish" },
  bearish: { dot: "bg-down", text: "text-down", label: "bearish" },
  neutral: { dot: "bg-hairline-strong", text: "text-ink-muted", label: "neutral" },
};

export function NewsList({ articles }: { articles: NewsArticle[] }) {
  return (
    <ul className="divide-y divide-hairline">
      {articles.map((article) => {
        const sentiment = article.sentiment ? SENTIMENT_STYLE[article.sentiment] : null;
        return (
          <li key={article.url} className="py-3 first:pt-0 last:pb-0">
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="focusable rounded text-sm leading-snug text-ink hover:underline"
            >
              {article.title}
            </a>

            <div className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-2xs">
              {sentiment && (
                <span className="flex items-center gap-1">
                  <span aria-hidden className={`h-2 w-2 rounded-full ${sentiment.dot}`} />
                  <span className={sentiment.text}>{sentiment.label}</span>
                  {article.sentiment_confidence !== null && (
                    <span className="tabular text-ink-muted">
                      {Math.round(article.sentiment_confidence * 100)}%
                    </span>
                  )}
                </span>
              )}
              {article.source_name && (
                <span className="text-ink-muted">{article.source_name}</span>
              )}
              <span className="tabular text-ink-muted">
                {formatRelative(article.published_at)}
              </span>
              {article.tickers.slice(0, 3).map((symbol) => (
                <span
                  key={symbol}
                  className="rounded border border-hairline px-1 py-px tabular text-ink-secondary"
                >
                  {symbol}
                </span>
              ))}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
