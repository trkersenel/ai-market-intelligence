"use client";

/**
 * Global symbol search.
 *
 * The primary way into the app: ⌘K from anywhere, type, arrow, enter. It
 * searches all ~5,700 stored NASDAQ listings, which is only affordable because
 * the backend serves them from PostgreSQL rather than proxying the provider --
 * a search box fires a request per keystroke and the free tier allows sixty a
 * minute.
 *
 * Each result says whether the platform *analyses* that symbol. That is the one
 * distinction a user needs to understand about this product, so it is stated at
 * the moment they choose, not discovered on the page afterwards.
 */

import { AnimatePresence, motion } from "motion/react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, CornerDownLeft, Search } from "lucide-react";
import { useUniverseSearch } from "@/lib/api/hooks";
import { cn } from "@/lib/cn";
import { Badge } from "@/components/ui/primitives";

/** Debounce, in ms. Long enough to skip intermediate keystrokes, short enough
 *  that the list still feels attached to the keyboard. */
const DEBOUNCE = 140;

export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
        >
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-[2px]"
            onClick={() => onOpenChange(false)}
            aria-hidden
          />
          {/* The stateful half lives in its own component that exists only
              while the palette is open. Closing unmounts it, which resets the
              query and the highlight for free -- no effect clearing state on
              reopen, and so no cascading render on the way in. */}
          <PaletteContent onClose={() => onOpenChange(false)} />
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

function PaletteContent({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query), DEBOUNCE);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    // The palette animates in; focusing on the next frame avoids the browser
    // scrolling a not-yet-positioned element into view.
    const frame = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, []);

  const { data: results = [], isFetching } = useUniverseSearch(debounced);

  // Clamped during render rather than reset from an effect. The result list
  // shrinks as the query narrows, and an index left pointing past the end would
  // make Enter do nothing -- with no visible highlight to explain why.
  const highlighted = Math.min(active, Math.max(results.length - 1, 0));

  const select = useCallback(
    (symbol: string) => {
      onClose();
      router.push(`/symbol/${symbol}`);
    },
    [onClose, router],
  );

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive(Math.min(highlighted + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive(Math.max(highlighted - 1, 0));
    } else if (event.key === "Enter" && results[highlighted]) {
      event.preventDefault();
      select(results[highlighted].symbol);
    } else if (event.key === "Escape") {
      onClose();
    }
  }

  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-label="Search symbols"
      className={cn(
        "relative w-full max-w-xl overflow-hidden rounded-xl border shadow-2xl",
        "border-[var(--color-line-strong)] bg-[var(--color-surface)]",
      )}
      initial={{ opacity: 0, y: -8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      transition={{ duration: 0.14, ease: [0.2, 0, 0, 1] }}
    >
      <div className="flex items-center gap-2.5 border-b border-[var(--color-line)] px-4">
        <Search size={16} className="shrink-0 text-[var(--color-subtle)]" aria-hidden />
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActive(0);
          }}
          onKeyDown={onKeyDown}
          placeholder="Search 5,600+ NASDAQ symbols…"
          aria-label="Search symbols"
          className="w-full bg-transparent py-3.5 text-sm outline-none placeholder:text-[var(--color-subtle)]"
        />
        {isFetching ? (
          <span className="shrink-0 text-[10px] text-[var(--color-subtle)]">…</span>
        ) : null}
      </div>

      <ul className="scroll-thin max-h-[52vh] overflow-y-auto p-1.5" role="listbox">
        {results.map((listing, index) => (
          <li key={listing.symbol}>
            <button
              role="option"
              aria-selected={index === highlighted}
              onMouseEnter={() => setActive(index)}
              onClick={() => select(listing.symbol)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition-colors",
                index === highlighted
                  ? "bg-[var(--color-raised)]"
                  : "hover:bg-[var(--color-raised)]",
              )}
            >
              <span className="tnum w-16 shrink-0 text-sm font-semibold text-[var(--color-ink)]">
                {listing.symbol}
              </span>
              <span className="min-w-0 flex-1 truncate text-xs text-[var(--color-muted)]">
                {listing.name}
              </span>
              {listing.is_tracked ? (
                <Badge tone="accent">
                  <Activity size={9} aria-hidden />
                  Analysed
                </Badge>
              ) : (
                <span className="shrink-0 text-[10px] text-[var(--color-subtle)]">
                  {listing.security_type}
                </span>
              )}
            </button>
          </li>
        ))}

        {debounced && results.length === 0 && !isFetching ? (
          <li className="px-3 py-8 text-center text-xs text-[var(--color-subtle)]">
            No symbol matches “{debounced}”.
          </li>
        ) : null}

        {!debounced ? (
          <li className="px-3 py-8 text-center text-xs text-[var(--color-subtle)]">
            Type a ticker or company name.
          </li>
        ) : null}
      </ul>

      <div className="flex items-center gap-3 border-t border-[var(--color-line)] px-4 py-2 text-[10px] text-[var(--color-subtle)]">
        <span className="inline-flex items-center gap-1">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd> navigate
        </span>
        <span className="inline-flex items-center gap-1">
          <Kbd>
            <CornerDownLeft size={9} aria-hidden />
          </Kbd>
          open
        </span>
        <span className="inline-flex items-center gap-1">
          <Kbd>esc</Kbd> close
        </span>
      </div>
    </motion.div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex h-4 min-w-4 items-center justify-center rounded border border-[var(--color-line)] bg-[var(--color-canvas)] px-1 font-sans text-[10px] text-[var(--color-muted)]">
      {children}
    </kbd>
  );
}
