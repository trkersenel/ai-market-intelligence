"use client";

/**
 * The application chrome: a top bar carrying navigation, the search trigger and
 * the theme toggle, with the command palette bound to ⌘K globally.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { Search, Moon, Sun, LineChart } from "lucide-react";
import { CommandPalette } from "@/components/search/command-palette";
import { cn } from "@/lib/cn";

const NAV = [{ href: "/", label: "Dashboard" }] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);

  // ⌘K / Ctrl+K from anywhere. Registered on the document rather than on an
  // element so it works regardless of where focus currently sits.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-40 border-b border-[var(--color-line)] bg-[color-mix(in_oklch,var(--color-canvas)_88%,transparent)] backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-4 sm:px-6">
          <Link href="/" className="flex shrink-0 items-center gap-2">
            <LineChart size={18} className="text-[var(--color-accent)]" aria-hidden />
            <span className="text-sm font-semibold tracking-tight">Market Intelligence</span>
          </Link>

          <nav className="hidden items-center gap-1 sm:flex">
            {NAV.map((item) => {
              const active =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
                    active
                      ? "bg-[var(--color-raised)] text-[var(--color-ink)]"
                      : "text-[var(--color-subtle)] hover:text-[var(--color-muted)]",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex flex-1 items-center justify-end gap-2">
            <button
              onClick={() => setPaletteOpen(true)}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-[var(--color-line)]",
                "bg-[var(--color-surface)] px-2.5 py-1.5 text-xs text-[var(--color-subtle)]",
                "transition-colors hover:border-[var(--color-line-strong)] hover:text-[var(--color-muted)]",
                "min-w-[13rem] justify-between",
              )}
            >
              <span className="inline-flex items-center gap-2">
                <Search size={13} aria-hidden />
                Search symbols
              </span>
              <kbd className="rounded border border-[var(--color-line)] px-1 text-[10px]">⌘K</kbd>
            </button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6">{children}</main>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}

/**
 * Dark/light toggle.
 *
 * `data-theme` on `<html>` is the single hook the token layer keys off, and the
 * DOM -- not React -- is where it lives: an inline script in the document head
 * sets it before first paint, so a light-theme user never sees a dark flash.
 *
 * That makes the attribute an *external store*, which is exactly what
 * `useSyncExternalStore` exists for. Mirroring it into `useState` from an
 * effect would work but is the pattern React 19 warns about: it renders once
 * with the wrong value and then immediately again with the right one, and the
 * server render would disagree with the client's first paint.
 */
function subscribeToTheme(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  return () => observer.disconnect();
}

function readTheme(): "dark" | "light" {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function ThemeToggle() {
  // The server has no DOM, so it renders the default the stylesheet also
  // assumes. The inline head script has already corrected the attribute by the
  // time the client subscribes.
  const theme = useSyncExternalStore(subscribeToTheme, readTheme, () => "dark" as const);

  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  }

  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className="rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] p-1.5 text-[var(--color-subtle)] transition-colors hover:text-[var(--color-muted)]"
    >
      {theme === "dark" ? <Sun size={14} aria-hidden /> : <Moon size={14} aria-hidden />}
    </button>
  );
}
