/** Application shell: navigation and the page container. */

import { NavLink, Outlet } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Dashboard" },
  { to: "/companies", label: "Companies" },
  { to: "/chat", label: "Ask" },
] as const;

export function Layout() {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-hairline bg-surface-page/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 sm:px-6">
          <NavLink to="/" className="focusable flex items-baseline gap-2 rounded">
            <span className="text-sm font-semibold tracking-tight">Market Intelligence</span>
            <span className="hidden text-2xs text-ink-muted sm:inline">
              AI infrastructure &amp; semiconductors
            </span>
          </NavLink>

          <nav className="ml-auto flex items-center gap-1" aria-label="Main">
            {LINKS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `focusable rounded px-2.5 py-1.5 text-xs transition-colors ${
                    isActive
                      ? "bg-surface-raised text-ink"
                      : "text-ink-secondary hover:text-ink"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 pt-2 sm:px-6">
        <p className="text-2xs text-ink-muted">
          Prices are end-of-day. Sessions still trading are marked provisional and
          excluded from every statistic. Not investment advice.
        </p>
      </footer>
    </div>
  );
}
