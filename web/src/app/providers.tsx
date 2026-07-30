"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

/**
 * Client-side providers.
 *
 * The `QueryClient` is created inside `useState` rather than at module scope so
 * each browser session gets its own. A module-level client is shared across
 * requests on the server, which on a multi-user deployment means one visitor's
 * cached quotes can be served to another.
 */
export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Market data is fetched on a schedule that the individual hooks
            // set deliberately; refetching every time the window regains focus
            // would override those and spend quota on tab-switching.
            refetchOnWindowFocus: false,
            gcTime: 30 * 60 * 1000,
          },
        },
      }),
  );

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
