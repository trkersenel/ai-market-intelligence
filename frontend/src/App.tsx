/** Routes and the query client. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Skeleton } from "./components/states";

// Route-level code splitting. Both data pages pull in the charting chunk; the
// chat page does not, so a visitor who only asks a question never downloads it.
const Dashboard = lazy(async () => ({ default: (await import("./pages/Dashboard")).Dashboard }));
const Companies = lazy(async () => ({ default: (await import("./pages/Companies")).Companies }));
const Chat = lazy(async () => ({ default: (await import("./pages/Chat")).Chat }));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // End-of-day data does not change minute to minute, so a short stale
      // window avoids refetching on every navigation without going stale in a
      // way a reader would notice.
      staleTime: 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Routes>
        <Route
          element={
            <Suspense
              fallback={
                <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
                  <Skeleton className="h-64" />
                </div>
              }
            >
              <Layout />
            </Suspense>
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="companies" element={<Companies />} />
          <Route path="companies/:slug" element={<Companies />} />
          <Route path="chat" element={<Chat />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </QueryClientProvider>
  );
}
