/**
 * The single door to the backend.
 *
 * Nothing else in the app calls `fetch`. That is what makes the base URL, the
 * error envelope and the "capability unavailable" case one decision each rather
 * than a convention every component has to remember.
 */

import type { ApiErrorBody } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const PREFIX = "/api/v1";

/**
 * A failure the backend described in its own envelope.
 *
 * Carries the code, not just the message, because the UI branches on it: a 501
 * `capability_not_supported` is a panel that should not render, while a 429
 * `provider_quota_exceeded` is a panel that should retry later. Both are
 * "error" to a naive client and neither is worth an error page.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;
  readonly details?: Record<string, unknown>;

  constructor(status: number, body: ApiErrorBody["error"]) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.requestId = body.request_id;
    this.details = body.details;
  }

  /** No configured provider serves this data. A UI state, not a fault. */
  get isUnsupported(): boolean {
    return this.status === 501 || this.code === "capability_not_supported";
  }

  /** The free tier is spent. Worth saying plainly rather than showing "error". */
  get isQuotaExceeded(): boolean {
    return this.status === 429 || this.code === "provider_quota_exceeded";
  }

  get isNotFound(): boolean {
    return this.status === 404;
  }
}

interface RequestOptions {
  signal?: AbortSignal;
  /** Query parameters. Undefined and null values are dropped rather than sent
   *  as the strings "undefined" and "null", which is what a naive builder does
   *  and what makes a filter mysteriously match nothing. */
  params?: Record<string, string | number | boolean | undefined | null>;
}

function buildUrl(path: string, params?: RequestOptions["params"]): string {
  const url = new URL(`${PREFIX}${path}`, BASE_URL);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

async function request<T>(
  path: string,
  { method = "GET", ...options }: RequestOptions & { method?: string } = {},
): Promise<T> {
  const response = await fetch(buildUrl(path, options.params), {
    method,
    signal: options.signal,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    // The backend always answers with its envelope, but a proxy, a crash or a
    // CORS rejection will not -- so a body that cannot be parsed becomes a
    // synthetic envelope rather than a second, differently-shaped failure mode
    // for callers to handle.
    let body: ApiErrorBody["error"];
    try {
      body = ((await response.json()) as ApiErrorBody).error;
    } catch {
      body = {
        code: "unexpected_response",
        message: `The API returned ${response.status} ${response.statusText}.`,
      };
    }
    throw new ApiError(response.status, body);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string, options?: RequestOptions) => request<T>(path, options),
  post: <T>(path: string, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "POST" }),
};
