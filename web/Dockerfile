# syntax=docker/dockerfile:1

# --- Dependencies ------------------------------------------------------------
# Separate from the build stage so a source-only change reuses the installed
# tree; npm install is by far the slowest step and source changes are the most
# frequent.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# `npm ci` rather than `npm install`: it installs exactly the lockfile and fails
# when the manifest and lock have drifted, which is the point of committing one.
RUN npm ci

# --- Build -------------------------------------------------------------------
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# The API URL is baked into the client bundle at build time -- `NEXT_PUBLIC_*`
# is a compile-time substitution, not a runtime lookup. Changing where the API
# lives therefore means rebuilding this image, not restarting the container.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --- Runtime -----------------------------------------------------------------
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Unprivileged: nothing here needs root, and the standalone server writes
# nothing to disk.
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

# The standalone output already contains a minimal node_modules. Static assets
# are not included in it and must be copied separately, or every stylesheet and
# chunk 404s while the pages themselves render.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
