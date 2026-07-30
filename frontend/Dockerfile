# syntax=docker/dockerfile:1.7
#
# The frontend ships as static files behind nginx, not as a Node process. A Vite
# dev server in production would be slower, larger and would serve unminified
# modules; the build output is plain assets that a web server does better.

# ---------------------------------------------------------------------------
# Stage 1 - build
# ---------------------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /build

# Manifests first, so the dependency layer is cached and only rebuilt when
# dependencies actually change rather than on every source edit.
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src

# `npm run build` runs tsc before vite, so a type error fails the image build
# rather than shipping.
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 - serve
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost/ || exit 1
