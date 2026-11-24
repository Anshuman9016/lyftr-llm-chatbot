# ---------- build stage: compile the React app ----------
FROM node:22-alpine AS builder
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# ---------- runtime stage: static files served by nginx ----------
FROM nginx:1.27-alpine AS runtime
ENV API_URL=http://localhost:8000
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.d/40-runtime-config.sh /docker-entrypoint.d/40-runtime-config.sh
RUN chmod +x /docker-entrypoint.d/40-runtime-config.sh
COPY --from=builder /build/dist /usr/share/nginx/html
EXPOSE 3000
