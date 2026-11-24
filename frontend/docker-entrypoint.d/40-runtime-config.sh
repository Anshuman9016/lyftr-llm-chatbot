#!/bin/sh
# Writes the backend URL from the API_URL env var into config.js at container start,
# so the same image works against any backend without rebuilding.
set -e
cat > /usr/share/nginx/html/config.js <<CFG
window.__APP_CONFIG__ = { API_URL: "${API_URL}" };
CFG
echo "runtime-config: API_URL=${API_URL}"
