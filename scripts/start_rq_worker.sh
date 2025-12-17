#!/usr/bin/env bash
# Start RQ worker for processing queued video jobs
set -euo pipefail
if [ -z "${REDIS_URL:-}" ]; then
  echo "REDIS_URL not set. Export REDIS_URL (e.g. redis://localhost:6379/0)" >&2
  exit 2
fi

echo "Starting RQ worker (will listen to default queue)..."
exec rq worker --url "$REDIS_URL"
