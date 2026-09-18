#!/usr/bin/env bash
# SIH26145 — start/stop the inference API and the dashboard static server.
#
# pkill -f <pattern> is deliberately NOT used here: the pattern matches the
# invoking shell's own command line, so it kills its own parent. PIDs are
# tracked in data/run/ instead.
#
# Usage: bash scripts/serve.sh start|stop|restart|status
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/data/run"
API_PORT=8200
DASH_PORT=8401
mkdir -p "$RUN" "$ROOT/data"

_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

_stop_one() {
  local pidf="$1" name="$2"
  if _alive "$pidf"; then
    local pid; pid="$(cat "$pidf")"
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -9 "$pid" 2>/dev/null
    echo "  stopped $name (pid $pid)"
  else
    echo "  $name not running"
  fi
  rm -f "$pidf"
}

start() {
  if _alive "$RUN/api.pid"; then
    echo "  api already running (pid $(cat "$RUN/api.pid"))"
  else
    ( cd "$ROOT" && setsid "$PY" -m uvicorn serving.app:app \
        --host 127.0.0.1 --port "$API_PORT" --log-level warning \
        > "$ROOT/data/api.log" 2>&1 &
      echo $! > "$RUN/api.pid" )
    echo "  api    -> http://127.0.0.1:$API_PORT  (log data/api.log)"
  fi
  if _alive "$RUN/dash.pid"; then
    echo "  dashboard already running (pid $(cat "$RUN/dash.pid"))"
  else
    ( cd "$ROOT" && setsid "$PY" -m http.server "$DASH_PORT" \
        --bind 127.0.0.1 --directory dashboard > "$ROOT/data/dash.log" 2>&1 &
      echo $! > "$RUN/dash.pid" )
    echo "  console-> http://127.0.0.1:$DASH_PORT"
  fi
  for _ in $(seq 1 40); do
    curl -sf --max-time 2 "http://127.0.0.1:$API_PORT/health" >/dev/null && break
    sleep 0.5
  done
  status
}

stop() { _stop_one "$RUN/api.pid" api; _stop_one "$RUN/dash.pid" dashboard; }

status() {
  local h
  h=$(curl -sf --max-time 3 "http://127.0.0.1:$API_PORT/health" || echo "")
  if [ -n "$h" ]; then echo "  health: $h"; else echo "  health: API NOT RESPONDING"; fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 2 ;;
esac
