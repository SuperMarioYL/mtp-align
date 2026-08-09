#!/usr/bin/env bash
# Run the MTP-Align kill-gate benchmark against a live llama.cpp server.
#
# Usage:
#   scripts/run_benchmark.sh [UPSTREAM_URL]   # default http://localhost:8080
#
# Requires a running llama.cpp server with DeepSeek-V4 + MTP/DSpark enabled.
# For a GPU-free synthetic run, use `mtp-align bench` (no --upstream) instead.
set -euo pipefail

UPSTREAM="${1:-http://localhost:8080}"

python -m mtp_align.cli bench --upstream "$UPSTREAM" \
    --tokens 512 --tool-calls 12 --window 4
