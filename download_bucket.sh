#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VEON_SYNC_PYTHON:-python3}" -u "$SCRIPT_DIR/tools/download_bucket.py" "$@"
