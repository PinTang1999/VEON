#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for entry in tools/__init__.py tools/dtf_visualization.py; do
  if [[ ! -f "$entry" ]]; then
    printf 'Missing %s/%s. Sync the VEON tools directory to this checkout.\n' "$PWD" "$entry" >&2
    exit 1
  fi
done
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  run|export|plot|-h|--help) ;;
  *) set -- run "$@" ;;
esac

# Use the server environment for inference. Offline plotting uses local Python.
if [[ "$1" == run || "$1" == export ]]; then
  if [[ -z "${PYTHON:-}" && "${CONDA_DEFAULT_ENV:-}" != veon ]] && command -v conda >/dev/null; then
    exec conda run --no-capture-output -n veon python -m tools.dtf_visualization "$@"
  fi
fi
exec "${PYTHON:-python3}" -m tools.dtf_visualization "$@"
