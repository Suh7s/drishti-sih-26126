#!/usr/bin/env bash
set -euo pipefail
DRISHTI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ISAAC_SIM_PATH:-}" || ! -x "$ISAAC_SIM_PATH/python.sh" ]]; then
  echo 'Set ISAAC_SIM_PATH to the extracted Isaac Sim directory containing python.sh.'
  echo 'Example: export ISAAC_SIM_PATH=/home/yourname/isaacsim'
  exit 2
fi
exec "$ISAAC_SIM_PATH/python.sh" "$DRISHTI_ROOT/isaac/run_spot.py" "$@"
