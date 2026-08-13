#!/usr/bin/env bash
# Starts the ACTG175 Bayesian decision-support Flask API (Linux venv).
#
# The Windows .venv cannot be reached from WSL2 networking, so the server
# runs from the Linux venv symlinked as ./.venv-linux.
#
# PYTHONPYCACHEPREFIX keeps bytecode out of the repo and out of the
# Windows filesystem (huge import speedup on WSL2 /drives).
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.pycache}"

if [ ! -x ./.venv-linux/bin/python ]; then
  echo "error: ./.venv-linux not found. See the frontend setup notes." >&2
  exit 1
fi

exec ./.venv-linux/bin/python src/api/app.py "$@"
