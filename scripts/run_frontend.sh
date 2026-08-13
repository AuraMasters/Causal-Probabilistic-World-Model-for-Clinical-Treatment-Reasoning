#!/usr/bin/env bash
# Starts the Vite dev server for the decision-support dashboard.
set -euo pipefail

cd "$(dirname "$0")/../frontend"
exec npm run dev "$@"
