#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
python -m jira_categorizer.enclave.server --config "${ROOT}/config/default.yaml" "$@"
