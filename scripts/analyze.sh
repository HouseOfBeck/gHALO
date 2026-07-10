#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python_bin="${PYTHON:-python3}"
exec "${python_bin}" "${ROOT}/tools/ghalo_analyze.py" "$@"
