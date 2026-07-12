#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/generate_analysis_report.sh --input RESULT_DIR --output ANALYSIS_DIR [options]

Options:
  --input DIR             Result tree or result bundle to analyze.
  --output DIR            Analysis report directory to create or update.
  --style NAME            Plot style: default or publication. Default: publication.
  --x-axis NAME           Scaling x-axis: auto, nodes, or ranks. Default: auto.
  --legend-position NAME  Plot legend position: auto, inside, or outside. Default: auto.
  --dpi N                 Raster output DPI passed to ghalo_analyze.py.
  -h, --help              Show this help.

Example:
  scripts/generate_analysis_report.sh \
    --input results/frontier/rocm-6.4.2/validation \
    --output analysis/frontier-rocm-6.4.2-validation
EOF
}

python_bin="${PYTHON:-python3}"
input=""
output=""
style="publication"
x_axis="auto"
legend_position="auto"
dpi=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || { echo "--input requires a value" >&2; exit 2; }
      input="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a value" >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    --style)
      [[ $# -ge 2 ]] || { echo "--style requires a value" >&2; exit 2; }
      style="$2"
      shift 2
      ;;
    --x-axis)
      [[ $# -ge 2 ]] || { echo "--x-axis requires a value" >&2; exit 2; }
      x_axis="$2"
      shift 2
      ;;
    --legend-position)
      [[ $# -ge 2 ]] || { echo "--legend-position requires a value" >&2; exit 2; }
      legend_position="$2"
      shift 2
      ;;
    --dpi)
      [[ $# -ge 2 ]] || { echo "--dpi requires a value" >&2; exit 2; }
      dpi="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${input}" ]] || { echo "--input is required" >&2; usage >&2; exit 2; }
[[ -n "${output}" ]] || { echo "--output is required" >&2; usage >&2; exit 2; }

command=(
  "${python_bin}"
  "${ROOT}/tools/ghalo_analyze.py"
  report
  --style "${style}"
  --x-axis "${x_axis}"
  --legend-position "${legend_position}"
  --output-dir "${output}"
)
if [[ -n "${dpi}" ]]; then
  command+=(--dpi "${dpi}")
fi
command+=("${input}")

printf 'Generating gHALO analysis report:\n'
printf '  input:  %s\n' "${input}"
printf '  output: %s\n' "${output}"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}"
