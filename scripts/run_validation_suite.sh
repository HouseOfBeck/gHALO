#!/usr/bin/env bash

# shellcheck source-path=SCRIPTDIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/run_validation_suite.sh [options]

Options:
  --system NAME             System configuration name. Default: frontier.
  --rocm-version VERSION    Expected normalized ROCm version. Default: 6.4.2.
  --target-seconds SECONDS  Target seconds per halo size. Default: 0.1.
  --min-halo N              Minimum halo length. Default: 2.
  --max-halo N              Maximum halo length. Default: 1024.
  --halo-multiplier N       Halo length multiplier. Default: 2.
  --logs DIR                Harness log directory.
  -h, --help                Show this help.

The suite runs:
  MPI-HIP 1 node
  MPI-HIP 2 nodes
  RCCL conservative 1 node
  RCCL conservative 2 nodes
  RCCL stream-ordered 1 node
  RCCL stream-ordered 2 nodes
EOF
}

root="$(ghalo_repo_root)"
system="frontier"
rocm_version="6.4.2"
target_seconds="0.1"
min_halo="2"
max_halo="1024"
halo_multiplier="2"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      [[ $# -ge 2 ]] || ghalo_die "--system requires a value"
      system="$2"
      shift 2
      ;;
    --rocm-version)
      [[ $# -ge 2 ]] || ghalo_die "--rocm-version requires a value"
      rocm_version="$(ghalo_normalize_rocm_version "$2")"
      shift 2
      ;;
    --target-seconds)
      [[ $# -ge 2 ]] || ghalo_die "--target-seconds requires a value"
      target_seconds="$2"
      shift 2
      ;;
    --min-halo)
      [[ $# -ge 2 ]] || ghalo_die "--min-halo requires a value"
      min_halo="$2"
      shift 2
      ;;
    --max-halo)
      [[ $# -ge 2 ]] || ghalo_die "--max-halo requires a value"
      max_halo="$2"
      shift 2
      ;;
    --halo-multiplier)
      [[ $# -ge 2 ]] || ghalo_die "--halo-multiplier requires a value"
      halo_multiplier="$2"
      shift 2
      ;;
    --logs)
      [[ $# -ge 2 ]] || ghalo_die "--logs requires a value"
      log_dir="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      ghalo_die "unknown argument: $1"
      ;;
  esac
done

for value_name in min_halo max_halo halo_multiplier; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
    ghalo_die "--${value_name//_/-} must be a positive integer"
  fi
done
[[ "${halo_multiplier}" -gt 1 ]] ||
  ghalo_die "--halo-multiplier must be greater than 1"
[[ "${max_halo}" -ge "${min_halo}" ]] ||
  ghalo_die "--max-halo must be greater than or equal to --min-halo"

command -v jq >/dev/null 2>&1 ||
  ghalo_die "jq is required to validate ghalo.json"

if [[ -z "${log_dir}" ]]; then
  log_dir="${root}/test-logs/${system}-rocm-${rocm_version}-validation-${timestamp}"
fi
mkdir -p "${log_dir}"

declare -a PASSED_CASES=()
declare -a FAILED_CASES=()
declare -a RESULT_DIRS=()

record_pass() {
  PASSED_CASES+=("$1")
}

record_fail() {
  local label="$1"
  local reason="$2"
  FAILED_CASES+=("${label}: ${reason}")
  printf 'FAIL: %s: %s\n' "${label}" "${reason}" >&2
}

require_file() {
  local path="$1"
  local description="$2"
  [[ -f "${path}" ]] || {
    printf 'missing %s: %s\n' "${description}" "${path}"
    return 1
  }
}

require_metadata_key() {
  local path="$1"
  local key="$2"
  grep -Eq "^${key}=" "${path}" || {
    printf 'missing metadata key %s in %s\n' "${key}" "${path}"
    return 1
  }
}

verify_result_bundle() {
  local label="$1"
  local result_dir="$2"
  local expected_backend="$3"
  local expected_nodes="$4"
  local expected_ranks="$5"
  local expected_ranks_per_node="$6"
  local expected_sync_mode="$7"

  local json_path="${result_dir}/ghalo.json"
  local system_metadata="${result_dir}/system-resolution.txt"
  local result_metadata="${result_dir}/result-metadata.txt"
  local exit_status_path="${result_dir}/exit-status.txt"
  local detected_nodes
  local detected_ranks

  require_file "${json_path}" "ghalo.json" || return 1
  require_file "${system_metadata}" "system metadata" || return 1
  require_file "${result_metadata}" "result metadata" || return 1
  require_file "${exit_status_path}" "exit status" || return 1

  [[ "$(<"${exit_status_path}")" == "0" ]] || {
    printf 'benchmark exit-status.txt is nonzero: %s\n' "$(<"${exit_status_path}")"
    return 1
  }

  jq -e '(.results | type == "array") and (.results | length > 0)' \
    "${json_path}" >/dev/null ||
    { printf 'ghalo.json lacks a non-empty results array\n'; return 1; }
  jq -e --arg backend "${expected_backend}" \
    '[.results[] | select(.backend != $backend)] | length == 0' \
    "${json_path}" >/dev/null ||
    { printf 'unexpected backend in ghalo.json\n'; return 1; }
  jq -e --argjson ranks "${expected_ranks}" \
    '[.results[] | select(.topology.world_size != $ranks)] | length == 0' \
    "${json_path}" >/dev/null ||
    { printf 'unexpected rank count in ghalo.json\n'; return 1; }
  detected_nodes="$(
    jq -r '[.results[0].metadata.ranks[]?.hostname |
             select(. != null and . != "")] | unique | length' "${json_path}"
  )"
  if [[ "${detected_nodes}" != "${expected_nodes}" ]]; then
    printf 'detected node count mismatch: requested %s, detected %s unique hostnames\n' \
      "${expected_nodes}" "${detected_nodes}"
    return 1
  fi
  detected_ranks="$(
    jq -r '[.results[0].metadata.ranks[]?] | length' "${json_path}"
  )"
  if [[ "${detected_ranks}" != "${expected_ranks}" ]]; then
    printf 'detected rank metadata mismatch: requested %s ranks, found %s rank metadata entries\n' \
      "${expected_ranks}" "${detected_ranks}"
    return 1
  fi
  if [[ "${expected_ranks_per_node}" -gt 0 &&
        "${detected_nodes}" -gt 0 &&
        ( $((detected_ranks % detected_nodes)) -ne 0 ||
          $((detected_ranks / detected_nodes)) -ne "${expected_ranks_per_node}" ) ]]; then
    printf 'detected ranks-per-node mismatch: requested %s, detected %s\n' \
      "${expected_ranks_per_node}" "$((detected_ranks / detected_nodes))"
    return 1
  fi
  jq -e '[.results[] | select(.metadata.validation_enabled != true)] | length == 0' \
    "${json_path}" >/dev/null ||
    { printf 'validation_enabled is not true for every result\n'; return 1; }
  jq -e '[.results[] | select(.metadata.validation_passed != true)] | length == 0' \
    "${json_path}" >/dev/null ||
    { printf 'validation_passed is not true for every result\n'; return 1; }

  if [[ -n "${expected_sync_mode}" ]]; then
    jq -e --arg mode "${expected_sync_mode}" \
      '[.results[] | select(.metadata.rccl_sync_mode != $mode)] | length == 0' \
      "${json_path}" >/dev/null ||
      { printf 'unexpected RCCL synchronization mode in ghalo.json\n'; return 1; }
  fi

  require_metadata_key "${system_metadata}" active_system || return 1
  require_metadata_key "${system_metadata}" build_system || return 1
  require_metadata_key "${system_metadata}" backend || return 1
  require_metadata_key "${system_metadata}" binary || return 1
  require_metadata_key "${result_metadata}" result_layout || return 1
  require_metadata_key "${result_metadata}" result_category || return 1
  require_metadata_key "${result_metadata}" rocm_version || return 1
  require_metadata_key "${result_metadata}" min_halo || return 1
  require_metadata_key "${result_metadata}" max_halo || return 1
  require_metadata_key "${result_metadata}" halo_multiplier || return 1
  require_metadata_key "${result_metadata}" requested_nodes || return 1
  require_metadata_key "${result_metadata}" requested_ranks || return 1
  require_metadata_key "${result_metadata}" requested_ranks_per_node || return 1

  grep -qx "active_system=${system}" "${system_metadata}" ||
    { printf 'active_system metadata mismatch\n'; return 1; }
  grep -qx "result_category=validation" "${result_metadata}" ||
    { printf 'result_category metadata mismatch\n'; return 1; }
  grep -qx "rocm_version=${rocm_version}" "${result_metadata}" ||
    { printf 'rocm_version metadata mismatch\n'; return 1; }
  grep -qx "requested_nodes=${expected_nodes}" "${result_metadata}" ||
    { printf 'requested_nodes metadata mismatch\n'; return 1; }
  grep -qx "requested_ranks=${expected_ranks}" "${result_metadata}" ||
    { printf 'requested_ranks metadata mismatch\n'; return 1; }
  grep -qx "requested_ranks_per_node=${expected_ranks_per_node}" "${result_metadata}" ||
    { printf 'requested_ranks_per_node metadata mismatch\n'; return 1; }
}

run_validation_case() {
  local label="$1"
  local backend="$2"
  local nodes="$3"
  local ranks="$4"
  local ranks_per_node="$5"
  local expected_backend="$6"
  local sync_mode="$7"
  local result_label="$8"
  local log
  local result_dir=""
  local status
  local verify_output

  log="${log_dir}/$(ghalo_sanitize_label "${label}").log"

  printf '\n============================================================\n'
  printf '%s\n' "${label}"
  printf '============================================================\n'

  local command=(
    env "GHALO_SYSTEM_NAME=${system}"
    "${root}/scripts/run.sh"
    --backend "${backend}"
    --nodes "${nodes}"
    --ranks "${ranks}"
    --ranks-per-node "${ranks_per_node}"
    --validate
    --target-seconds "${target_seconds}"
    --min-halo "${min_halo}"
    --max-halo "${max_halo}"
    --halo-multiplier "${halo_multiplier}"
    --category validation
    --label "${result_label}"
  )
  if [[ -n "${sync_mode}" ]]; then
    command+=(--rccl-sync-mode "${sync_mode}")
  fi

  set +e
  "${command[@]}" > >(tee "${log}") 2>&1
  status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    record_fail "${label}" "run command exited with status ${status}; see ${log}"
    return
  fi

  result_dir="$(sed -n 's/^Result directory: //p' "${log}" | tail -n 1)"
  if [[ -z "${result_dir}" ]]; then
    record_fail "${label}" "run output did not report a result directory; see ${log}"
    return
  fi

  if [[ -x "${SCRIPT_DIR}/capture_environment.sh" ]]; then
    "${SCRIPT_DIR}/capture_environment.sh" "${result_dir}"
  fi

  if verify_output="$(verify_result_bundle "${label}" "${result_dir}" \
      "${expected_backend}" "${nodes}" "${ranks}" "${ranks_per_node}" \
      "${sync_mode}" 2>&1)"; then
    record_pass "${label}"
    RESULT_DIRS+=("${result_dir}")
    printf 'PASS: %s\n' "${label}"
    printf 'Result directory: %s\n' "${result_dir}"
  else
    record_fail "${label}" "${verify_output}; result directory: ${result_dir}"
  fi
}

print_summary() {
  printf '\n============================================================\n'
  printf 'Validation suite summary\n'
  printf '============================================================\n'
  printf 'PASS:\n'
  if [[ "${#PASSED_CASES[@]}" -eq 0 ]]; then
    printf '  None\n'
  else
    printf '  %s\n' "${PASSED_CASES[@]}"
  fi
  printf '\nFAIL:\n'
  if [[ "${#FAILED_CASES[@]}" -eq 0 ]]; then
    printf '  None\n'
  else
    printf '  %s\n' "${FAILED_CASES[@]}"
  fi
  printf '\nBenchmark results:\n'
  if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
    printf '  None\n'
  else
    printf '  %s\n' "${RESULT_DIRS[@]}"
  fi
  printf '\nLogs:\n'
  printf '  %s\n' "${log_dir}"
}

printf 'gHALO validation suite\n'
printf '  system: %s\n' "${system}"
printf '  ROCm version: %s\n' "${rocm_version}"
printf '  logs: %s\n' "${log_dir}"

run_validation_case "MPI-HIP 1 node" mpi-hip 1 8 8 MPIHIPBackend "" \
  validation-1node-8ranks
run_validation_case "MPI-HIP 2 nodes" mpi-hip 2 16 8 MPIHIPBackend "" \
  validation-2nodes-16ranks
run_validation_case "RCCL conservative 1 node" rccl 1 8 8 RCCLBackend \
  conservative conservative_validation-1node-8ranks
run_validation_case "RCCL conservative 2 nodes" rccl 2 16 8 RCCLBackend \
  conservative conservative_validation-2nodes-16ranks
run_validation_case "RCCL stream-ordered 1 node" rccl 1 8 8 RCCLBackend \
  stream-ordered stream-ordered_validation-1node-8ranks
run_validation_case "RCCL stream-ordered 2 nodes" rccl 2 16 8 RCCLBackend \
  stream-ordered stream-ordered_validation-2nodes-16ranks

print_summary

if [[ "${#FAILED_CASES[@]}" -gt 0 ]]; then
  exit 1
fi
