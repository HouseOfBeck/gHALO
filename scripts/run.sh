#!/usr/bin/env bash

# shellcheck source-path=SCRIPTDIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/run.sh --backend mpi|mpi-hip|rccl [options]

Options:
  --system NAME              System configuration name. GHALO_SYSTEM_NAME wins if set.
  --backend mpi|mpi-hip|rccl Backend to run.
  --nodes N                  Number of nodes. Default: 1.
  --ranks N                  Total MPI ranks. Default: 1.
  --ranks-per-node N         MPI ranks per node.
  --target-seconds SECONDS   gHALO target seconds per halo size. Default: 3.
  --validate                 Pass --validate to gHALO.
  --phase-timing             Pass --phase-timing to gHALO.
  --rccl-stage-b             Run RCCL north/south Stage B validation only.
  --label TEXT               Optional result directory label.
  --extra-srun-args ARGS     Extra launcher arguments for systems that use srun.
  -h, --help                 Show this help.
EOF
}

root="$(ghalo_repo_root)"
requested_system=""
backend=""
nodes="1"
ranks="1"
ranks_per_node=""
target_seconds="3"
validate=false
phase_timing=false
rccl_stage_b=false
label="run"
extra_srun_args=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      [[ $# -ge 2 ]] || ghalo_die "--system requires a value"
      requested_system="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || ghalo_die "--backend requires a value"
      backend="$2"
      shift 2
      ;;
    --nodes)
      [[ $# -ge 2 ]] || ghalo_die "--nodes requires a value"
      nodes="$2"
      shift 2
      ;;
    --ranks)
      [[ $# -ge 2 ]] || ghalo_die "--ranks requires a value"
      ranks="$2"
      shift 2
      ;;
    --ranks-per-node)
      [[ $# -ge 2 ]] || ghalo_die "--ranks-per-node requires a value"
      ranks_per_node="$2"
      shift 2
      ;;
    --target-seconds)
      [[ $# -ge 2 ]] || ghalo_die "--target-seconds requires a value"
      target_seconds="$2"
      shift 2
      ;;
    --validate)
      validate=true
      shift
      ;;
    --phase-timing)
      phase_timing=true
      shift
      ;;
    --rccl-stage-b)
      rccl_stage_b=true
      shift
      ;;
    --label)
      [[ $# -ge 2 ]] || ghalo_die "--label requires a value"
      label="$2"
      shift 2
      ;;
    --extra-srun-args)
      [[ $# -ge 2 ]] || ghalo_die "--extra-srun-args requires a value"
      extra_srun_args="$2"
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

[[ -n "${backend}" ]] || ghalo_die "--backend is required"
ghalo_validate_backend "${backend}"
for value_name in nodes ranks; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
    ghalo_die "--${value_name} must be a positive integer"
  fi
done
if [[ -n "${ranks_per_node}" &&
      ( ! "${ranks_per_node}" =~ ^[0-9]+$ || "${ranks_per_node}" -lt 1 ) ]]; then
  ghalo_die "--ranks-per-node must be a positive integer"
fi

system="$(ghalo_resolve_system "${requested_system}")"
ghalo_validate_system_name "${system}"
ghalo_load_system_config "${root}" "${system}"

export GHALO_ACTIVE_SYSTEM="${system}"
export GHALO_ACTIVE_BACKEND="${backend}"
build_system="$(ghalo_system_build_alias "${backend}")"
ghalo_validate_system_name "${build_system}"

build_dir="$(ghalo_build_dir "${root}" "${build_system}" "${backend}")"
binary="$(ghalo_binary_path "${root}" "${build_system}" "${backend}")"
ghalo_require_binary "${binary}" "${system}" "${build_system}" "${backend}"

if [[ -f "${build_dir}/build-info/system.txt" ]]; then
  built_system="$(<"${build_dir}/build-info/system.txt")"
  [[ "${built_system}" == "${build_system}" ]] ||
    ghalo_die "build directory system mismatch: expected ${build_system}, found ${built_system}"
fi
if [[ -f "${build_dir}/build-info/backend.txt" ]]; then
  built_backend="$(<"${build_dir}/build-info/backend.txt")"
  [[ "${built_backend}" == "${backend}" ]] ||
    ghalo_die "build directory backend mismatch: expected ${backend}, found ${built_backend}"
fi

ghalo_system_setup_run "${backend}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_label="$(ghalo_sanitize_label "${label}")"
result_dir="${root}/results/${system}/${timestamp}_${backend}_${safe_label}"
mkdir -p "${result_dir}"

ghalo_args=(
  --backend "${backend}"
  --target-seconds "${target_seconds}"
  --csv "${result_dir}/ghalo.csv"
  --json "${result_dir}/ghalo.json"
)
if [[ "${validate}" == true ]]; then
  ghalo_args+=(--validate)
fi
if [[ "${phase_timing}" == true ]]; then
  ghalo_args+=(--phase-timing)
fi
if [[ "${rccl_stage_b}" == true ]]; then
  ghalo_args+=(--rccl-stage-b)
fi

mapfile -t launch_command < <(
  ghalo_system_launch \
    "${backend}" \
    "${nodes}" \
    "${ranks}" \
    "${ranks_per_node}" \
    "${extra_srun_args}" \
    "${binary}" \
    "${ghalo_args[@]}"
)
[[ "${#launch_command[@]}" -gt 0 ]] ||
  ghalo_die "system launcher produced an empty command"

printf '%q ' "${launch_command[@]}" >"${result_dir}/command.txt"
printf '\n' >>"${result_dir}/command.txt"
ghalo_write_system_resolution "${result_dir}/system-resolution.txt" \
  "${system}" "${build_system}" "${backend}" "${binary}"
if [[ -n "${GHALO_SUBMIT_COMMAND:-}" || -n "${SLURM_JOB_ID:-}" ]]; then
  {
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-}"
    printf 'submission_system=%s\n' "${GHALO_SUBMISSION_SYSTEM:-${system}}"
    printf 'account=%s\n' "${GHALO_SUBMISSION_ACCOUNT:-}"
    printf 'partition=%s\n' "${GHALO_SUBMISSION_PARTITION:-}"
    printf 'node_list=%s\n' "${SLURM_JOB_NODELIST:-}"
    printf 'batch_stdout=%s\n' "${GHALO_BATCH_STDOUT:-}"
    printf 'batch_stderr=%s\n' "${GHALO_BATCH_STDERR:-}"
    printf 'submit_command=%s\n' "${GHALO_SUBMIT_COMMAND:-}"
  } >"${result_dir}/submission.txt"
fi

ghalo_capture_modules "${result_dir}/modules.txt"
ghalo_capture_environment "${result_dir}/environment.txt"
ghalo_capture_git "${result_dir}/git.txt"
hostname >"${result_dir}/hostname.txt"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  scontrol show job "${SLURM_JOB_ID}" >"${result_dir}/slurm-job.txt" 2>&1 || true
else
  printf 'No active Slurm job detected.\n' >"${result_dir}/slurm-job.txt"
fi
{
  ls -l "${binary}"
  if command -v file >/dev/null 2>&1; then
    file "${binary}" || true
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${binary}" || true
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${binary}" || true
  fi
} >"${result_dir}/binary-info.txt"

echo "Result directory: ${result_dir}"
set +e
"${launch_command[@]}" >"${result_dir}/stdout.txt" 2>"${result_dir}/stderr.txt"
status=$?
set -e
echo "${status}" >"${result_dir}/exit-status.txt"

echo "gHALO exit status: ${status}"
exit "${status}"
