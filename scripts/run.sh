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
  --min-halo N               Minimum halo length. Default: 2.
  --max-halo N               Maximum halo length. Default: 1024.
  --halo-multiplier N        Halo length multiplier. Default: 2.
  --samples-per-halo N       Independent timed samples per halo. Default: 1.
  --record-iteration-times   Write per-iteration max-rank timing diagnostics.
  --record-iteration-phase-times
                             Write per-phase timing for threshold-matched iterations.
  --record-stalled-rank-times
                             Write per-rank timing rows for stalled iterations.
  --iteration-stall-threshold-us VALUE
                             Emit iteration timing records at or above VALUE us.
                             Default/zero emits every measured iteration.
  --validate                 Pass --validate to gHALO.
  --phase-timing             Pass --phase-timing to gHALO.
  --rccl-stage-b             Run RCCL north/south Stage B validation only.
  --rccl-sync-mode MODE      Pass conservative or stream-ordered to RCCL.
  --category NAME            Result category: validation, scaling, repeatability, or phase-timing.
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
min_halo="2"
max_halo="1024"
halo_multiplier="2"
samples_per_halo="1"
record_iteration_times=false
record_iteration_phase_times=false
record_stalled_rank_times=false
iteration_stall_threshold_us=""
validate=false
phase_timing=false
rccl_stage_b=false
rccl_sync_mode=""
category=""
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
    --samples-per-halo)
      [[ $# -ge 2 ]] || ghalo_die "--samples-per-halo requires a value"
      samples_per_halo="$2"
      shift 2
      ;;
    --record-iteration-times)
      record_iteration_times=true
      shift
      ;;
    --record-iteration-phase-times)
      record_iteration_phase_times=true
      shift
      ;;
    --record-stalled-rank-times)
      record_stalled_rank_times=true
      shift
      ;;
    --iteration-stall-threshold-us)
      [[ $# -ge 2 ]] || ghalo_die "--iteration-stall-threshold-us requires a value"
      iteration_stall_threshold_us="$2"
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
    --rccl-sync-mode)
      [[ $# -ge 2 ]] || ghalo_die "--rccl-sync-mode requires a value"
      rccl_sync_mode="$2"
      case "${rccl_sync_mode}" in
        conservative | stream-ordered) ;;
        *) ghalo_die "--rccl-sync-mode must be conservative or stream-ordered" ;;
      esac
      shift 2
      ;;
    --category)
      [[ $# -ge 2 ]] || ghalo_die "--category requires a value"
      category="$2"
      ghalo_validate_result_category "${category}"
      shift 2
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
for value_name in min_halo max_halo halo_multiplier samples_per_halo; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
    ghalo_die "--${value_name//_/-} must be a positive integer"
  fi
done
[[ "${halo_multiplier}" -gt 1 ]] ||
  ghalo_die "--halo-multiplier must be greater than 1"
[[ "${max_halo}" -ge "${min_halo}" ]] ||
  ghalo_die "--max-halo must be greater than or equal to --min-halo"
if [[ -n "${iteration_stall_threshold_us}" ]]; then
  [[ "${iteration_stall_threshold_us}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    ghalo_die "--iteration-stall-threshold-us must be nonnegative"
fi
if [[ "${record_iteration_phase_times}" == true &&
      "${record_iteration_times}" != true ]]; then
  ghalo_die "--record-iteration-phase-times requires --record-iteration-times"
fi
if [[ "${record_stalled_rank_times}" == true &&
      ( "${record_iteration_times}" != true ||
        "${record_iteration_phase_times}" != true ) ]]; then
  ghalo_die "--record-stalled-rank-times requires --record-iteration-times and --record-iteration-phase-times"
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
rocm_version="$(ghalo_normalize_rocm_version)"
category="$(ghalo_infer_result_category "${category}" "${validate}" "${phase_timing}" "${label}")"
result_dir="$(ghalo_unique_result_dir "${root}" "${system}" "${rocm_version}" "${category}" "${timestamp}" "${backend}" "${label}")"
mkdir -p "${result_dir}"

ghalo_args=(
  --backend "${backend}"
  --target-seconds "${target_seconds}"
  --min-halo "${min_halo}"
  --max-halo "${max_halo}"
  --halo-multiplier "${halo_multiplier}"
  --samples-per-halo "${samples_per_halo}"
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
if [[ -n "${rccl_sync_mode}" ]]; then
  ghalo_args+=(--rccl-sync-mode "${rccl_sync_mode}")
fi
if [[ "${record_iteration_times}" == true ]]; then
  ghalo_args+=(--record-iteration-times)
fi
if [[ "${record_iteration_phase_times}" == true ]]; then
  ghalo_args+=(--record-iteration-phase-times)
fi
if [[ "${record_stalled_rank_times}" == true ]]; then
  ghalo_args+=(--record-stalled-rank-times)
fi
if [[ -n "${iteration_stall_threshold_us}" ]]; then
  ghalo_args+=(--iteration-stall-threshold-us "${iteration_stall_threshold_us}")
fi

launch_command=()
while IFS= read -r launch_arg; do
  launch_command+=("${launch_arg}")
done < <(
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
{
  printf 'result_layout=versioned\n'
  printf 'result_category=%s\n' "${category}"
  printf 'rocm_version=%s\n' "${rocm_version}"
  printf 'loaded_rocm_module=%s\n' "${GHALO_LOADED_ROCM_MODULE:-}"
  printf 'requested_rocm_version=%s\n' "${GHALO_REQUESTED_ROCM_VERSION:-${rocm_version}}"
  printf 'rocm_path=%s\n' "${ROCM_PATH:-}"
  printf 'rccl_root=%s\n' "${RCCL_ROOT:-}"
  printf 'olcf_ofi_nccl_root=%s\n' "${OLCF_OFI_NCCL_ROOT:-}"
  printf 'rccl_net_plugin_module=%s\n' "${GHALO_RCCL_NET_PLUGIN_MODULE:-}"
  printf 'expected_rccl_version_code=%s\n' "${GHALO_RCCL_VERSION_CODE_EXPECTED:-}"
  printf 'observed_rccl_version_code=%s\n' "${GHALO_RCCL_VERSION_CODE_OBSERVED:-}"
  printf 'librccl_realpath=%s\n' "${GHALO_RESOLVED_RCCL_LIBRARY_REALPATH:-${GHALO_RESOLVED_RCCL_LIBRARY:-}}"
  printf 'libamdhip64_realpath=%s\n' "${GHALO_RESOLVED_HIP_LIBRARY_REALPATH:-${GHALO_RESOLVED_HIP_LIBRARY:-}}"
  printf 'libnccl_net_realpath=%s\n' "${GHALO_RESOLVED_NCCL_NET_LIBRARY_REALPATH:-}"
  printf 'build_dir_override=%s\n' "${GHALO_BUILD_DIR_OVERRIDE:-}"
  printf 'mpich_gpu_support_enabled=%s\n' "${MPICH_GPU_SUPPORT_ENABLED:-0}"
  printf 'mpich_smp_single_copy_mode=%s\n' "${MPICH_SMP_SINGLE_COPY_MODE:-}"
  printf 'min_halo=%s\n' "${min_halo}"
  printf 'max_halo=%s\n' "${max_halo}"
  printf 'halo_multiplier=%s\n' "${halo_multiplier}"
  printf 'samples_per_halo=%s\n' "${samples_per_halo}"
  printf 'iteration_timing_enabled=%s\n' "${record_iteration_times}"
  printf 'iteration_phase_timing_enabled=%s\n' "${record_iteration_phase_times}"
  printf 'stalled_rank_timing_enabled=%s\n' "${record_stalled_rank_times}"
  printf 'iteration_stall_threshold_us=%s\n' "${iteration_stall_threshold_us:-0}"
  printf 'requested_nodes=%s\n' "${nodes}"
  printf 'requested_ranks=%s\n' "${ranks}"
  printf 'requested_ranks_per_node=%s\n' "${ranks_per_node:-}"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} >"${result_dir}/result-metadata.txt"
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
{
  if [[ "${record_iteration_times}" == true && -s "${result_dir}/iteration-times.csv" ]]; then
    iteration_records_emitted="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' \
      "${result_dir}/iteration-times.csv")"
    iteration_total_observed="$(awk -F, '
      NR == 1 {
        for (i = 1; i <= NF; ++i) {
          if ($i == "iterations") {
            iterations_column = i
          }
        }
        next
      }
      iterations_column { total += $iterations_column }
      END { print total + 0 }
    ' "${result_dir}/ghalo.csv")"
    if [[ -n "${iteration_stall_threshold_us}" &&
          "${iteration_stall_threshold_us}" != "0" &&
          "${iteration_stall_threshold_us}" != "0.0" ]]; then
      iteration_stall_count="${iteration_records_emitted}"
    else
      iteration_stall_count="0"
    fi
  else
    iteration_total_observed="0"
    iteration_records_emitted="0"
    iteration_stall_count="0"
  fi
  if [[ "${record_iteration_phase_times}" == true && -s "${result_dir}/iteration-phase-times.csv" ]]; then
    iteration_phase_records_emitted="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' \
      "${result_dir}/iteration-phase-times.csv")"
    iteration_phase_backend_schema="$(awk -F, 'NR == 2 { gsub(/^"|"$/, "", $13); print $13 }' \
      "${result_dir}/iteration-phase-times.csv")"
  else
    iteration_phase_records_emitted="0"
    iteration_phase_backend_schema=""
  fi
  if [[ "${record_iteration_phase_times}" == true && -z "${iteration_phase_backend_schema}" ]]; then
    case "${backend}:${rccl_sync_mode:-}" in
      rccl:stream-ordered) iteration_phase_backend_schema="rccl-stream-ordered" ;;
      rccl:*) iteration_phase_backend_schema="rccl-conservative" ;;
      mpi-hip:*) iteration_phase_backend_schema="mpi-hip" ;;
    esac
  fi
  if [[ "${record_stalled_rank_times}" == true && -s "${result_dir}/stalled-rank-times.csv" ]]; then
    stalled_rank_rows_emitted="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' \
      "${result_dir}/stalled-rank-times.csv")"
    stalled_rank_iterations_recorded="$(awk -F, '
      NR > 1 {
        key = $5 ":" $6 ":" $8
        seen[key] = 1
      }
      END {
        for (key in seen) {
          ++count
        }
        print count + 0
      }
    ' "${result_dir}/stalled-rank-times.csv")"
    stalled_rank_schema="$(awk -F, 'NR == 2 { gsub(/^"|"$/, "", $21); print $21 }' \
      "${result_dir}/stalled-rank-times.csv")"
  else
    stalled_rank_rows_emitted="0"
    stalled_rank_iterations_recorded="0"
    stalled_rank_schema=""
  fi
  if [[ "${record_stalled_rank_times}" == true && -z "${stalled_rank_schema}" ]]; then
    stalled_rank_schema="${iteration_phase_backend_schema}"
  fi
  printf 'iteration_total_observed=%s\n' "${iteration_total_observed}"
  printf 'iteration_records_emitted=%s\n' "${iteration_records_emitted}"
  printf 'iteration_stall_count=%s\n' "${iteration_stall_count}"
  printf 'iteration_phase_records_emitted=%s\n' "${iteration_phase_records_emitted}"
  printf 'iteration_phase_backend_schema=%s\n' "${iteration_phase_backend_schema}"
  printf 'iteration_phase_stall_threshold_us=%s\n' "${iteration_stall_threshold_us:-0}"
  printf 'stalled_rank_iterations_recorded=%s\n' "${stalled_rank_iterations_recorded}"
  printf 'stalled_rank_rows_emitted=%s\n' "${stalled_rank_rows_emitted}"
  printf 'stalled_rank_schema=%s\n' "${stalled_rank_schema}"
  printf 'stalled_rank_stall_threshold_us=%s\n' "${iteration_stall_threshold_us:-0}"
} >>"${result_dir}/result-metadata.txt"
exit "${status}"
