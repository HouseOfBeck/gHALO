#!/usr/bin/env bash

# shellcheck source-path=SCRIPTDIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/submit.sh --backend mpi|mpi-hip|rccl --nodes N --ranks N --ranks-per-node N --time HH:MM:SS [options]

Options:
  --system NAME                 System configuration name. GHALO_SYSTEM_NAME wins if set.
  --backend mpi|mpi-hip|rccl    Backend to run.
  --account ACCOUNT             Slurm account.
  --partition PARTITION         Slurm partition.
  --nodes N                     Number of nodes.
  --ranks N                     Total MPI ranks.
  --ranks-per-node N            MPI ranks per node.
  --time HH:MM:SS               Slurm wall-clock limit.
  --target-seconds SECONDS      gHALO target seconds per halo size. Default: 3.
  --min-halo N                  Minimum halo length. Default: 2.
  --max-halo N                  Maximum halo length. Default: 1024.
  --halo-multiplier N           Halo length multiplier. Default: 2.
  --samples-per-halo N          Independent timed samples per halo. Default: 1.
  --record-iteration-times      Write per-iteration max-rank timing diagnostics.
  --record-iteration-phase-times
                                Write per-phase timing for threshold-matched iterations.
  --iteration-stall-threshold-us VALUE
                                Emit iteration timing records at or above VALUE us.
                                Default/zero emits every measured iteration.
  --validate                    Pass --validate to gHALO.
  --phase-timing                Pass --phase-timing to gHALO.
  --rccl-stage-b                Run RCCL north/south Stage B validation only.
  --rccl-sync-mode MODE         Pass conservative or stream-ordered to RCCL.
  --category NAME               Result category: validation, scaling, repeatability, or phase-timing.
  --label TEXT                  Optional result label.
  --job-name NAME               Slurm job name.
  --constraint CONSTRAINT       Slurm constraint.
  --reservation RESERVATION     Slurm reservation.
  --qos QOS                     Slurm QOS.
  --exclusive                   Request exclusive nodes.
  --dependency EXPR             Slurm dependency expression.
  --extra-sbatch-args ARGS      Extra sbatch arguments, split on whitespace.
  --extra-srun-args ARGS        Extra srun arguments forwarded to scripts/run.sh.
  --confirm-large-run           Required above the large-run threshold.
  --dry-run                     Print the sbatch command without submitting.
  -h, --help                    Show this help.
EOF
}

shell_join() {
  printf '%q ' "$@"
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ && "$1" -ge 1 ]]
}

is_positive_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
    awk -v value="$1" 'BEGIN { exit !(value > 0.0) }'
}

root="$(ghalo_repo_root)"
original_args=("$@")
original_submit_command="$(shell_join "$0" "${original_args[@]}")"
requested_system=""
backend=""
account=""
partition=""
nodes=""
ranks=""
ranks_per_node=""
wall_time=""
target_seconds="3"
min_halo="2"
max_halo="1024"
halo_multiplier="2"
samples_per_halo="1"
record_iteration_times=0
record_iteration_phase_times=0
iteration_stall_threshold_us=""
validate=0
phase_timing=0
rccl_stage_b=0
rccl_sync_mode=""
category=""
label="run"
job_name=""
constraint=""
reservation=""
qos=""
exclusive=0
dependency=""
extra_sbatch_args=""
extra_srun_args=""
dry_run=0
confirm_large_run=0
large_warning_nodes="${GHALO_LARGE_RUN_WARNING_NODES:-128}"
large_confirm_nodes="${GHALO_LARGE_RUN_CONFIRM_NODES:-512}"

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
    --account)
      [[ $# -ge 2 ]] || ghalo_die "--account requires a value"
      account="$2"
      shift 2
      ;;
    --partition)
      [[ $# -ge 2 ]] || ghalo_die "--partition requires a value"
      partition="$2"
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
    --time)
      [[ $# -ge 2 ]] || ghalo_die "--time requires a value"
      wall_time="$2"
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
      record_iteration_times=1
      shift
      ;;
    --record-iteration-phase-times)
      record_iteration_phase_times=1
      shift
      ;;
    --iteration-stall-threshold-us)
      [[ $# -ge 2 ]] || ghalo_die "--iteration-stall-threshold-us requires a value"
      iteration_stall_threshold_us="$2"
      shift 2
      ;;
    --validate)
      validate=1
      shift
      ;;
    --phase-timing)
      phase_timing=1
      shift
      ;;
    --rccl-stage-b)
      rccl_stage_b=1
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
    --job-name)
      [[ $# -ge 2 ]] || ghalo_die "--job-name requires a value"
      job_name="$2"
      shift 2
      ;;
    --constraint)
      [[ $# -ge 2 ]] || ghalo_die "--constraint requires a value"
      constraint="$2"
      shift 2
      ;;
    --reservation)
      [[ $# -ge 2 ]] || ghalo_die "--reservation requires a value"
      reservation="$2"
      shift 2
      ;;
    --qos)
      [[ $# -ge 2 ]] || ghalo_die "--qos requires a value"
      qos="$2"
      shift 2
      ;;
    --exclusive)
      exclusive=1
      shift
      ;;
    --dependency)
      [[ $# -ge 2 ]] || ghalo_die "--dependency requires a value"
      dependency="$2"
      shift 2
      ;;
    --extra-sbatch-args)
      [[ $# -ge 2 ]] || ghalo_die "--extra-sbatch-args requires a value"
      extra_sbatch_args="$2"
      shift 2
      ;;
    --extra-srun-args)
      [[ $# -ge 2 ]] || ghalo_die "--extra-srun-args requires a value"
      extra_srun_args="$2"
      shift 2
      ;;
    --confirm-large-run)
      confirm_large_run=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
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

system="$(ghalo_resolve_system "${requested_system}")"
ghalo_validate_system_name "${system}"
ghalo_load_system_config "${root}" "${system}"

if [[ -z "${account}" ]]; then
  account="$(ghalo_system_default_account || true)"
fi
if [[ -z "${partition}" ]]; then
  partition="$(ghalo_system_default_partition || true)"
fi
if [[ -z "${wall_time}" ]]; then
  wall_time="$(ghalo_system_default_batch_time || true)"
fi

[[ -n "${account}" ]] || ghalo_die "--account is required for system '${system}'"
[[ -n "${partition}" ]] || ghalo_die "--partition is required for system '${system}'"
[[ -n "${nodes}" ]] || ghalo_die "--nodes is required"
[[ -n "${ranks}" ]] || ghalo_die "--ranks is required"
[[ -n "${ranks_per_node}" ]] || ghalo_die "--ranks-per-node is required"
[[ -n "${wall_time}" ]] || ghalo_die "--time is required"

is_positive_integer "${nodes}" || ghalo_die "--nodes must be a positive integer"
is_positive_integer "${ranks}" || ghalo_die "--ranks must be a positive integer"
is_positive_integer "${ranks_per_node}" ||
  ghalo_die "--ranks-per-node must be a positive integer"
is_positive_integer "${large_warning_nodes}" ||
  ghalo_die "GHALO_LARGE_RUN_WARNING_NODES must be a positive integer"
is_positive_integer "${large_confirm_nodes}" ||
  ghalo_die "GHALO_LARGE_RUN_CONFIRM_NODES must be a positive integer"
is_positive_number "${target_seconds}" ||
  ghalo_die "--target-seconds must be positive"
is_positive_integer "${min_halo}" || ghalo_die "--min-halo must be a positive integer"
is_positive_integer "${max_halo}" || ghalo_die "--max-halo must be a positive integer"
is_positive_integer "${halo_multiplier}" ||
  ghalo_die "--halo-multiplier must be a positive integer"
is_positive_integer "${samples_per_halo}" ||
  ghalo_die "--samples-per-halo must be a positive integer"
if [[ -n "${iteration_stall_threshold_us}" ]]; then
  [[ "${iteration_stall_threshold_us}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    ghalo_die "--iteration-stall-threshold-us must be nonnegative"
fi
if [[ "${record_iteration_phase_times}" -eq 1 &&
      "${record_iteration_times}" -ne 1 ]]; then
  ghalo_die "--record-iteration-phase-times requires --record-iteration-times"
fi
[[ "${halo_multiplier}" -gt 1 ]] ||
  ghalo_die "--halo-multiplier must be greater than 1"
[[ "${max_halo}" -ge "${min_halo}" ]] ||
  ghalo_die "--max-halo must be greater than or equal to --min-halo"
[[ "${wall_time}" =~ ^[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] ||
  ghalo_die "--time must use HH:MM:SS"

expected_ranks=$((nodes * ranks_per_node))
[[ "${ranks}" -le "${expected_ranks}" ]] ||
  ghalo_die "ranks (${ranks}) cannot exceed nodes * ranks-per-node (${expected_ranks})"
[[ "${ranks}" -eq "${expected_ranks}" ]] ||
  ghalo_die "gHALO batch submissions require ranks == nodes * ranks-per-node (${expected_ranks}); adjust the request or use scripts/run.sh directly for unusual layouts"

if [[ "${nodes}" -gt "${large_warning_nodes}" ]]; then
  echo "Warning: submitting a large gHALO run with ${nodes} nodes" >&2
fi
if [[ "${nodes}" -gt "${large_confirm_nodes}" && "${confirm_large_run}" -ne 1 ]]; then
  ghalo_die "runs above ${large_confirm_nodes} nodes require --confirm-large-run"
fi

command -v sbatch >/dev/null 2>&1 || [[ "${dry_run}" -eq 1 ]] ||
  ghalo_die "sbatch is not available"

if [[ -z "${job_name}" ]]; then
  safe_label="$(ghalo_sanitize_label "${label}")"
  safe_label="$(ghalo_strip_backend_label_prefixes "${backend}" "${safe_label}")"
  job_name="ghalo-${system}-${backend}-${safe_label}"
fi

log_dir="${root}/batch-logs/${system}"
mkdir -p "${log_dir}"
stdout_path="${log_dir}/%x-%j.out"
stderr_path="${log_dir}/%x-%j.err"

sbatch_command=(
  sbatch
  --parsable
  --job-name "${job_name}"
  --account "${account}"
  --partition "${partition}"
  --nodes "${nodes}"
  --ntasks "${ranks}"
  --ntasks-per-node "${ranks_per_node}"
  --time "${wall_time}"
  --output "${stdout_path}"
  --error "${stderr_path}"
)

if [[ -n "${constraint}" ]]; then
  sbatch_command+=(--constraint "${constraint}")
fi
if [[ -n "${reservation}" ]]; then
  sbatch_command+=(--reservation "${reservation}")
fi
if [[ -n "${qos}" ]]; then
  sbatch_command+=(--qos "${qos}")
fi
if [[ "${exclusive}" -eq 1 ]]; then
  sbatch_command+=(--exclusive)
fi
if [[ -n "${dependency}" ]]; then
  sbatch_command+=(--dependency "${dependency}")
fi
if [[ -n "${extra_sbatch_args}" ]]; then
  extra_args=()
  read -r -a extra_args <<<"${extra_sbatch_args}"
  sbatch_command+=("${extra_args[@]}")
fi

sbatch_command+=(
  "${root}/scripts/batch-job.sh"
  "${root}"
  "${system}"
  "${backend}"
  "${account}"
  "${partition}"
  "${nodes}"
  "${ranks}"
  "${ranks_per_node}"
  "${target_seconds}"
  "${min_halo}"
  "${max_halo}"
  "${halo_multiplier}"
  "${samples_per_halo}"
  "${record_iteration_times}"
  "${record_iteration_phase_times}"
  "${iteration_stall_threshold_us}"
  "${validate}"
  "${phase_timing}"
  "${rccl_stage_b}"
  "${rccl_sync_mode}"
  "${category}"
  "${label}"
  "${extra_srun_args}"
  "${original_submit_command}"
  "${stdout_path}"
  "${stderr_path}"
)

echo "gHALO batch submission request:"
echo "  system: ${system}"
echo "  backend: ${backend}"
echo "  nodes: ${nodes}"
echo "  ranks: ${ranks}"
echo "  ranks_per_node: ${ranks_per_node}"
echo "  estimated_benchmark_duration: approximately ${target_seconds}s per halo size plus startup and scheduler overhead"
echo "  wall_time: ${wall_time}"
echo "  partition: ${partition}"
echo "  account: ${account}"
echo
echo "sbatch command:"
shell_join "${sbatch_command[@]}"
printf '\n'

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Dry run: not submitting."
  exit 0
fi

set +e
job_id="$("${sbatch_command[@]}")"
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
  echo "Submitted Slurm job: ${job_id}"
fi
exit "${status}"
