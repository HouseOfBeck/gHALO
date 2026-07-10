#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/batch-job.sh SYSTEM BACKEND ACCOUNT PARTITION NODES RANKS RANKS_PER_NODE TARGET_SECONDS VALIDATE PHASE_TIMING LABEL EXTRA_SRUN_ARGS SUBMIT_COMMAND BATCH_STDOUT BATCH_STDERR
EOF
}

[[ -n "${SLURM_JOB_ID:-}" ]] ||
  { echo "gHALO batch-job error: SLURM_JOB_ID is not set; this script must run inside a Slurm job" >&2; exit 1; }
[[ $# -eq 15 ]] || { usage >&2; exit 2; }

system="$1"
backend="$2"
account="$3"
partition="$4"
nodes="$5"
ranks="$6"
ranks_per_node="$7"
target_seconds="$8"
validate="$9"
phase_timing="${10}"
label="${11}"
extra_srun_args="${12}"
submit_command="${13}"
batch_stdout="${14}"
batch_stderr="${15}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

expand_slurm_log_path() {
  local path="$1"
  path="${path//%x/${SLURM_JOB_NAME:-}}"
  path="${path//%j/${SLURM_JOB_ID:-}}"
  printf '%s\n' "${path}"
}

batch_stdout_resolved="$(expand_slurm_log_path "${batch_stdout}")"
batch_stderr_resolved="$(expand_slurm_log_path "${batch_stderr}")"

export GHALO_SYSTEM_NAME="${system}"
export GHALO_SUBMIT_COMMAND="${submit_command}"
export GHALO_BATCH_STDOUT="${batch_stdout_resolved}"
export GHALO_BATCH_STDERR="${batch_stderr_resolved}"
export GHALO_SUBMISSION_SYSTEM="${system}"
export GHALO_SUBMISSION_ACCOUNT="${account}"
export GHALO_SUBMISSION_PARTITION="${partition}"

git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
utc_date="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

cat <<EOF
gHALO batch job
  utc_date: ${utc_date}
  slurm_job_id: ${SLURM_JOB_ID}
  slurm_job_name: ${SLURM_JOB_NAME:-}
  system: ${system}
  backend: ${backend}
  account: ${account}
  partition: ${partition}
  nodes: ${nodes}
  ranks: ${ranks}
  ranks_per_node: ${ranks_per_node}
  allocated_nodes: ${SLURM_JOB_NODELIST:-}
  git_commit: ${git_commit}
  workdir: ${ROOT}
EOF

run_args=(
  --system "${system}"
  --backend "${backend}"
  --nodes "${nodes}"
  --ranks "${ranks}"
  --ranks-per-node "${ranks_per_node}"
  --target-seconds "${target_seconds}"
  --label "${label}"
)

if [[ "${validate}" == "1" ]]; then
  run_args+=(--validate)
fi
if [[ "${phase_timing}" == "1" ]]; then
  run_args+=(--phase-timing)
fi
if [[ -n "${extra_srun_args}" ]]; then
  run_args+=(--extra-srun-args "${extra_srun_args}")
fi

set +e
"${ROOT}/scripts/run.sh" "${run_args[@]}"
status=$?
set -e

echo "gHALO batch run exit status: ${status}"
exit "${status}"
