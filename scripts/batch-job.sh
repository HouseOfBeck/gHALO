#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/batch-job.sh REPO_ROOT SYSTEM BACKEND ACCOUNT PARTITION NODES RANKS RANKS_PER_NODE TARGET_SECONDS VALIDATE PHASE_TIMING LABEL EXTRA_SRUN_ARGS SUBMIT_COMMAND BATCH_STDOUT BATCH_STDERR
EOF
}

[[ -n "${SLURM_JOB_ID:-}" ]] ||
  { echo "gHALO batch-job error: SLURM_JOB_ID is not set; this script must run inside a Slurm job" >&2; exit 1; }
[[ $# -eq 16 ]] || { usage >&2; exit 2; }

repo_root="$1"
system="$2"
backend="$3"
account="$4"
partition="$5"
nodes="$6"
ranks="$7"
ranks_per_node="$8"
target_seconds="$9"
validate="${10}"
phase_timing="${11}"
label="${12}"
extra_srun_args="${13}"
submit_command="${14}"
batch_stdout="${15}"
batch_stderr="${16}"

[[ "${repo_root}" = /* ]] ||
  { echo "gHALO batch-job error: REPO_ROOT must be an absolute path: ${repo_root}" >&2; exit 2; }
[[ -d "${repo_root}" ]] ||
  { echo "gHALO batch-job error: REPO_ROOT is not a directory: ${repo_root}" >&2; exit 2; }
[[ -x "${repo_root}/scripts/run.sh" ]] ||
  { echo "gHALO batch-job error: REPO_ROOT/scripts/run.sh is not executable: ${repo_root}/scripts/run.sh" >&2; exit 2; }

cd "${repo_root}"

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
  workdir: ${repo_root}
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
"${repo_root}/scripts/run.sh" "${run_args[@]}"
status=$?
set -e

echo "gHALO batch run exit status: ${status}"
exit "${status}"
