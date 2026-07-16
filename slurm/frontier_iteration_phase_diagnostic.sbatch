#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:10:00
#SBATCH -J ghalo-frontier-iteration-phase
#SBATCH -o batch-logs/frontier-iteration-phase-%j.out
#SBATCH -e batch-logs/frontier-iteration-phase-%j.err

# shellcheck source-path=SCRIPTDIR

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}" || exit

utc_start="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
log_dir="test-logs/frontier-iteration-phase-${SLURM_JOB_ID:-manual}"
mkdir -p "${log_dir}"

cat <<EOF
gHALO Frontier iteration phase diagnostic
  utc_start: ${utc_start}
  git_commit: ${git_commit}
  slurm_job_id: ${SLURM_JOB_ID:-}
  slurm_job_name: ${SLURM_JOB_NAME:-}
  node_list: ${SLURM_JOB_NODELIST:-}
  log_dir: ${log_dir}
EOF

if [[ -x scripts/capture_environment.sh ]]; then
  scripts/capture_environment.sh "${log_dir}" || \
    printf 'WARNING: environment capture failed\n' >&2
fi

run_case() {
  local backend="$1"
  local sync_mode="$2"
  local halo="$3"
  local label="$4"

  printf '\n== %s ==\n' "${label}"
  local command=(
    env GHALO_SYSTEM_NAME=frontier
    scripts/run.sh
    --backend "${backend}"
    --nodes 8
    --ranks 64
    --ranks-per-node 8
    --min-halo "${halo}"
    --max-halo "${halo}"
    --samples-per-halo 100
    --target-seconds 0.1
    --validate
    --phase-timing
    --record-iteration-times
    --record-iteration-phase-times
    --iteration-stall-threshold-us 1000
    --category repeatability
    --label "${label}"
  )
  if [[ -n "${sync_mode}" ]]; then
    command+=(--rccl-sync-mode "${sync_mode}")
  fi
  "${command[@]}"
}

run_case rccl conservative 128 frontier-iteration-phase-rccl-conservative-halo128
run_case rccl stream-ordered 64 frontier-iteration-phase-rccl-stream-halo64
run_case mpi-hip "" 128 frontier-iteration-phase-mpi-hip-halo128

utc_end="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cat <<EOF
gHALO Frontier iteration phase diagnostic complete
  utc_start: ${utc_start}
  utc_end: ${utc_end}
  slurm_job_id: ${SLURM_JOB_ID:-}
EOF
