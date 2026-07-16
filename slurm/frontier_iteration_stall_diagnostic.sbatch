#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:15:00
#SBATCH -J ghalo-frontier-stalls
#SBATCH -o batch-logs/frontier-iteration-stalls-%j.out
#SBATCH -e batch-logs/frontier-iteration-stalls-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p batch-logs

echo "gHALO Frontier iteration-stall diagnostic"
echo "=========================================="
echo "Start time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit:   $(git rev-parse HEAD)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo

echo "Loaded modules:"
module list 2>&1
echo

echo "============================================================"
echo "Case 1: RCCL conservative, halo 128"
echo "============================================================"

scripts/run.sh \
  --system frontier \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 128 \
  --max-halo 128 \
  --halo-multiplier 2 \
  --samples-per-halo 100 \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --validate \
  --phase-timing \
  --rccl-sync-mode conservative \
  --category repeatability \
  --label frontier-iteration-stall-rccl-conservative-halo128

echo
echo "============================================================"
echo "Case 2: RCCL stream-ordered, halo 64"
echo "============================================================"

scripts/run.sh \
  --system frontier \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 64 \
  --max-halo 64 \
  --halo-multiplier 2 \
  --samples-per-halo 100 \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --validate \
  --phase-timing \
  --rccl-sync-mode stream-ordered \
  --category repeatability \
  --label frontier-iteration-stall-rccl-stream-halo64

echo
echo "============================================================"
echo "Case 3: MPI-HIP control, halo 128"
echo "============================================================"

scripts/run.sh \
  --system frontier \
  --backend mpi-hip \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 128 \
  --max-halo 128 \
  --halo-multiplier 2 \
  --samples-per-halo 100 \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --validate \
  --phase-timing \
  --category repeatability \
  --label frontier-iteration-stall-mpi-hip-halo128

echo
echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
