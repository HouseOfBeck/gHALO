#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p testing
#SBATCH --reservation=jlbeck.testing
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00
#SBATCH -J ghalo-borg-single-process
#SBATCH -o batch-logs/borg-single-process-%j.out
#SBATCH -e batch-logs/borg-single-process-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/setenv.borg

echo "gHALO Borg single-process mode persistence experiment"
echo "====================================================="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit: $(git rev-parse HEAD)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo

echo "RCCL conservative, halo 128, 50 samples in one executable"
echo "---------------------------------------------------------"

scripts/run.sh \
  --system borg \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 128 \
  --max-halo 128 \
  --halo-multiplier 2 \
  --samples-per-halo 50 \
  --validate \
  --phase-timing \
  --rccl-sync-mode conservative \
  --category repeatability \
  --label borg-single-process-conservative-halo128

echo
echo "RCCL stream-ordered, halo 64, 50 samples in one executable"
echo "----------------------------------------------------------"

scripts/run.sh \
  --system borg \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 64 \
  --max-halo 64 \
  --halo-multiplier 2 \
  --samples-per-halo 50 \
  --validate \
  --phase-timing \
  --rccl-sync-mode stream-ordered \
  --category repeatability \
  --label borg-single-process-stream-halo64

echo
echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
