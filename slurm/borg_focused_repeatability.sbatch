#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p testing
#SBATCH --reservation=jlbeck.testing
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 04:00:00
#SBATCH -J ghalo-borg-focused
#SBATCH -o batch-logs/borg-focused-%j.out
#SBATCH -e batch-logs/borg-focused-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/setenv.borg

repeats=5

echo "Git commit: $(git rev-parse HEAD)"
echo "Nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo

for repeat in $(seq -w 1 "${repeats}"); do
  echo "=================================================="
  echo "Repeat ${repeat}"
  echo "=================================================="

  scripts/run.sh \
    --system borg \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 8 \
    --max-halo 8 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode conservative \
    --category repeatability \
    --label "borg-conservative-halo8-repeat-${repeat}"

  scripts/run.sh \
    --system borg \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 128 \
    --max-halo 128 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode conservative \
    --category repeatability \
    --label "borg-conservative-halo128-repeat-${repeat}"

  scripts/run.sh \
    --system borg \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 2 \
    --max-halo 2 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode stream-ordered \
    --category repeatability \
    --label "borg-stream-halo2-repeat-${repeat}"

  scripts/run.sh \
    --system borg \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 64 \
    --max-halo 64 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode stream-ordered \
    --category repeatability \
    --label "borg-stream-halo64-repeat-${repeat}"
done
