#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:30:00
#SBATCH -J ghalo-frontier-repeat
#SBATCH -o batch-logs/frontier-repeat-%j.out
#SBATCH -e batch-logs/frontier-repeat-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

repeats=5

echo "Git commit: $(git rev-parse HEAD)"
echo "Nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

for repeat in $(seq 1 "${repeats}"); do
  printf '\n============================================================\n'
  printf 'Repeat %d of %d: MPI-HIP\n' "${repeat}" "${repeats}"
  printf '============================================================\n'

  scripts/run.sh \
    --system frontier \
    --backend mpi-hip \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 2 \
    --max-halo 4096 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --category repeatability \
    --label "frontier-8node-mpi-hip-repeat-${repeat}"
done

for repeat in $(seq 1 "${repeats}"); do
  printf '\n============================================================\n'
  printf 'Repeat %d of %d: RCCL conservative\n' "${repeat}" "${repeats}"
  printf '============================================================\n'

  scripts/run.sh \
    --system frontier \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 2 \
    --max-halo 4096 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode conservative \
    --category repeatability \
    --label "conservative-frontier-8node-repeat-${repeat}"
done

for repeat in $(seq 1 "${repeats}"); do
  printf '\n============================================================\n'
  printf 'Repeat %d of %d: RCCL stream-ordered\n' "${repeat}" "${repeats}"
  printf '============================================================\n'

  scripts/run.sh \
    --system frontier \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 1.0 \
    --min-halo 2 \
    --max-halo 4096 \
    --halo-multiplier 2 \
    --validate \
    --phase-timing \
    --rccl-sync-mode stream-ordered \
    --category repeatability \
    --label "stream-ordered-frontier-8node-repeat-${repeat}"
done
