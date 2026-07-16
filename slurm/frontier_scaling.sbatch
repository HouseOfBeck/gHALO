#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:00:00
#SBATCH -J ghalo-frontier-scale
#SBATCH -o batch-logs/frontier-scale-%j.out
#SBATCH -e batch-logs/frontier-scale-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "Git commit: $(git rev-parse HEAD)"
echo "Running on:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

echo
echo "MPI-HIP"
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
   --category scaling \
   --label frontier-8node-64ranks

echo
echo "RCCL conservative"
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
   --category scaling \
   --label conservative-frontier-8node-64ranks

echo
echo "RCCL stream ordered"
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
   --category scaling \
   --label stream-ordered-frontier-8node-64ranks
