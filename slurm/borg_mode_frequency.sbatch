#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p testing
#SBATCH --reservation=jlbeck.testing
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 03:30:00
#SBATCH -J ghalo-borg-modes
#SBATCH -o batch-logs/borg-modes-%j.out
#SBATCH -e batch-logs/borg-modes-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/setenv.borg

REPEATS=50

echo "Git commit: $(git rev-parse HEAD)"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Start: $(date -u)"
echo "Nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo

echo "============================================================"
echo "RCCL conservative, halo 128 — ${REPEATS} consecutive runs"
echo "============================================================"

for repeat in $(seq -w 1 "${REPEATS}"); do
    echo
    echo "Conservative halo 128: repeat ${repeat}/${REPEATS}"
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
        --label "borg-mode-frequency-conservative-halo128-repeat-${repeat}"
done

echo
echo "============================================================"
echo "RCCL stream-ordered, halo 64 — ${REPEATS} consecutive runs"
echo "============================================================"

for repeat in $(seq -w 1 "${REPEATS}"); do
    echo
    echo "Stream-ordered halo 64: repeat ${repeat}/${REPEATS}"
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
        --label "borg-mode-frequency-stream-halo64-repeat-${repeat}"
done

echo
echo "Completed: $(date -u)"
