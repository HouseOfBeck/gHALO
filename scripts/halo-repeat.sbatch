#!/bin/bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 08:00:00
#SBATCH -J ghalo-repeatability
#SBATCH -o batch-logs/frontier-repeatability-%j.out
#SBATCH -e batch-logs/frontier-repeatability-%j.err

cd "$SLURM_SUBMIT_DIR"

echo "Running on:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo

for repeat in $(seq -w 1 10)
do

    echo "=================================================="
    echo "Repeat $repeat"
    echo "=================================================="

    #
    # RCCL conservative halo 8
    #

    scripts/run.sh \
        --system frontier \
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
        --label conservative-halo8-repeat-${repeat}


    #
    # RCCL conservative halo 128
    #

    scripts/run.sh \
        --system frontier \
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
        --label conservative-halo128-repeat-${repeat}


    #
    # RCCL stream ordered halo 2
    #

    scripts/run.sh \
        --system frontier \
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
        --label stream-halo2-repeat-${repeat}


    #
    # RCCL stream ordered halo 64
    #

    scripts/run.sh \
        --system frontier \
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
        --label stream-halo64-repeat-${repeat}

done
