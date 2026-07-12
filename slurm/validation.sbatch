#!/bin/bash
#SBATCH -A VEN004
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00
#SBATCH -J ghalo-validation
#SBATCH -o batch-logs/validation-%j.out
#SBATCH -e batch-logs/validation-%j.err

cd "$SLURM_SUBMIT_DIR"

scripts/run_validation_suite.sh
