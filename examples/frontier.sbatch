#!/bin/bash
#SBATCH -J ghalo
#SBATCH -A <project>
#SBATCH -t 00:10:00
#SBATCH -N 2

set -euo pipefail

cmake -S . -B build-frontier -DGHALO_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=CC
cmake --build build-frontier

srun -n 16 ./build-frontier/ghalo --csv ghalo.csv --json ghalo.json
