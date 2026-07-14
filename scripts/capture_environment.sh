#!/bin/bash
set -euo pipefail

if [[ $# -ge 1 ]]; then
    OUTDIR="$1"
else
    OUTDIR="environment-capture-$(date -u +%Y%m%dT%H%M%SZ)"
fi


mkdir -p "$OUTDIR"

echo "gHALO environment capture" > "$OUTDIR/environment.txt"
echo "================================" >> "$OUTDIR/environment.txt"

echo >> "$OUTDIR/environment.txt"
echo "Timestamp:" >> "$OUTDIR/environment.txt"
date -u >> "$OUTDIR/environment.txt"

echo >> "$OUTDIR/environment.txt"
echo "Git:" >> "$OUTDIR/environment.txt"
git rev-parse HEAD >> "$OUTDIR/environment.txt"
git describe --tags --always >> "$OUTDIR/environment.txt"

echo >> "$OUTDIR/environment.txt"
echo "SLURM:" >> "$OUTDIR/environment.txt"
echo "Job ID: ${SLURM_JOB_ID:-none}" >> "$OUTDIR/environment.txt"
echo "Nodes:" >> "$OUTDIR/environment.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" >> "$OUTDIR/environment.txt" 2>/dev/null || true
lscpu

echo >> "$OUTDIR/environment.txt"
echo "Modules:" >> "$OUTDIR/environment.txt"
module list 2>&1 >> "$OUTDIR/environment.txt"

echo >> "$OUTDIR/environment.txt"
echo "Environment variables:" >> "$OUTDIR/environment.txt"
env | grep -E '^(MPICH|ROCR|HIP|RCCL|SLURM)' | sort >> "$OUTDIR/environment.txt"

echo >> "$OUTDIR/environment.txt"
echo "HIP compiler:" >> "$OUTDIR/environment.txt"
hipcc --version >> "$OUTDIR/environment.txt"

echo
echo "MPI:"
which mpirun || true
mpirun --version || true

echo
echo "RCCL:"
echo "RCCL_ROOT=${RCCL_ROOT:-}"
if [[ -n "${RCCL_ROOT:-}" && -f "${RCCL_ROOT}/lib/librccl.so" ]]; then
	    strings "${RCCL_ROOT}/lib/librccl.so" | grep -i version | head || true
fi

echo
echo "GPU topology:"
rocm-smi --showtopo || true

echo
echo "CPU topology:"
lscpu || true

echo >> "$OUTDIR/environment.txt"
echo "ROCm devices:" >> "$OUTDIR/environment.txt"
rocm-smi --showproductname >> "$OUTDIR/environment.txt" 2>&1

echo "RCCL:"
/opt/rocm/bin/rccl-test --version 2>/dev/null || true
rpm -qa | grep rccl 2>/dev/null || true

echo
echo "GPU topology:"
rocm-smi --showtopo

echo
echo "MPI:"
which mpirun
mpirun --version

echo
echo "MPICH environment:"
env | grep ^MPICH

echo >> "$OUTDIR/environment.txt"
echo "Node hardware:" >> "$OUTDIR/environment.txt"
scontrol show node "$(hostname)" | \
	    grep -E "NodeName|Gres|CfgTRES|RealMemory|Sockets" \
	        >> "$OUTDIR/environment.txt"

echo "Environment captured in $OUTDIR/environment.txt"

