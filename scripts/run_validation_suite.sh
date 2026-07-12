#!/usr/bin/env bash

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM="frontier"
ROCM_MODULE="rocm/6.4.2"
RCCL_PLUGIN_MODULE="rccl-net-plugin/1.0"
TARGET_SECONDS="0.1"

cd "${REPO}"

if [[ ! -x scripts/run.sh ]]; then
    echo "ERROR: ${REPO}/scripts/run.sh is missing or not executable." >&2
    exit 1
fi

allocated_nodes="${SLURM_JOB_NUM_NODES:-0}"

if [[ "${allocated_nodes}" -lt 2 ]]; then
    echo "ERROR: This script requires an active Slurm reservation of at least 2 nodes." >&2
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}" >&2
    echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset}" >&2
    exit 1
fi

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

prepare_rocm() {
    module unload rccl-net-plugin 2>/dev/null || true
    module unload rocm 2>/dev/null || true
    module load "${ROCM_MODULE}"
}

prepare_mpi_hip() {
    prepare_rocm
    export MPICH_GPU_SUPPORT_ENABLED=1

    echo "ROCM_PATH=${ROCM_PATH:-unset}"
    echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
}

prepare_rccl() {
    prepare_rocm
    module load "${RCCL_PLUGIN_MODULE}"

    export RCCL_ROOT="${ROCM_PATH}"

    echo "ROCM_PATH=${ROCM_PATH:-unset}"
    echo "RCCL_ROOT=${RCCL_ROOT:-unset}"
    echo "OLCF_OFI_NCCL_ROOT=${OLCF_OFI_NCCL_ROOT:-unset}"
}

run_case() {
    local description="$1"
    shift

    section "${description}"
    env GHALO_SYSTEM_NAME="${SYSTEM}" scripts/run.sh "$@"
}

echo "============================================================"
echo "gHALO validation suite"
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Repository: ${REPO}"
echo "Slurm job: ${SLURM_JOB_ID:-unset}"
echo "Allocated nodes: ${allocated_nodes}"
echo "ROCm module: ${ROCM_MODULE}"
echo "============================================================"

section "Preparing MPI-HIP environment"
prepare_mpi_hip
module list

run_case "MPI-HIP: 1 node, 8 ranks" \
    --backend mpi-hip \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label validation-1node-8ranks

run_case "MPI-HIP: 2 nodes, 16 ranks" \
    --backend mpi-hip \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label validation-2nodes-16ranks

section "Preparing RCCL environment"
prepare_rccl
module list

run_case "RCCL conservative: 1 node, 8 ranks" \
    --backend rccl \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --rccl-sync-mode conservative \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label conservative-validation-1node-8ranks

run_case "RCCL conservative: 2 nodes, 16 ranks" \
    --backend rccl \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --rccl-sync-mode conservative \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label conservative-validation-2nodes-16ranks

run_case "RCCL stream-ordered: 1 node, 8 ranks" \
    --backend rccl \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --rccl-sync-mode stream-ordered \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label stream-ordered-validation-1node-8ranks

run_case "RCCL stream-ordered: 2 nodes, 16 ranks" \
    --backend rccl \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --rccl-sync-mode stream-ordered \
    --validate \
    --target-seconds "${TARGET_SECONDS}" \
    --label stream-ordered-validation-2nodes-16ranks

echo
echo "============================================================"
echo "All validation cases completed successfully"
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Results were written by scripts/run.sh under the results tree."
echo "============================================================"
