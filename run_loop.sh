#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/gHALO"
MPI_HIP_BINARY="${REPO}/builds/frontier/mpi-hip/ghalo"
RCCL_BINARY="${REPO}/builds/frontier/rccl/ghalo"
RCCL_SMOKE="${REPO}/builds/frontier/rccl/tests/ghalo_rccl_smoke"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO}/test-logs/rocm-6.4.2-validation-${TIMESTAMP}"
RESULT_DIRS=()

ROCM_MODULE="rocm/6.4.2"
RCCL_PLUGIN_MODULE="rccl-net-plugin/1.0"

cd "${REPO}"
mkdir -p "${LOG_DIR}"

exec > >(tee "${LOG_DIR}/run-loop.log") 2>&1

echo "============================================================"
echo "gHALO ROCm 6.4.2 build and validation"
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
echo "Repository: ${REPO}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

prepare_rocm() {
    module unload rccl-net-plugin 2>/dev/null || true
    module unload rocm 2>/dev/null || true
    module load "${ROCM_MODULE}"

    unset RCCL_ROOT
    unset OLCF_OFI_NCCL_ROOT
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
}

prepare_mpi_hip() {
    echo
    echo "=== Preparing MPI-HIP environment ==="

    prepare_rocm
    export MPICH_GPU_SUPPORT_ENABLED=1

    echo "ROCM_PATH=${ROCM_PATH:-unset}"
    echo "hipcc=$(command -v hipcc)"
    echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
    module list
}

prepare_rccl() {
    echo
    echo "=== Preparing RCCL environment ==="

    prepare_rocm
    unset MPICH_GPU_SUPPORT_ENABLED

    module load "${RCCL_PLUGIN_MODULE}"
    export RCCL_ROOT="${ROCM_PATH}"

    echo "ROCM_PATH=${ROCM_PATH:-unset}"
    echo "RCCL_ROOT=${RCCL_ROOT}"
    echo "OLCF_OFI_NCCL_ROOT=${OLCF_OFI_NCCL_ROOT:-unset}"
    echo "hipcc=$(command -v hipcc)"
    module list

    test -f "${RCCL_ROOT}/include/rccl/rccl.h" || {
        echo "ERROR: RCCL header not found under ${RCCL_ROOT}" >&2
        exit 1
    }

    test -e "${RCCL_ROOT}/lib/librccl.so" || {
        echo "ERROR: RCCL library not found under ${RCCL_ROOT}" >&2
        exit 1
    }
}

verify_rocm_cache() {
    local cache="$1"
    local label="$2"

    echo
    echo "=== Verifying ${label} CMake configuration ==="

    grep -E \
      'CMAKE_HIP_COMPILER:|GHALO_RCCL_INCLUDE_DIR:|GHALO_RCCL_LIBRARY:' \
      "${cache}" || true

    local compiler
    compiler="$(
        sed -n 's|^CMAKE_HIP_COMPILER:FILEPATH=||p' "${cache}"
    )"

    case "${compiler}" in
        /opt/rocm-6.4.2/*)
            ;;
        *)
            echo "ERROR: ${label} used unexpected HIP compiler: ${compiler}" >&2
            exit 1
            ;;
    esac
}

verify_linkage() {
    local binary="$1"
    local label="$2"
    local require_rccl="$3"
    local linkage_log="${LOG_DIR}/${label}-ldd.log"

    echo
    echo "=== Verifying ${label} runtime linkage ==="

    ldd "${binary}" | tee "${linkage_log}"

    if grep -q 'not found' "${linkage_log}"; then
        echo "ERROR: unresolved library in ${label}" >&2
        exit 1
    fi

    grep -q '/opt/rocm-6.4.2/lib/libamdhip64' "${linkage_log}" || {
        echo "ERROR: ${label} did not resolve HIP runtime from ROCm 6.4.2" >&2
        exit 1
    }

    if [[ "${require_rccl}" == "yes" ]]; then
        grep -q '/opt/rocm-6.4.2/lib/librccl' "${linkage_log}" || {
            echo "ERROR: ${label} did not resolve RCCL from ROCm 6.4.2" >&2
            exit 1
        }
    fi
}

run_and_check() {
    local label="$1"
    local expected="$2"
    shift 2

    local log="${LOG_DIR}/${label}.log"

    echo
    echo "============================================================"
    echo "${label}"
    echo "============================================================"

    "$@" 2>&1 | tee "${log}"

    grep -q "${expected}" "${log}" || {
        echo "ERROR: expected success text not found for ${label}" >&2
        echo "Expected: ${expected}" >&2
        exit 1
    }
}

run_benchmark_result() {
    local label="$1"
    local expected="$2"
    shift 2

    local log="${LOG_DIR}/${label}.log"
    local result_dir

    echo
    echo "============================================================"
    echo "${label}"
    echo "============================================================"

    "$@" 2>&1 | tee "${log}"

    result_dir="$(sed -n 's/^Result directory: //p' "${log}" | tail -n 1)"
    if [[ -z "${result_dir}" || "${result_dir}" != "${REPO}/results/frontier/rocm-6.4.2/"* ]]; then
        echo "ERROR: benchmark ${label} did not create a results/frontier/rocm-6.4.2 bundle" >&2
        exit 1
    fi
    python3 -c '
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"missing or empty JSON: {path}")
with path.open(encoding="utf-8") as handle:
    data = json.load(handle)
results = data.get("results") if isinstance(data, dict) else None
if not isinstance(results, list) or not results:
    raise SystemExit(f"JSON lacks a non-empty results array: {path}")
' "${result_dir}/ghalo.json" || {
        echo "ERROR: malformed structured JSON result for ${label}: ${result_dir}/ghalo.json" >&2
        exit 1
    }
    test -f "${result_dir}/ghalo.csv" || {
        echo "ERROR: missing structured CSV result for ${label}: ${result_dir}/ghalo.csv" >&2
        exit 1
    }
    grep -q "${expected}" "${result_dir}/stdout.txt" || {
        echo "ERROR: expected success text not found for ${label}" >&2
        echo "Expected: ${expected}" >&2
        echo "Checked: ${result_dir}/stdout.txt" >&2
        exit 1
    }

    RESULT_DIRS+=("${result_dir}")
    echo "Benchmark result directory: ${result_dir}"
}

echo
echo "############################################################"
echo "# Build MPI-HIP"
echo "############################################################"

prepare_mpi_hip

GHALO_SYSTEM_NAME=frontier \
scripts/build.sh \
  --backend mpi-hip \
  --clean

test -x "${MPI_HIP_BINARY}" || {
    echo "ERROR: MPI-HIP binary missing: ${MPI_HIP_BINARY}" >&2
    exit 1
}

verify_rocm_cache \
  "${REPO}/builds/frontier/mpi-hip/CMakeCache.txt" \
  "MPI-HIP"

verify_linkage "${MPI_HIP_BINARY}" "mpi-hip" "no"

echo
echo "############################################################"
echo "# Test MPI-HIP"
echo "############################################################"

prepare_mpi_hip

run_benchmark_result \
  "mpi-hip-1node-8ranks" \
  "Backend: MPIHIPBackend" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend mpi-hip \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label validation-1node-8ranks

run_benchmark_result \
  "mpi-hip-2nodes-16ranks" \
  "Backend: MPIHIPBackend" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend mpi-hip \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label validation-2nodes-16ranks

echo
echo "############################################################"
echo "# Build RCCL"
echo "############################################################"

prepare_rccl

GHALO_SYSTEM_NAME=frontier \
scripts/build.sh \
  --backend rccl \
  --clean

test -x "${RCCL_BINARY}" || {
    echo "ERROR: RCCL binary missing: ${RCCL_BINARY}" >&2
    exit 1
}

verify_rocm_cache \
  "${REPO}/builds/frontier/rccl/CMakeCache.txt" \
  "RCCL"

verify_linkage "${RCCL_BINARY}" "rccl" "yes"

echo
echo "############################################################"
echo "# Test standalone RCCL transport"
echo "############################################################"

prepare_rccl

if [[ -x "${RCCL_SMOKE}" ]]; then
    run_and_check \
      "rccl-smoke-1node-8ranks" \
      "RCCL smoke validation PASSED" \
      srun -N 1 -n 8 --ntasks-per-node=8 \
        "${RCCL_SMOKE}" \
        --count 1024 \
        --iterations 10 \
        --validate

    run_and_check \
      "rccl-smoke-2nodes-16ranks" \
      "RCCL smoke validation PASSED" \
      srun -N 2 -n 16 --ntasks-per-node=8 \
        "${RCCL_SMOKE}" \
        --count 1024 \
        --iterations 10 \
        --validate
else
    echo "WARNING: RCCL smoke binary not found; skipping: ${RCCL_SMOKE}"
fi

echo
echo "############################################################"
echo "# Test RCCL conservative mode"
echo "############################################################"

prepare_rccl

run_benchmark_result \
  "rccl-conservative-1node-8ranks" \
  "RCCL full halo validation PASSED" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend rccl \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --rccl-sync-mode conservative \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label conservative_validation-1node-8ranks

run_benchmark_result \
  "rccl-conservative-2nodes-16ranks" \
  "RCCL full halo validation PASSED" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend rccl \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --rccl-sync-mode conservative \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label conservative_validation-2nodes-16ranks

echo
echo "############################################################"
echo "# Test RCCL stream-ordered mode"
echo "############################################################"

prepare_rccl

run_benchmark_result \
  "rccl-stream-ordered-1node-8ranks" \
  "RCCL full halo validation PASSED" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend rccl \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --rccl-sync-mode stream-ordered \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label stream-ordered_validation-1node-8ranks

run_benchmark_result \
  "rccl-stream-ordered-2nodes-16ranks" \
  "RCCL full halo validation PASSED" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend rccl \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --rccl-sync-mode stream-ordered \
    --validate \
    --target-seconds 0.1 \
    --category validation \
    --label stream-ordered_validation-2nodes-16ranks

echo
echo "############################################################"
echo "# Optional phase comparison"
echo "############################################################"

prepare_mpi_hip

run_benchmark_result \
  "mpi-hip-1node-8ranks-phase" \
  "Backend: MPIHIPBackend" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend mpi-hip \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --validate \
    --phase-timing \
    --target-seconds 3 \
    --category phase-timing \
    --label phase-1node-8ranks

prepare_rccl

run_benchmark_result \
  "rccl-stream-ordered-1node-8ranks-phase" \
  "RCCL full halo validation PASSED" \
  env GHALO_SYSTEM_NAME=frontier scripts/run.sh \
    --backend rccl \
    --nodes 1 \
    --ranks 8 \
    --ranks-per-node 8 \
    --rccl-sync-mode stream-ordered \
    --validate \
    --phase-timing \
    --target-seconds 3 \
    --category phase-timing \
    --label stream-ordered_phase-1node-8ranks

echo
echo "============================================================"
echo "All ROCm 6.4.2 GPU backend builds and tests passed"
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo
echo "Harness logs:"
echo "  ${LOG_DIR}"
echo
echo "Benchmark results:"
for result_dir in "${RESULT_DIRS[@]}"; do
    echo "  ${result_dir}"
done
echo "============================================================"
