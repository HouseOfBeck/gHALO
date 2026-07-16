#!/usr/bin/env bash
#SBATCH -A VEN004
#SBATCH -p testing
#SBATCH --reservation=jlbeck.testing
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:15:00
#SBATCH -J ghalo-borg-rccl-version-matrix
#SBATCH -o batch-logs/borg-rccl-version-matrix-%j.out
#SBATCH -e batch-logs/borg-rccl-version-matrix-%j.err

# shellcheck source-path=SCRIPTDIR
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"

source "$ROOT/scripts/common.sh"
source "$ROOT/scripts/systems/borg.sh"

realpath_portable() {
  local path="$1"
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${path}"
}

resolve_library() {
  local root="$1"
  local name="$2"
  local found=""
  found="$(find "${root}" -name "${name}" -print -quit 2>/dev/null || true)"
  [[ -n "${found}" ]] || return 1
  realpath_portable "${found}"
}

rccl_version_code_from_headers() {
  local root="$1"
  local header
  for header in \
    "${root}/include/rccl/rccl.h" \
    "${root}/include/rccl.h" \
    "${root}/include/nccl.h"; do
    if [[ -f "${header}" ]]; then
      awk '/#define[[:space:]]+(NCCL_VERSION_CODE|RCCL_VERSION_CODE)[[:space:]]+[0-9]+/ { print $3; exit }' "${header}"
      return
    fi
  done
}

print_stack_identity() {
  local version="$1"
  local expected_code="$2"
  local observed_code="$3"
  local librccl="$4"
  local libhip="$5"
  local libnet="$6"

  cat <<EOF
gHALO Borg RCCL stack
  requested_rocm_version: ${version}
  loaded_rocm_module: ${GHALO_LOADED_ROCM_MODULE:-}
  ROCM_PATH: ${ROCM_PATH:-}
  RCCL_ROOT: ${RCCL_ROOT:-}
  expected_rccl_version_code: ${expected_code}
  observed_rccl_version_code: ${observed_code}
  librccl_realpath: ${librccl}
  libamdhip64_realpath: ${libhip}
  OLCF_OFI_NCCL_ROOT: ${OLCF_OFI_NCCL_ROOT:-}
  libnccl_net_realpath: ${libnet}
  rccl_net_plugin_module: ${GHALO_RCCL_NET_PLUGIN_MODULE:-}
  node_list: ${SLURM_JOB_NODELIST:-}
EOF
  printf 'Relevant environment:\n'
  env | sort | grep -E '^(GHALO|ROCM|RCCL|OLCF_OFI|MPICH|FI_|OFI_|HIP|HSA|ROCR|SLURM)=' || true
  printf 'Loaded modules:\n'
  module list
}

verify_ldd_stack() {
  local binary="$1"
  local version="$2"
  local output="$3"
  if command -v ldd >/dev/null 2>&1; then
    ldd "${binary}" >"${output}" 2>&1
    grep -E 'librccl|libamdhip64' "${output}" || true
    grep -E "librccl.*rocm-${version}|librccl.*rocm/${version}" "${output}" >/dev/null ||
      ghalo_die "ldd did not resolve librccl from ROCm ${version}; see ${output}"
    grep -E "libamdhip64.*rocm-${version}|libamdhip64.*rocm/${version}" "${output}" >/dev/null ||
      ghalo_die "ldd did not resolve libamdhip64 from ROCm ${version}; see ${output}"
  else
    printf 'ldd not available on this system.\n' >"${output}"
  fi
}

prepare_stack() {
  local version="$1"
  local expected_code="$2"
  local observed_code
  local librccl
  local libhip
  local libnet

  export GHALO_BORG_RCCL_ROCM_VERSION="${version}"
  export GHALO_REQUESTED_ROCM_VERSION="${version}"
  export GHALO_RCCL_VERSION_CODE_EXPECTED="${expected_code}"
  export GHALO_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0

  ghalo_borg_load_rccl_gpu
  module load miniforge3 >/dev/null 2>&1 || module load miniforge3/26.1.0-0 ||
    ghalo_die "failed to load miniforge3 on Borg"

  observed_code="$(rccl_version_code_from_headers "${RCCL_ROOT}")"
  [[ -n "${observed_code}" ]] ||
    ghalo_die "could not determine RCCL/NCCL version code from ${RCCL_ROOT}/include"
  [[ "${observed_code}" == "${expected_code}" ]] ||
    ghalo_die "ROCm ${version} expected RCCL version code ${expected_code}, observed ${observed_code}"

  librccl="$(ghalo_borg_resolve_rccl_library "${RCCL_ROOT}")" ||
    ghalo_die "could not resolve librccl.so under RCCL_ROOT=${RCCL_ROOT}"
  libhip="$(ghalo_borg_resolve_hip_library "${ROCM_PATH}")" ||
    ghalo_die "could not resolve libamdhip64.so under ROCM_PATH=${ROCM_PATH}"
  [[ -n "${OLCF_OFI_NCCL_ROOT:-}" ]] ||
    ghalo_die "OLCF_OFI_NCCL_ROOT is not set after loading rccl-net-plugin/1.0"
  libnet="$(resolve_library "${OLCF_OFI_NCCL_ROOT}" 'libnccl-net.so*')" ||
    ghalo_die "could not resolve libnccl-net.so under OLCF_OFI_NCCL_ROOT=${OLCF_OFI_NCCL_ROOT}"

  export GHALO_RCCL_VERSION_CODE_OBSERVED="${observed_code}"
  export GHALO_RESOLVED_RCCL_LIBRARY_REALPATH
  GHALO_RESOLVED_RCCL_LIBRARY_REALPATH="$(realpath_portable "${librccl}")"
  export GHALO_RESOLVED_HIP_LIBRARY_REALPATH
  GHALO_RESOLVED_HIP_LIBRARY_REALPATH="$(realpath_portable "${libhip}")"
  export GHALO_RESOLVED_NCCL_NET_LIBRARY_REALPATH="${libnet}"

  print_stack_identity "${version}" "${expected_code}" "${observed_code}" \
    "${GHALO_RESOLVED_RCCL_LIBRARY_REALPATH}" \
    "${GHALO_RESOLVED_HIP_LIBRARY_REALPATH}" \
    "${GHALO_RESOLVED_NCCL_NET_LIBRARY_REALPATH}"
}

run_version() {
  local version="$1"
  local expected_code="$2"
  local label="borg-rccl-version-rocm-${version}"
  local build_dir="${ROOT}/builds/borg/rccl-rocm-${version}"
  local binary="${build_dir}/ghalo"
  local log_dir="test-logs/borg-rccl-version-matrix-${SLURM_JOB_ID:-manual}"

  mkdir -p "${log_dir}"
  printf '\n== ROCm %s / RCCL version code %s ==\n' "${version}" "${expected_code}"
  prepare_stack "${version}" "${expected_code}"

  binary="$ROOT/builds/borg/rccl-rocm-${version}/ghalo"

  if [[ ! -x "${binary}" ]]; then
      echo "UNUSABLE STACK: ROCm ${version} build did not produce ${binary}" >&2
      return 0
  fi


  if ! scripts/build.sh \
       --system borg \
       --backend rccl \
       --build-dir "${build_dir}" \
       --clean; then
    echo "UNUSABLE STACK: ROCm ${version} build failed" >&2
      return 0
  fi

   if [[ ! -x "${binary}" ]]; then
      echo "UNUSABLE STACK: ROCm ${version} build did not produce ${binary}" >&2
      return 0
   fi

   echo "Runtime dependencies for ${binary}:"
   ldd "${binary}"

   if ldd "${binary}" | grep -q 'not found'; then
     echo "UNUSABLE STACK: ROCm ${version} has unresolved runtime dependencies" >&2
     ldd "${binary}" >&2
     return 0
   fi

   verify_ldd_stack \
    "${binary}" \
    "${version}" \
    "${log_dir}/ldd-rocm-${version}.txt"

  verify_ldd_stack "${binary}" "${version}" "${log_dir}/ldd-rocm-${version}.txt"

  GHALO_SYSTEM_NAME=borg \
  GHALO_BORG_RCCL_ROCM_VERSION="${version}" \
  GHALO_BUILD_DIR_OVERRIDE="${build_dir}" \
  GHALO_REQUESTED_ROCM_VERSION="${version}" \
  GHALO_RCCL_VERSION_CODE_EXPECTED="${expected_code}" \
  GHALO_RCCL_VERSION_CODE_OBSERVED="${GHALO_RCCL_VERSION_CODE_OBSERVED}" \
  GHALO_RCCL_NET_PLUGIN_MODULE="${GHALO_RCCL_NET_PLUGIN_MODULE}" \
  GHALO_RESOLVED_RCCL_LIBRARY_REALPATH="${GHALO_RESOLVED_RCCL_LIBRARY_REALPATH}" \
  GHALO_RESOLVED_HIP_LIBRARY_REALPATH="${GHALO_RESOLVED_HIP_LIBRARY_REALPATH}" \
  GHALO_RESOLVED_NCCL_NET_LIBRARY_REALPATH="${GHALO_RESOLVED_NCCL_NET_LIBRARY_REALPATH}" \
  scripts/run.sh \
    --system borg \
    --backend rccl \
    --nodes 8 \
    --ranks 64 \
    --ranks-per-node 8 \
    --target-seconds 0.1 \
    --min-halo 128 \
    --max-halo 128 \
    --samples-per-halo 200 \
    --validate \
    --phase-timing \
    --record-iteration-times \
    --record-iteration-phase-times \
    --record-stalled-rank-times \
    --iteration-stall-threshold-us 1000 \
    --rccl-sync-mode conservative \
    --category repeatability \
    --label "${label}"
}

utc_start="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"

cat <<EOF
gHALO Borg RCCL version matrix
  utc_start: ${utc_start}
  git_commit: ${git_commit}
  slurm_job_id: ${SLURM_JOB_ID:-}
  slurm_job_name: ${SLURM_JOB_NAME:-}
  node_list: ${SLURM_JOB_NODELIST:-}
EOF

if [[ -f scripts/setenv.borg ]]; then
  # shellcheck source=../scripts/setenv.borg
  source scripts/setenv.borg
fi

run_version 6.4.2 22203
run_version 7.0.2 22606
run_version 7.2.0 22707

utc_end="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cat <<EOF
gHALO Borg RCCL version matrix complete
  utc_start: ${utc_start}
  utc_end: ${utc_end}
  slurm_job_id: ${SLURM_JOB_ID:-}
EOF
