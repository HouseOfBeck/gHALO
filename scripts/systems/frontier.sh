#!/usr/bin/env bash

ghalo_frontier_have_modules() {
  type module >/dev/null 2>&1
}

ghalo_frontier_load_common() {
  if ghalo_frontier_have_modules; then
    module load PrgEnv-cray >/dev/null 2>&1 || true
  fi
  command -v CC >/dev/null 2>&1 ||
    ghalo_die "Frontier builds require the Cray C++ wrapper 'CC'"
}

ghalo_frontier_load_gpu() {
  if ghalo_frontier_have_modules; then
    module load craype-accel-amd-gfx90a
  fi
}

ghalo_frontier_require_modules() {
  ghalo_frontier_have_modules ||
    ghalo_die "Frontier GPU backends require environment modules"
}

ghalo_frontier_unload_rocm_and_rccl() {
  ghalo_frontier_require_modules
  module unload rccl-net-plugin >/dev/null 2>&1 || true
  module unload rccl-net-plugin/1.0 >/dev/null 2>&1 || true
  module unload rocm >/dev/null 2>&1 || true
  module unload rocm/6.2.4 >/dev/null 2>&1 || true
  module unload rocm/6.4.2 >/dev/null 2>&1 || true
}

ghalo_frontier_path_matches_rocm_version() {
  local path="$1"
  local version="$2"
  [[ "${path}" == *"rocm-${version}"* || "${path}" == *"rocm/${version}"* ]]
}

ghalo_frontier_rccl_header_exists() {
  local root="$1"
  [[ -f "${root}/include/rccl/rccl.h" ]]
}

ghalo_frontier_resolve_rccl_library() {
  local root="$1"
  local candidate
  for candidate in \
    "${root}/lib/librccl.so" \
    "${root}/lib64/librccl.so" \
    "${root}/lib/librccl.so.1" \
    "${root}/lib64/librccl.so.1"; do
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ghalo_frontier_resolve_hip_library() {
  local root="$1"
  local candidate
  for candidate in \
    "${root}/lib/libamdhip64.so" \
    "${root}/lib64/libamdhip64.so" \
    "${root}/lib/libamdhip64.so.6" \
    "${root}/lib64/libamdhip64.so.6"; do
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ghalo_frontier_verify_rccl_rocm_consistency() {
  local version="6.4.2"
  local expected_root="/opt/rocm-${version}"
  local hip_compiler
  local hip_library
  local rccl_library

  [[ -n "${ROCM_PATH:-}" ]] ||
    ghalo_die "Frontier RCCL setup requires ROCM_PATH after loading rocm/${version}"
  if [[ "${ROCM_PATH}" != "${expected_root}" ]] &&
     ! ghalo_frontier_path_matches_rocm_version "${ROCM_PATH}" "${version}"; then
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  expected=${expected_root}"
  fi

  export RCCL_ROOT="${ROCM_PATH}"
  [[ -n "${OLCF_OFI_NCCL_ROOT:-}" ]] ||
    ghalo_die "Frontier RCCL setup requires OLCF_OFI_NCCL_ROOT from rccl-net-plugin/1.0"

  hip_compiler="$(command -v hipcc 2>/dev/null || true)"
  [[ -n "${hip_compiler}" ]] ||
    ghalo_die "Frontier RCCL setup requires hipcc from rocm/${version}"
  hip_compiler="$(readlink -f "${hip_compiler}" 2>/dev/null || printf '%s\n' "${hip_compiler}")"
  if ! ghalo_frontier_path_matches_rocm_version "${hip_compiler}" "${version}"; then
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  RCCL_ROOT=${RCCL_ROOT}
  hipcc=${hip_compiler}"
  fi

  [[ "${RCCL_ROOT}" == "${ROCM_PATH}" ]] ||
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  RCCL_ROOT=${RCCL_ROOT}
  hipcc=${hip_compiler}"
  ghalo_frontier_path_matches_rocm_version "${RCCL_ROOT}" "${version}" ||
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  RCCL_ROOT=${RCCL_ROOT}
  hipcc=${hip_compiler}"

  ghalo_frontier_rccl_header_exists "${RCCL_ROOT}" ||
    ghalo_die "Frontier RCCL setup could not find ${RCCL_ROOT}/include/rccl/rccl.h"
  rccl_library="$(ghalo_frontier_resolve_rccl_library "${RCCL_ROOT}")" ||
    ghalo_die "Frontier RCCL setup could not find librccl.so under RCCL_ROOT='${RCCL_ROOT}'"
  ghalo_frontier_path_matches_rocm_version "${rccl_library}" "${version}" ||
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  RCCL_ROOT=${RCCL_ROOT}
  hipcc=${hip_compiler}
  librccl=${rccl_library}"

  hip_library="$(ghalo_frontier_resolve_hip_library "${ROCM_PATH}")" ||
    ghalo_die "Frontier RCCL setup could not find libamdhip64.so under ROCM_PATH='${ROCM_PATH}'"
  ghalo_frontier_path_matches_rocm_version "${hip_library}" "${version}" ||
    ghalo_die "Frontier RCCL environment mismatch:
  ROCM_PATH=${ROCM_PATH}
  RCCL_ROOT=${RCCL_ROOT}
  hipcc=${hip_compiler}
  libamdhip64=${hip_library}
  librccl=${rccl_library}"

  export GHALO_LOADED_ROCM_MODULE=rocm/${version}
  export GHALO_ROCM_VERSION="${version}"
  export GHALO_RESOLVED_HIP_COMPILER="${hip_compiler}"
  export GHALO_RESOLVED_HIP_LIBRARY="${hip_library}"
  export GHALO_RESOLVED_RCCL_LIBRARY="${rccl_library}"
}

ghalo_frontier_load_mpi_hip_gpu() {
  local hip_compiler
  local hip_library
  ghalo_frontier_load_gpu
  ghalo_frontier_unload_rocm_and_rccl
  module load rocm/6.4.2 ||
    ghalo_die "failed to load required Frontier ROCm module rocm/6.4.2"
  export GHALO_LOADED_ROCM_MODULE=rocm/6.4.2
  export GHALO_ROCM_VERSION=6.4.2
  hip_compiler="$(command -v hipcc 2>/dev/null || true)"
  [[ -n "${hip_compiler}" ]] ||
    ghalo_die "Frontier MPI-HIP setup requires hipcc from rocm/6.4.2"
  hip_compiler="$(readlink -f "${hip_compiler}" 2>/dev/null || printf '%s\n' "${hip_compiler}")"
  ghalo_frontier_path_matches_rocm_version "${hip_compiler}" "6.4.2" ||
    ghalo_die "Frontier MPI-HIP environment mismatch: hipcc='${hip_compiler}' does not match rocm/6.4.2"
  hip_library="$(ghalo_frontier_resolve_hip_library "${ROCM_PATH}")" ||
    ghalo_die "Frontier MPI-HIP setup could not find libamdhip64.so under ROCM_PATH='${ROCM_PATH}'"
  ghalo_frontier_path_matches_rocm_version "${hip_library}" "6.4.2" ||
    ghalo_die "Frontier MPI-HIP environment mismatch: libamdhip64='${hip_library}' does not match rocm/6.4.2"
  export GHALO_RESOLVED_HIP_COMPILER="${hip_compiler}"
  export GHALO_RESOLVED_HIP_LIBRARY="${hip_library}"
  unset RCCL_ROOT
  unset OLCF_OFI_NCCL_ROOT
  unset GHALO_RESOLVED_RCCL_LIBRARY
}

ghalo_frontier_load_rccl_gpu() {
  ghalo_frontier_load_gpu
  ghalo_frontier_unload_rocm_and_rccl
  module load rocm/6.4.2 ||
    ghalo_die "failed to load required Frontier ROCm module rocm/6.4.2"
  module load rccl-net-plugin/1.0 ||
    ghalo_die "failed to load required Frontier RCCL network plugin rccl-net-plugin/1.0"
  ghalo_frontier_verify_rccl_rocm_consistency
}

ghalo_frontier_rccl_availability_hint() {
  cat <<'EOF'
Inspect RCCL availability with:
  module avail rccl
  module spider rccl
  find "${ROCM_PATH:-/opt/rocm}" -name 'librccl.so*' 2>/dev/null
  find "${ROCM_PATH:-/opt/rocm}" -path '*include*' \( -name rccl.h -o -name nccl.h \) 2>/dev/null
Set RCCL_ROOT, RCCL_PATH, or ROCM_PATH if RCCL is installed outside default search paths.
EOF
}

ghalo_system_setup_build() {
  local backend="$1"
  ghalo_frontier_load_common
  if [[ "${backend}" == "mpi-hip" ]]; then
    ghalo_frontier_load_mpi_hip_gpu
  elif [[ "${backend}" == "rccl" ]]; then
    ghalo_frontier_load_rccl_gpu
  fi
  if [[ "${backend}" == "mpi-hip" ]]; then
    export MPICH_GPU_SUPPORT_ENABLED=1
    if [[ -n "${MPICH_DIR:-}" && ! -f "${MPICH_DIR}/include/mpi.h" ]]; then
      ghalo_die "MPICH_DIR is set to '${MPICH_DIR}', but '${MPICH_DIR}/include/mpi.h' does not exist"
    fi
  else
    unset MPICH_GPU_SUPPORT_ENABLED
  fi
}

ghalo_system_setup_run() {
  local backend="$1"
  ghalo_frontier_load_common
  if [[ "${backend}" == "mpi-hip" ]]; then
    ghalo_frontier_load_mpi_hip_gpu
  elif [[ "${backend}" == "rccl" ]]; then
    ghalo_frontier_load_rccl_gpu
  fi
  if [[ "${backend}" == "mpi-hip" ]]; then
    export MPICH_GPU_SUPPORT_ENABLED=1
  else
    unset MPICH_GPU_SUPPORT_ENABLED
  fi
}

ghalo_system_cmake_args() {
  local backend="$1"
  printf '%s\n' -DCMAKE_CXX_COMPILER=CC
  if [[ "${backend}" == "mpi-hip" || "${backend}" == "rccl" ]]; then
    printf '%s\n' -DCMAKE_HIP_ARCHITECTURES=gfx90a
  fi
}

ghalo_system_default_partition() {
  printf '%s\n' batch
}

ghalo_system_launch() {
  local nodes="$2"
  local ranks="$3"
  local ranks_per_node="$4"
  local extra_srun_args="$5"
  shift 5

  local command=(srun -N "${nodes}" -n "${ranks}")
  if [[ -n "${ranks_per_node}" ]]; then
    command+=(--ntasks-per-node "${ranks_per_node}")
  fi
  if [[ -n "${extra_srun_args}" ]]; then
    local extra_args=()
    read -r -a extra_args <<<"${extra_srun_args}"
    command+=("${extra_args[@]}")
  fi

  command+=("$@")
  printf '%s\n' "${command[@]}"
}
