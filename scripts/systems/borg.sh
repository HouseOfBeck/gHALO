#!/usr/bin/env bash

ghalo_borg_have_modules() {
  type module >/dev/null 2>&1
}

ghalo_borg_require_modules() {
  ghalo_borg_have_modules ||
    ghalo_die "Borg GPU backend runs require environment modules"
}

ghalo_borg_load_common() {
  if ghalo_borg_have_modules; then
    module load PrgEnv-cray >/dev/null 2>&1 || true
  fi
  command -v CC >/dev/null 2>&1 ||
    ghalo_die "Borg builds require the Cray C++ wrapper 'CC'"
}

ghalo_borg_load_gpu() {
  local hip_compiler
  local hip_library
  ghalo_borg_require_modules
  module load craype-accel-amd-gfx90a ||
    ghalo_die "failed to load craype-accel-amd-gfx90a on Borg"
  ghalo_borg_unload_rocm
  module load rocm/6.4.2 ||
    ghalo_die "failed to load required Borg ROCm module rocm/6.4.2"
  export GHALO_LOADED_ROCM_MODULE=rocm/6.4.2
  export GHALO_ROCM_VERSION=6.4.2
  hip_compiler="$(command -v hipcc 2>/dev/null || true)"
  [[ -n "${hip_compiler}" ]] ||
    ghalo_die "Borg MPI-HIP setup requires hipcc from rocm/6.4.2"
  ghalo_borg_path_matches_rocm_version "${hip_compiler}" "6.4.2" ||
    ghalo_die "Borg MPI-HIP mixed ROCm configuration: hipcc='${hip_compiler}' does not match rocm/6.4.2"
  hip_library="$(ghalo_borg_resolve_hip_library "${ROCM_PATH}")" ||
    ghalo_die "Borg MPI-HIP setup could not find libamdhip64.so under ROCM_PATH='${ROCM_PATH}'"
  ghalo_borg_path_matches_rocm_version "${hip_library}" "6.4.2" ||
    ghalo_die "Borg MPI-HIP mixed ROCm configuration: libamdhip64='${hip_library}' does not match rocm/6.4.2"
  export GHALO_RESOLVED_HIP_COMPILER="${hip_compiler}"
  export GHALO_RESOLVED_HIP_LIBRARY="${hip_library}"
  unset RCCL_ROOT
  unset OLCF_OFI_NCCL_ROOT
  unset GHALO_RESOLVED_RCCL_LIBRARY
}

ghalo_borg_unload_rocm() {
  ghalo_borg_require_modules
  module unload rccl-net-plugin >/dev/null 2>&1 || true
  module unload rccl-net-plugin/1.0 >/dev/null 2>&1 || true
  module unload rocm >/dev/null 2>&1 || true
  module unload rocm/6.2.4 >/dev/null 2>&1 || true
  module unload rocm/6.4.2 >/dev/null 2>&1 || true
  module unload rocm/7.0.2 >/dev/null 2>&1 || true
  module unload rocm/7.2.0 >/dev/null 2>&1 || true
}

ghalo_borg_rccl_header_exists() {
  local root="$1"
  [[ -f "${root}/include/rccl/rccl.h" ||
     -f "${root}/include/rccl.h" ||
     -f "${root}/include/nccl.h" ]]
}

ghalo_borg_resolve_rccl_library() {
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

ghalo_borg_resolve_hip_library() {
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

ghalo_borg_path_matches_rocm_version() {
  local path="$1"
  local version="$2"
  [[ "${path}" == *"rocm-${version}"* || "${path}" == *"rocm/${version}"* ]]
}

ghalo_borg_verify_rccl_rocm_consistency() {
  local version="${1:-6.4.2}"
  local expected_root="/opt/rocm-${version}"
  local hip_compiler
  local rccl_library

  [[ -n "${ROCM_PATH:-}" ]] ||
    ghalo_die "Borg RCCL setup requires ROCM_PATH after loading rocm/${version}"
  if [[ "${ROCM_PATH}" != "${expected_root}" ]] &&
     ! ghalo_borg_path_matches_rocm_version "${ROCM_PATH}" "${version}"; then
    ghalo_die "Borg RCCL setup loaded rocm/${version}, but ROCM_PATH='${ROCM_PATH}' does not match ${expected_root}"
  fi

  export RCCL_ROOT="${ROCM_PATH}"
  [[ -n "${OLCF_OFI_NCCL_ROOT:-}" ]] ||
    ghalo_die "Borg RCCL setup requires OLCF_OFI_NCCL_ROOT from rccl-net-plugin/1.0"

  hip_compiler="$(command -v hipcc 2>/dev/null || true)"
  [[ -n "${hip_compiler}" ]] ||
    ghalo_die "Borg RCCL setup requires hipcc from rocm/${version}"
  ghalo_borg_path_matches_rocm_version "${hip_compiler}" "${version}" ||
    ghalo_die "Borg RCCL mixed ROCm configuration: hipcc='${hip_compiler}' does not match rocm/${version}"

  [[ "${RCCL_ROOT}" == "${ROCM_PATH}" ]] ||
    ghalo_die "Borg RCCL mixed ROCm configuration: RCCL_ROOT='${RCCL_ROOT}' differs from ROCM_PATH='${ROCM_PATH}'"
  ghalo_borg_path_matches_rocm_version "${RCCL_ROOT}" "${version}" ||
    ghalo_die "Borg RCCL mixed ROCm configuration: RCCL_ROOT='${RCCL_ROOT}' does not match rocm/${version}"

  ghalo_borg_rccl_header_exists "${RCCL_ROOT}" ||
    ghalo_die "Borg RCCL setup could not find rccl.h or nccl.h under RCCL_ROOT='${RCCL_ROOT}'"
  rccl_library="$(ghalo_borg_resolve_rccl_library "${RCCL_ROOT}")" ||
    ghalo_die "Borg RCCL setup could not find librccl.so under RCCL_ROOT='${RCCL_ROOT}'"
  ghalo_borg_path_matches_rocm_version "${rccl_library}" "${version}" ||
    ghalo_die "Borg RCCL mixed ROCm configuration: RCCL library='${rccl_library}' does not match rocm/${version}"

  export GHALO_LOADED_ROCM_MODULE=rocm/${version}
  export GHALO_ROCM_VERSION="${version}"
  export GHALO_RESOLVED_HIP_COMPILER="${hip_compiler}"
  export GHALO_RESOLVED_RCCL_LIBRARY="${rccl_library}"
}

ghalo_borg_load_rccl_gpu() {
  local version="${GHALO_BORG_RCCL_ROCM_VERSION:-${GHALO_ROCM_VERSION:-6.4.2}}"
  ghalo_borg_require_modules
  module load craype-accel-amd-gfx90a ||
    ghalo_die "failed to load craype-accel-amd-gfx90a on Borg"
  ghalo_borg_unload_rocm
  module load "rocm/${version}" ||
    ghalo_die "failed to load required Borg ROCm module rocm/${version}"
  module load rccl-net-plugin/1.0 ||
    ghalo_die "failed to load required Borg RCCL network plugin rccl-net-plugin/1.0"
  ghalo_borg_verify_rccl_rocm_consistency "${version}"
}

ghalo_borg_rccl_availability_hint() {
  cat <<'EOF'
Inspect RCCL availability with:
  module avail rccl
  module spider rccl
  find "${ROCM_PATH:-/opt/rocm}" -name 'librccl.so*' 2>/dev/null
  find "${ROCM_PATH:-/opt/rocm}" -path '*include*' \( -name rccl.h -o -name nccl.h \) 2>/dev/null
Set RCCL_ROOT, RCCL_PATH, or ROCM_PATH if RCCL is installed outside default search paths.
EOF
}

ghalo_system_build_alias() {
  if [[ -n "${GHALO_BUILD_SYSTEM_ALIAS:-}" ]]; then
    printf '%s\n' "${GHALO_BUILD_SYSTEM_ALIAS}"
  else
    printf '%s\n' "${GHALO_ACTIVE_SYSTEM}"
  fi
}

ghalo_system_setup_build() {
  local backend="$1"
  ghalo_borg_load_common
  if [[ "${backend}" == "mpi-hip" ]]; then
    ghalo_borg_load_gpu
  elif [[ "${backend}" == "rccl" ]]; then
    ghalo_borg_load_rccl_gpu
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
  ghalo_borg_load_common
  command -v srun >/dev/null 2>&1 ||
    ghalo_die "Borg runs require Slurm launcher 'srun'"
  if [[ "${backend}" == "mpi-hip" ]]; then
    ghalo_borg_load_gpu
  elif [[ "${backend}" == "rccl" ]]; then
    ghalo_borg_load_rccl_gpu
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

ghalo_system_launch() {
  local nodes="$2"
  local ranks="$3"
  local ranks_per_node="$4"
  local extra_srun_args="$5"
  shift 5

  local command=(srun --exact -N "${nodes}" -n "${ranks}")
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
