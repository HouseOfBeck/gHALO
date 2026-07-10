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
    module load rocm
  fi
}

ghalo_system_setup_build() {
  local backend="$1"
  ghalo_frontier_load_common
  if [[ "${backend}" == "mpi-hip" ]]; then
    ghalo_frontier_load_gpu
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
    ghalo_frontier_load_gpu
    export MPICH_GPU_SUPPORT_ENABLED=1
  else
    unset MPICH_GPU_SUPPORT_ENABLED
  fi
}

ghalo_system_cmake_args() {
  local backend="$1"
  printf '%s\n' -DCMAKE_CXX_COMPILER=CC
  if [[ "${backend}" == "mpi-hip" ]]; then
    printf '%s\n' -DCMAKE_HIP_ARCHITECTURES=gfx90a
  fi
}

ghalo_system_launch() {
  local backend="$1"
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
