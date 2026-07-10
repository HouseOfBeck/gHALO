#!/usr/bin/env bash

# Copy this file to scripts/systems/<system>.sh and adjust it for the target
# cluster. Keep benchmark semantics in gHALO itself; this file should only
# describe build environment, CMake additions, and launch policy.

ghalo_system_setup_build() {
  local backend="$1"
  # module load cmake mpi rocm
  if [[ "${backend}" == "mpi-hip" ]]; then
    export MPICH_GPU_SUPPORT_ENABLED=1
  else
    unset MPICH_GPU_SUPPORT_ENABLED
  fi
}

ghalo_system_setup_run() {
  local backend="$1"
  if [[ "${backend}" == "mpi-hip" ]]; then
    export MPICH_GPU_SUPPORT_ENABLED=1
  else
    unset MPICH_GPU_SUPPORT_ENABLED
  fi
}

ghalo_system_cmake_args() {
  local backend="$1"
  # printf '%s\n' -DCMAKE_CXX_COMPILER=mpicxx
  if [[ "${backend}" == "mpi-hip" ]]; then
    # printf '%s\n' -DCMAKE_HIP_ARCHITECTURES=<gpu-architecture>
    :
  fi
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
