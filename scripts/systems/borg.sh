#!/usr/bin/env bash

ghalo_borg_have_modules() {
  type module >/dev/null 2>&1
}

ghalo_borg_require_modules() {
  ghalo_borg_have_modules ||
    ghalo_die "Borg GPU backend runs require environment modules to load rocm/6.2.4"
}

ghalo_borg_load_common() {
  if ghalo_borg_have_modules; then
    module load PrgEnv-cray >/dev/null 2>&1 || true
  fi
  command -v CC >/dev/null 2>&1 ||
    ghalo_die "Borg builds require the Cray C++ wrapper 'CC'"
}

ghalo_borg_load_gpu() {
  ghalo_borg_require_modules
  module load craype-accel-amd-gfx90a ||
    ghalo_die "failed to load craype-accel-amd-gfx90a on Borg"
  module load rocm/6.2.4 ||
    ghalo_die "failed to load required Borg ROCm module rocm/6.2.4"
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
  if [[ "${GHALO_USE_NATIVE_BUILD:-0}" == "1" ]]; then
    printf '%s\n' "${GHALO_ACTIVE_SYSTEM}"
  elif [[ -n "${GHALO_BUILD_SYSTEM_ALIAS:-}" ]]; then
    printf '%s\n' "${GHALO_BUILD_SYSTEM_ALIAS}"
  else
    printf '%s\n' frontier
  fi
}

ghalo_system_setup_build() {
  local backend="$1"
  ghalo_borg_load_common
  if [[ "${backend}" == "mpi-hip" || "${backend}" == "rccl" ]]; then
    ghalo_borg_load_gpu
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
  if [[ "${backend}" == "mpi-hip" || "${backend}" == "rccl" ]]; then
    ghalo_borg_load_gpu
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
