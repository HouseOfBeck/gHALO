#!/usr/bin/env bash

set -euo pipefail

ghalo_die() {
  echo "gHALO workflow error: $*" >&2
  exit 1
}

ghalo_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

ghalo_short_hostname() {
  hostname -s 2>/dev/null || hostname
}

ghalo_resolve_system() {
  local requested="${1:-}"
  if [[ -n "${GHALO_SYSTEM_NAME:-}" ]]; then
    printf '%s\n' "${GHALO_SYSTEM_NAME}"
  elif [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
  else
    ghalo_short_hostname
  fi
}

ghalo_validate_system_name() {
  local system="$1"
  [[ "${system}" =~ ^[A-Za-z0-9._-]+$ ]] ||
    ghalo_die "invalid system name '${system}'. Use letters, numbers, '.', '_', or '-'."
}

ghalo_validate_backend() {
  local backend="$1"
  case "${backend}" in
    mpi | mpi-hip | rccl) ;;
    *) ghalo_die "unsupported backend '${backend}'. Expected 'mpi', 'mpi-hip', or 'rccl'." ;;
  esac
}

ghalo_sanitize_label() {
  local label="${1:-run}"
  label="${label// /_}"
  label="$(printf '%s' "${label}" | tr -c 'A-Za-z0-9._-' '_')"
  label="${label##_}"
  label="${label%%_}"
  [[ -n "${label}" ]] || label="run"
  printf '%s\n' "${label}"
}

ghalo_load_system_config() {
  local root="$1"
  local system="$2"
  local config="${root}/scripts/systems/${system}.sh"
  [[ -f "${config}" ]] ||
    ghalo_die "missing system configuration '${config}'. Set GHALO_SYSTEM_NAME to a stable cluster name or add scripts/systems/${system}.sh."
  # The system profile path is validated before this dynamic source.
  # shellcheck disable=SC1090
  source "${config}"
}

ghalo_build_dir() {
  local root="$1"
  local system="$2"
  local backend="$3"
  printf '%s/builds/%s/%s\n' "${root}" "${system}" "${backend}"
}

ghalo_binary_path() {
  local root="$1"
  local build_system="$2"
  local backend="$3"
  printf '%s/ghalo\n' "$(ghalo_build_dir "${root}" "${build_system}" "${backend}")"
}

ghalo_require_binary() {
  local binary="$1"
  local active_system="$2"
  local build_system="$3"
  local backend="$4"
  [[ -x "${binary}" ]] ||
    ghalo_die "missing executable '${binary}' for active system '${active_system}' using build system '${build_system}'. Build it first with scripts/build.sh --system ${build_system} --backend ${backend}, or set GHALO_USE_NATIVE_BUILD=1 to use a native ${active_system} build."
}

ghalo_write_system_resolution() {
  local output="$1"
  local active_system="$2"
  local build_system="$3"
  local backend="$4"
  local binary="$5"
  {
    printf 'active_system=%s\n' "${active_system}"
    printf 'build_system=%s\n' "${build_system}"
    printf 'backend=%s\n' "${backend}"
    printf 'binary=%s\n' "${binary}"
  } >"${output}"
}

ghalo_system_build_alias() {
  if [[ -n "${GHALO_BUILD_SYSTEM_ALIAS:-}" ]]; then
    printf '%s\n' "${GHALO_BUILD_SYSTEM_ALIAS}"
  else
    printf '%s\n' "${GHALO_ACTIVE_SYSTEM:-}"
  fi
}

ghalo_capture_modules() {
  local output="$1"
  if command -v module >/dev/null 2>&1; then
    module list >"${output}" 2>&1 || true
  else
    printf 'Environment modules command not available.\n' >"${output}"
  fi
}

ghalo_capture_git() {
  local output="$1"
  {
    printf 'branch: '
    git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown\n'
    printf 'commit: '
    git rev-parse HEAD 2>/dev/null || printf 'unknown\n'
    printf 'dirty: '
    if git diff --quiet --ignore-submodules -- 2>/dev/null &&
       git diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
      printf 'false\n'
    else
      printf 'true\n'
    fi
    git status --short 2>/dev/null || true
  } >"${output}"
}

ghalo_capture_environment() {
  local output="$1"
  {
    env | sort
    printf '\n# Relevant MPI/HIP/ROCm variables\n'
    for name in \
      GHALO_SYSTEM_NAME \
      GHALO_BUILD_SYSTEM_ALIAS \
      GHALO_USE_NATIVE_BUILD \
      GHALO_SUBMIT_COMMAND \
      GHALO_BATCH_STDOUT \
      GHALO_BATCH_STDERR \
      GHALO_SUBMISSION_SYSTEM \
      GHALO_SUBMISSION_ACCOUNT \
      GHALO_SUBMISSION_PARTITION \
      MPICH_DIR \
      MPICH_GPU_SUPPORT_ENABLED \
      MPI_HOME \
      ROCM_PATH \
      RCCL_ROOT \
      RCCL_PATH \
      OLCF_OFI_NCCL_ROOT \
      GHALO_LOADED_ROCM_MODULE \
      GHALO_RESOLVED_HIP_COMPILER \
      GHALO_RESOLVED_RCCL_LIBRARY \
      HIP_PATH \
      ROCR_VISIBLE_DEVICES \
      HIP_VISIBLE_DEVICES \
      OMP_NUM_THREADS \
      SLURM_JOB_ID \
      SLURM_JOB_NUM_NODES \
      SLURM_NTASKS \
      SLURM_NTASKS_PER_NODE; do
      printf '%s=%s\n' "${name}" "${!name-}"
    done
  } >"${output}"
}

ghalo_capture_compiler() {
  local output="$1"
  {
    if command -v "${CXX:-c++}" >/dev/null 2>&1; then
      "${CXX:-c++}" --version 2>&1 || true
    else
      printf 'CXX compiler not found: %s\n' "${CXX:-c++}"
    fi
    if command -v CC >/dev/null 2>&1; then
      printf '\n# CC\n'
      CC --version 2>&1 || true
    fi
    if command -v hipcc >/dev/null 2>&1; then
      printf '\n# hipcc\n'
      hipcc --version 2>&1 || true
    fi
  } >"${output}"
}

ghalo_capture_cmake() {
  local output="$1"
  if command -v cmake >/dev/null 2>&1; then
    cmake --version >"${output}" 2>&1
  else
    printf 'cmake not found\n' >"${output}"
  fi
}

ghalo_system_setup_build() {
  :
}

ghalo_system_setup_run() {
  :
}

ghalo_system_cmake_args() {
  :
}

ghalo_system_launch() {
  ghalo_die "system configuration did not define ghalo_system_launch"
}

ghalo_system_default_account() {
  :
}

ghalo_system_default_partition() {
  :
}

ghalo_system_default_batch_time() {
  :
}
