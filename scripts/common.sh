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
    mpi | mpi-hip) ;;
    *) ghalo_die "unsupported backend '${backend}'. Expected 'mpi' or 'mpi-hip'." ;;
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
  # shellcheck source=/dev/null
  source "${config}"
}

ghalo_build_dir() {
  local root="$1"
  local system="$2"
  local backend="$3"
  printf '%s/builds/%s/%s\n' "${root}" "${system}" "${backend}"
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
      MPICH_DIR \
      MPICH_GPU_SUPPORT_ENABLED \
      MPI_HOME \
      ROCM_PATH \
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
