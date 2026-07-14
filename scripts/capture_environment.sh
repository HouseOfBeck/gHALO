#!/usr/bin/env bash

set -euo pipefail

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
timestamp_slug="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-environment-capture-${timestamp_slug}}"
output_file="${output_dir}/environment.txt"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

mkdir -p "${output_dir}"
: >"${output_file}"

write_heading() {
  local title="$1"
  {
    printf '\n## %s\n\n' "${title}"
  } >>"${output_file}"
}

write_note() {
  local message="$1"
  printf 'NOTE: %s\n' "${message}" >>"${output_file}"
}

capture_command() {
  local title="$1"
  shift

  write_heading "${title}"
  if ! command -v "$1" >/dev/null 2>&1; then
    write_note "command unavailable: $1"
    return 0
  fi
  if ! "$@" >>"${output_file}" 2>&1; then
    write_note "command failed: $*"
  fi
}

capture_shell() {
  local title="$1"
  local command_text="$2"

  write_heading "${title}"
  if ! bash -c "${command_text}" >>"${output_file}" 2>&1; then
    write_note "command failed: ${command_text}"
  fi
}

capture_module_list() {
  write_heading "Loaded modules"
  if type module >/dev/null 2>&1; then
    if ! module list >>"${output_file}" 2>&1; then
      write_note "command failed: module list"
    fi
  else
    write_note "command unavailable: module"
  fi
}

capture_slurm_allocation() {
  write_heading "Slurm allocation"
  {
    printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
    printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME:-}"
    printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-}"
    printf 'SLURM_JOB_RESERVATION=%s\n' "${SLURM_JOB_RESERVATION:-}"
    printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-}"
    printf 'SLURM_JOB_NUM_NODES=%s\n' "${SLURM_JOB_NUM_NODES:-}"
  } >>"${output_file}"

  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    {
      printf '\nExpanded hostnames:\n'
      scontrol show hostnames "${SLURM_JOB_NODELIST}"
    } >>"${output_file}" 2>&1 ||
      write_note "command failed: scontrol show hostnames ${SLURM_JOB_NODELIST}"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    write_note "command unavailable: scontrol"
  else
    write_note "SLURM_JOB_NODELIST is not set"
  fi
}

capture_relevant_environment() {
  write_heading "Relevant environment variables"
  env | sort | awk '
    /^(MPICH|FI_|OFI_|HIP|HSA|ROCM|ROCR|RCCL|NCCL|SLURM)/ {
      print
      found = 1
    }
    END {
      if (!found) {
        print "No matching variables set."
      }
    }
  ' >>"${output_file}" 2>&1
}

capture_git() {
  write_heading "Git"
  printf 'repository_root=%s\n' "${repo_root}" >>"${output_file}"
  (
    cd "${repo_root}"
    printf '\n'
    printf '$ git rev-parse HEAD\n'
    git rev-parse HEAD
    printf '\n$ git status --short --branch\n'
    git status --short --branch
    printf '\n$ git describe --tags --always --dirty\n'
    git describe --tags --always --dirty
  ) >>"${output_file}" 2>&1 ||
    write_note "one or more git commands failed"
}

capture_compiler_toolchain() {
  write_heading "Compiler/toolchain"
  {
    printf '$ command -v CC\n'
    if command -v CC >/dev/null 2>&1; then
      command -v CC
      printf '\n$ CC --version\n'
      CC --version
    else
      printf 'NOTE: command unavailable: CC\n'
    fi
    printf '\n$ command -v hipcc\n'
    if command -v hipcc >/dev/null 2>&1; then
      command -v hipcc
      printf '\n$ hipcc --version\n'
      hipcc --version
    else
      printf 'NOTE: command unavailable: hipcc\n'
    fi
  } >>"${output_file}" 2>&1 || write_note "compiler/toolchain capture failed"
}

capture_mpi_launcher() {
  write_heading "MPI/launcher"
  {
    printf '$ command -v srun\n'
    if command -v srun >/dev/null 2>&1; then
      command -v srun
      printf '\n$ srun --version\n'
      srun --version
    else
      printf 'NOTE: command unavailable: srun\n'
    fi
    printf '\nMPICH_DIR=%s\n' "${MPICH_DIR:-}"
  } >>"${output_file}" 2>&1 || write_note "MPI/launcher capture failed"
}

capture_rccl() {
  local rccl_root="${RCCL_ROOT:-${ROCM_PATH:-}}"
  local rccl_library=""
  local header=""

  write_heading "RCCL"
  {
    printf 'RCCL_ROOT=%s\n' "${RCCL_ROOT:-}"
    if [[ -n "${rccl_root}" ]]; then
      for candidate in \
        "${rccl_root}/lib/librccl.so" \
        "${rccl_root}/lib64/librccl.so" \
        "${rccl_root}/lib/librccl.so.1" \
        "${rccl_root}/lib64/librccl.so.1"; do
        if [[ -e "${candidate}" ]]; then
          rccl_library="${candidate}"
          break
        fi
      done
      for candidate in \
        "${rccl_root}/include/rccl/rccl.h" \
        "${rccl_root}/include/rccl.h" \
        "${rccl_root}/include/nccl.h"; do
        if [[ -f "${candidate}" ]]; then
          header="${candidate}"
          break
        fi
      done
    fi

    printf 'resolved_librccl=%s\n' "${rccl_library}"
    if [[ -n "${header}" ]]; then
      printf 'header=%s\n' "${header}"
      awk '
        /#define[[:space:]]+NCCL_VERSION_CODE/ { print "NCCL_VERSION_CODE=" $3; found = 1 }
        /#define[[:space:]]+RCCL_VERSION_CODE/ { print "RCCL_VERSION_CODE=" $3; found = 1 }
        END { if (!found) print "NOTE: no RCCL/NCCL version macro found in header" }
      ' "${header}"
    else
      printf 'NOTE: RCCL/NCCL header unavailable under RCCL_ROOT or ROCM_PATH\n'
    fi

    if [[ -n "${rccl_library}" ]] && command -v strings >/dev/null 2>&1; then
      version_line="$(
        strings "${rccl_library}" 2>/dev/null |
          grep -Eim 1 'RCCL version|NCCL version|^[0-9]+[.][0-9]+[.][0-9]+$' || true
      )"
      if [[ -n "${version_line}" ]]; then
        printf 'library_version_hint=%s\n' "${version_line}"
      else
        printf 'NOTE: no concise RCCL version string found in library\n'
      fi
    elif [[ -n "${rccl_library}" ]]; then
      printf 'NOTE: command unavailable: strings\n'
    fi
  } >>"${output_file}" 2>&1 || write_note "RCCL capture failed"
}

capture_node_hardware() {
  local node_name
  node_name="$(hostname 2>/dev/null || true)"

  write_heading "Node hardware"
  if [[ -z "${node_name}" ]]; then
    write_note "hostname command failed"
    return 0
  fi
  if ! command -v scontrol >/dev/null 2>&1; then
    write_note "command unavailable: scontrol"
    return 0
  fi
  if ! scontrol show node "${node_name}" 2>&1 |
      tr ' ' '\n' |
      grep -E '^(NodeName|Gres|RealMemory|Sockets|CfgTRES|State|Partitions)=' \
        >>"${output_file}" 2>&1; then
    write_note "command failed: scontrol show node ${node_name}"
  fi
}

write_heading "Timestamp"
printf '%s\n' "${timestamp}" >>"${output_file}"

capture_git
capture_slurm_allocation
capture_module_list
capture_relevant_environment
capture_compiler_toolchain
capture_mpi_launcher
capture_rccl
capture_command "ROCm devices" rocm-smi --showproductname
capture_command "GPU topology" rocm-smi --showtopo
capture_command "CPU topology" lscpu
capture_node_hardware

printf 'Environment captured in %s\n' "${output_file}"
