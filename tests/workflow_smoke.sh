#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GHALO_TEST_TMPDIR="${TMPDIR:-/tmp}"
# shellcheck source=../scripts/common.sh
source "${ROOT}/scripts/common.sh"

assert_eq() {
  local expected="$1"
  local actual="$2"
  local label="$3"
  if [[ "${expected}" != "${actual}" ]]; then
    printf 'assertion failed: %s: expected %q, got %q\n' \
      "${label}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

test_borg_build_alias_resolution() (
  export GHALO_ACTIVE_SYSTEM=borg
  unset GHALO_BUILD_SYSTEM_ALIAS
  unset GHALO_USE_NATIVE_BUILD
  # shellcheck source=../scripts/systems/borg.sh
  source "${ROOT}/scripts/systems/borg.sh"

  assert_eq frontier "$(ghalo_system_build_alias mpi)" "Borg default alias"

  export GHALO_BUILD_SYSTEM_ALIAS=frontier-test
  assert_eq frontier-test "$(ghalo_system_build_alias mpi)" \
    "explicit build alias"

  export GHALO_USE_NATIVE_BUILD=1
  assert_eq borg "$(ghalo_system_build_alias mpi)" "native Borg build"
)

test_native_default_for_generic_system() (
  export GHALO_ACTIVE_SYSTEM=example
  unset GHALO_BUILD_SYSTEM_ALIAS
  assert_eq example "$(ghalo_system_build_alias mpi)" "generic native alias"
)

test_missing_aliased_binary_error() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-workflow-missing-binary.txt"
  if (
    ghalo_require_binary \
      "${GHALO_TEST_TMPDIR}/ghalo-missing-binary-for-test" \
      borg frontier mpi
  ) >"${output}" 2>&1; then
    printf 'expected missing binary check to fail\n' >&2
    exit 1
  fi
  grep -q "active system 'borg' using build system 'frontier'" "${output}"
)

test_system_resolution_metadata() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-system-resolution-test.txt"
  ghalo_write_system_resolution \
    "${output}" \
    borg \
    frontier \
    mpi-hip \
    /repo/builds/frontier/mpi-hip/ghalo

  grep -q '^active_system=borg$' "${output}"
  grep -q '^build_system=frontier$' "${output}"
  grep -q '^backend=mpi-hip$' "${output}"
  grep -q '^binary=/repo/builds/frontier/mpi-hip/ghalo$' "${output}"
)

test_borg_environment_setup() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-borg-workflow-bin"
  mkdir -p "${bin_dir}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/srun"
  chmod +x "${bin_dir}/CC" "${bin_dir}/srun"
  export PATH="${bin_dir}:${PATH}"

  module() {
    if [[ "$1" != "load" ]]; then
      return 1
    fi
    case "$2" in
      PrgEnv-cray | craype-accel-amd-gfx90a)
        return 0
        ;;
      rocm/6.2.4)
        export GHALO_TEST_ROCM_624_LOADED=1
        return 0
        ;;
      *)
        return 1
        ;;
    esac
  }

  # shellcheck source=../scripts/systems/borg.sh
  source "${ROOT}/scripts/systems/borg.sh"

  export MPICH_GPU_SUPPORT_ENABLED=1
  ghalo_system_setup_run mpi
  if [[ -n "${MPICH_GPU_SUPPORT_ENABLED+x}" ]]; then
    printf 'Borg CPU MPI setup did not unset MPICH_GPU_SUPPORT_ENABLED\n' >&2
    exit 1
  fi

  unset GHALO_TEST_ROCM_624_LOADED
  ghalo_system_setup_run mpi-hip
  assert_eq 1 "${MPICH_GPU_SUPPORT_ENABLED}" "Borg mpi-hip GPU support"
  assert_eq 1 "${GHALO_TEST_ROCM_624_LOADED}" "Borg ROCm 6.2.4 load"
)

test_submit_dry_run() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-submit-dry-run.txt"
  GHALO_SYSTEM_NAME=frontier "${ROOT}/scripts/submit.sh" \
    --backend mpi \
    --account TEST123 \
    --nodes 2 \
    --ranks 16 \
    --ranks-per-node 8 \
    --time 00:05:00 \
    --target-seconds 0.1 \
    --label "dry run label" \
    --dry-run >"${output}"

  grep -q '^Dry run: not submitting\.$' "${output}"
  grep -q -- '--partition batch' "${output}"
  grep -q -- 'scripts/batch-job.sh' "${output}"
  grep -q -- 'dry\\ run\\ label' "${output}"
)

test_submit_rank_layout_failure() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-submit-rank-failure.txt"
  if GHALO_SYSTEM_NAME=frontier "${ROOT}/scripts/submit.sh" \
    --backend mpi \
    --account TEST123 \
    --nodes 2 \
    --ranks 15 \
    --ranks-per-node 8 \
    --time 00:05:00 \
    --dry-run >"${output}" 2>&1; then
    printf 'expected submit rank-layout check to fail\n' >&2
    exit 1
  fi
  grep -q 'ranks == nodes \* ranks-per-node' "${output}"
)

test_batch_job_requires_slurm() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-batch-job-no-slurm.txt"
  if env -u SLURM_JOB_ID "${ROOT}/scripts/batch-job.sh" \
    frontier mpi TEST123 batch 1 4 4 0.1 0 0 smoke "" submit out err \
    >"${output}" 2>&1; then
    printf 'expected batch-job without SLURM_JOB_ID to fail\n' >&2
    exit 1
  fi
  grep -q 'SLURM_JOB_ID is not set' "${output}"
)

test_borg_build_alias_resolution
test_native_default_for_generic_system
test_missing_aliased_binary_error
test_system_resolution_metadata
test_borg_environment_setup
test_submit_dry_run
test_submit_rank_layout_failure
test_batch_job_requires_slurm
