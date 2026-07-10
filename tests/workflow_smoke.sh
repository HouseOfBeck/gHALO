#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GHALO_TEST_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/ghalo-workflow-smoke.XXXXXX")"
trap 'rm -rf "${GHALO_TEST_TMPDIR}"' EXIT
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

assert_contains() {
  local needle="$1"
  local path="$2"
  local label="$3"
  if ! grep -Fq -- "${needle}" "${path}"; then
    printf 'assertion failed: %s: expected to find %q in %s\n' \
      "${label}" "${needle}" "${path}" >&2
    printf '%s\n' "--- ${path} contents ---" >&2
    sed -n '1,160p' "${path}" >&2 || true
    exit 1
  fi
}

assert_not_contains() {
  local needle="$1"
  local path="$2"
  local label="$3"
  if grep -Fq -- "${needle}" "${path}"; then
    printf 'assertion failed: %s: did not expect to find %q in %s\n' \
      "${label}" "${needle}" "${path}" >&2
    printf '%s\n' "--- ${path} contents ---" >&2
    sed -n '1,160p' "${path}" >&2 || true
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
  assert_contains "active system 'borg' using build system 'frontier'" \
    "${output}" "missing binary error identifies alias"
)

test_system_resolution_metadata() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-system-resolution-test.txt"
  ghalo_write_system_resolution \
    "${output}" \
    borg \
    frontier \
    mpi-hip \
    /repo/builds/frontier/mpi-hip/ghalo

  assert_contains 'active_system=borg' "${output}" "active system metadata"
  assert_contains 'build_system=frontier' "${output}" "build system metadata"
  assert_contains 'backend=mpi-hip' "${output}" "backend metadata"
  assert_contains 'binary=/repo/builds/frontier/mpi-hip/ghalo' \
    "${output}" "binary metadata"
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

  assert_contains 'Dry run: not submitting.' "${output}" "dry-run message"
  assert_contains '--partition batch' "${output}" "frontier partition default"
  assert_contains "scripts/batch-job.sh ${ROOT}" \
    "${output}" "repo root follows batch-job path"
  assert_contains 'dry\ run\ label' "${output}" "label is shell escaped"
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
  assert_contains 'ranks == nodes * ranks-per-node' \
    "${output}" "rank layout failure"
)

test_batch_job_requires_slurm() (
  local output="${GHALO_TEST_TMPDIR}/ghalo-batch-job-no-slurm.txt"
  if env -u SLURM_JOB_ID "${ROOT}/scripts/batch-job.sh" \
    "${ROOT}" frontier mpi TEST123 batch 1 4 4 0.1 0 0 smoke "" submit out err \
    >"${output}" 2>&1; then
    printf 'expected batch-job without SLURM_JOB_ID to fail\n' >&2
    exit 1
  fi
  assert_contains 'SLURM_JOB_ID is not set' "${output}" "Slurm guard"
)

test_batch_job_uses_explicit_repo_root_from_spool_copy() (
  local spool_dir="${GHALO_TEST_TMPDIR}/ghalo-slurm-spool-test"
  local fake_repo="${GHALO_TEST_TMPDIR}/ghalo-explicit-root"
  local marker="${GHALO_TEST_TMPDIR}/ghalo-run-marker.txt"
  local output="${GHALO_TEST_TMPDIR}/ghalo-batch-job-spool.txt"
  mkdir -p "${spool_dir}"
  mkdir -p "${fake_repo}/scripts"
  cp "${ROOT}/scripts/batch-job.sh" "${spool_dir}/slurm_script"
  chmod +x "${spool_dir}/slurm_script"
  cat >"${fake_repo}/scripts/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'mock run.sh path=%s\n' "$0" >"${GHALO_TEST_RUN_MARKER}"
printf 'mock run.sh args=%s\n' "$*" >>"${GHALO_TEST_RUN_MARKER}"
EOF
  chmod +x "${fake_repo}/scripts/run.sh"

  GHALO_TEST_RUN_MARKER="${marker}" \
  SLURM_JOB_ID=12345 \
  SLURM_JOB_NAME=ghalo-spool-test \
  SLURM_JOB_NODELIST=node001 \
  "${spool_dir}/slurm_script" \
    "${fake_repo}" \
    frontier \
    mpi \
    TEST123 \
    batch \
    1 \
    4 \
    4 \
    0.1 \
    0 \
    0 \
    spool-test \
    "" \
    submit \
    out-%j \
    err-%j \
    >"${output}" 2>&1

  assert_contains "workdir: ${fake_repo}" "${output}" \
    "batch job reports explicit repo root"
  assert_contains "gHALO batch run exit status: 0" "${output}" \
    "mock run.sh succeeds"
  assert_contains "mock run.sh path=${fake_repo}/scripts/run.sh" "${marker}" \
    "batch job invoked run.sh from explicit root"
  assert_not_contains "${spool_dir}/scripts/run.sh" "${output}" \
    "batch job did not derive run.sh from spool path"
  assert_not_contains "${spool_dir}/scripts/run.sh" "${marker}" \
    "mock marker did not use spool path"
)

test_borg_build_alias_resolution
test_native_default_for_generic_system
test_missing_aliased_binary_error
test_system_resolution_metadata
test_borg_environment_setup
test_submit_dry_run
test_submit_rank_layout_failure
test_batch_job_requires_slurm
test_batch_job_uses_explicit_repo_root_from_spool_copy
