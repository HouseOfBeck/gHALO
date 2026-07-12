#!/usr/bin/env bash

# shellcheck source-path=SCRIPTDIR

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

# This test intentionally mutates environment variables inside a subshell so
# alias settings cannot leak into later workflow tests.
# shellcheck disable=SC2030,SC2031
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

# This test intentionally mutates GHALO_ACTIVE_SYSTEM inside a subshell so the
# generic default-alias case is isolated from other tests.
# shellcheck disable=SC2030,SC2031
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

test_backend_validation_accepts_rccl() (
  ghalo_validate_backend rccl
  local output="${GHALO_TEST_TMPDIR}/ghalo-backend-validation.txt"
  if (ghalo_validate_backend bad-backend) >"${output}" 2>&1; then
    printf 'expected invalid backend to fail\n' >&2
    exit 1
  fi
  assert_contains "Expected 'mpi', 'mpi-hip', or 'rccl'" \
    "${output}" "backend validation lists rccl"
)

# This test intentionally modifies the active system inside a subshell while
# verifying Borg's default build alias behavior.
# shellcheck disable=SC2030,SC2031
test_borg_rccl_uses_frontier_build_alias() (
  export GHALO_ACTIVE_SYSTEM=borg
  unset GHALO_BUILD_SYSTEM_ALIAS
  unset GHALO_USE_NATIVE_BUILD
  # shellcheck source=../scripts/systems/borg.sh
  source "${ROOT}/scripts/systems/borg.sh"

  assert_eq frontier "$(ghalo_system_build_alias rccl)" \
    "Borg rccl default alias"
)

test_frontier_rccl_cmake_args() (
  # shellcheck source=../scripts/systems/frontier.sh
  source "${ROOT}/scripts/systems/frontier.sh"

  local output="${GHALO_TEST_TMPDIR}/ghalo-frontier-rccl-cmake.txt"
  ghalo_system_cmake_args rccl >"${output}"
  assert_contains '-DCMAKE_CXX_COMPILER=CC' "${output}" \
    "Frontier rccl uses Cray wrapper"
  assert_contains '-DCMAKE_HIP_ARCHITECTURES=gfx90a' "${output}" \
    "Frontier rccl sets HIP architecture"
)

# This test mocks Frontier modules to verify backend-specific ROCm selection
# without requiring OLCF modules on the local workstation.
# shellcheck disable=SC2030,SC2031
test_frontier_backend_specific_rocm_selection() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-frontier-backend-bin"
  local rocm624="${GHALO_TEST_TMPDIR}/frontier/rocm-6.2.4"
  local rocm642="${GHALO_TEST_TMPDIR}/frontier/rocm-6.4.2"
  local plugin_root="${GHALO_TEST_TMPDIR}/frontier/rccl-net-plugin-1.0"
  local module_log="${GHALO_TEST_TMPDIR}/ghalo-frontier-module-log.txt"
  local base_path
  local expected_rocm642
  mkdir -p "${bin_dir}" "${rocm624}/bin" "${rocm642}/bin" \
    "${rocm642}/include/rccl" "${rocm642}/lib" "${plugin_root}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm624}/bin/hipcc"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm642}/bin/hipcc"
  : >"${rocm642}/include/rccl/rccl.h"
  : >"${rocm642}/lib/librccl.so"
  : >"${rocm642}/lib/libamdhip64.so"
  chmod +x "${bin_dir}/CC" "${rocm624}/bin/hipcc" "${rocm642}/bin/hipcc"
  export PATH="${bin_dir}:${PATH}"
  base_path="${PATH}"
  expected_rocm642="$(readlink -f "${rocm642}")"

  # The mock is called indirectly by scripts/systems/frontier.sh.
  # shellcheck disable=SC2317
  module() {
    printf '%s %s\n' "$1" "${2:-}" >>"${module_log}"
    case "$1" in
      load)
        case "$2" in
          PrgEnv-cray | craype-accel-amd-gfx90a)
            return 0
            ;;
          rocm/6.2.4)
            export ROCM_PATH="${rocm624}"
            export PATH="${rocm624}/bin:${base_path}"
            unset OLCF_OFI_NCCL_ROOT
            return 0
            ;;
          rocm/6.4.2)
            export ROCM_PATH="${rocm642}"
            export PATH="${rocm642}/bin:${base_path}"
            return 0
            ;;
          rccl-net-plugin/1.0)
            export OLCF_OFI_NCCL_ROOT="${plugin_root}"
            return 0
            ;;
        esac
        ;;
      unload)
        if [[ "${2:-}" == rocm* ]]; then
          unset ROCM_PATH
          export PATH="${base_path}"
          return 0
        fi
        if [[ "${2:-}" == rccl-net-plugin* ]]; then
          unset OLCF_OFI_NCCL_ROOT
          return 0
        fi
        ;;
    esac
    return 1
  }

  # shellcheck source=../scripts/systems/frontier.sh
  source "${ROOT}/scripts/systems/frontier.sh"

  export OLCF_OFI_NCCL_ROOT="${plugin_root}"
  ghalo_system_setup_build mpi-hip
  assert_eq rocm/6.2.4 "${GHALO_LOADED_ROCM_MODULE}" \
    "Frontier mpi-hip selects ROCm 6.2.4"
  assert_eq "${rocm624}" "${ROCM_PATH}" "Frontier mpi-hip ROCM_PATH"
  if [[ -n "${OLCF_OFI_NCCL_ROOT+x}" ]]; then
    printf 'Frontier mpi-hip setup should unload rccl-net-plugin/1.0\n' >&2
    exit 1
  fi

  ghalo_system_setup_build rccl
  assert_eq rocm/6.4.2 "${GHALO_LOADED_ROCM_MODULE}" \
    "Frontier rccl selects ROCm 6.4.2"
  assert_eq "${rocm642}" "${ROCM_PATH}" "Frontier rccl ROCM_PATH"
  assert_eq "${rocm642}" "${RCCL_ROOT}" "Frontier rccl RCCL_ROOT"
  assert_eq "${plugin_root}" "${OLCF_OFI_NCCL_ROOT}" \
    "Frontier rccl loads OFI plugin"
  assert_eq "${expected_rocm642}/bin/hipcc" "${GHALO_RESOLVED_HIP_COMPILER}" \
    "Frontier rccl resolved hipcc"
  assert_eq "${rocm642}/lib/libamdhip64.so" "${GHALO_RESOLVED_HIP_LIBRARY}" \
    "Frontier rccl resolved libamdhip64"
  assert_eq "${rocm642}/lib/librccl.so" "${GHALO_RESOLVED_RCCL_LIBRARY}" \
    "Frontier rccl resolved librccl"
  assert_contains 'load rccl-net-plugin/1.0' "${module_log}" \
    "Frontier rccl loads rccl-net-plugin"
)

# This test intentionally mutates PATH and ROCm-related variables inside a
# subshell while validating Frontier mixed-version failure handling.
# shellcheck disable=SC2030,SC2031
test_frontier_rccl_mixed_rocm_versions_fail() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-frontier-mixed-bin"
  local rocm624="${GHALO_TEST_TMPDIR}/frontier-mixed/rocm-6.2.4"
  local rocm642="${GHALO_TEST_TMPDIR}/frontier-mixed/rocm-6.4.2"
  local plugin_root="${GHALO_TEST_TMPDIR}/frontier-mixed/rccl-net-plugin-1.0"
  local output="${GHALO_TEST_TMPDIR}/ghalo-frontier-mixed-rocm.txt"
  local base_path
  mkdir -p "${bin_dir}" "${rocm624}/bin" "${rocm642}/include/rccl" \
    "${rocm642}/lib" "${plugin_root}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm624}/bin/hipcc"
  : >"${rocm642}/include/rccl/rccl.h"
  : >"${rocm642}/lib/librccl.so"
  : >"${rocm642}/lib/libamdhip64.so"
  chmod +x "${bin_dir}/CC" "${rocm624}/bin/hipcc"
  export PATH="${bin_dir}:${rocm624}/bin:${PATH}"
  base_path="${PATH}"

  # The mock is called indirectly by scripts/systems/frontier.sh.
  # shellcheck disable=SC2317
  module() {
    case "$1" in
      load)
        case "$2" in
          PrgEnv-cray | craype-accel-amd-gfx90a)
            return 0
            ;;
          rocm/6.4.2)
            export ROCM_PATH="${rocm642}"
            export PATH="${base_path}"
            return 0
            ;;
          rccl-net-plugin/1.0)
            export OLCF_OFI_NCCL_ROOT="${plugin_root}"
            return 0
            ;;
        esac
        ;;
      unload)
        return 0
        ;;
    esac
    return 1
  }

  # shellcheck source=../scripts/systems/frontier.sh
  source "${ROOT}/scripts/systems/frontier.sh"

  if (ghalo_system_setup_build rccl) >"${output}" 2>&1; then
    printf 'expected mixed Frontier RCCL ROCm setup to fail\n' >&2
    exit 1
  fi
  assert_contains 'RCCL environment mismatch' "${output}" \
    "Frontier rccl rejects mixed ROCm versions"
)

# This test intentionally mocks PATH and MPICH_GPU_SUPPORT_ENABLED inside a
# subshell so the Borg environment setup cannot affect later tests.
# shellcheck disable=SC2030,SC2031
test_borg_environment_setup() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-borg-workflow-bin"
  local rocm_marker="${GHALO_TEST_TMPDIR}/ghalo-borg-rocm-loaded.txt"
  mkdir -p "${bin_dir}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/srun"
  chmod +x "${bin_dir}/CC" "${bin_dir}/srun"
  export PATH="${bin_dir}:${PATH}"

  # The mock is called indirectly by scripts/systems/borg.sh.
  # shellcheck disable=SC2317
  module() {
    if [[ "$1" != "load" ]]; then
      return 1
    fi
    case "$2" in
      PrgEnv-cray | craype-accel-amd-gfx90a)
        return 0
        ;;
      rocm/6.2.4)
        printf 'loaded\n' >"${rocm_marker}"
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

  rm -f "${rocm_marker}"
  ghalo_system_setup_run mpi-hip
  assert_eq 1 "${MPICH_GPU_SUPPORT_ENABLED}" "Borg mpi-hip GPU support"
  assert_contains 'loaded' "${rocm_marker}" "Borg ROCm 6.2.4 load"
)

# This test mocks Borg modules inside a subshell to verify backend-specific
# ROCm selection without requiring OLCF modules on the local workstation.
# shellcheck disable=SC2030,SC2031
test_borg_backend_specific_rocm_selection() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-borg-backend-bin"
  local rocm624="${GHALO_TEST_TMPDIR}/opt/rocm-6.2.4"
  local rocm642="${GHALO_TEST_TMPDIR}/opt/rocm-6.4.2"
  local plugin_root="${GHALO_TEST_TMPDIR}/opt/rccl-net-plugin-1.0"
  local module_log="${GHALO_TEST_TMPDIR}/ghalo-borg-module-log.txt"
  local base_path
  mkdir -p "${bin_dir}" "${rocm624}/bin" "${rocm642}/bin" \
    "${rocm642}/include/rccl" "${rocm642}/lib" "${plugin_root}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm624}/bin/hipcc"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm642}/bin/hipcc"
  : >"${rocm642}/include/rccl/rccl.h"
  : >"${rocm642}/lib/librccl.so"
  chmod +x "${bin_dir}/CC" "${rocm624}/bin/hipcc" "${rocm642}/bin/hipcc"
  export PATH="${bin_dir}:${PATH}"
  base_path="${PATH}"

  # The mock is called indirectly by scripts/systems/borg.sh.
  # shellcheck disable=SC2317
  module() {
    printf '%s %s\n' "$1" "${2:-}" >>"${module_log}"
    case "$1" in
      load)
        case "$2" in
          PrgEnv-cray | craype-accel-amd-gfx90a)
            return 0
            ;;
          rocm/6.2.4)
            export ROCM_PATH="${rocm624}"
            export PATH="${rocm624}/bin:${base_path}"
            unset OLCF_OFI_NCCL_ROOT
            return 0
            ;;
          rocm/6.4.2)
            export ROCM_PATH="${rocm642}"
            export PATH="${rocm642}/bin:${base_path}"
            return 0
            ;;
          rccl-net-plugin/1.0)
            export OLCF_OFI_NCCL_ROOT="${plugin_root}"
            return 0
            ;;
        esac
        ;;
      unload)
        if [[ "${2:-}" == rocm* ]]; then
          unset ROCM_PATH
          export PATH="${base_path}"
          return 0
        fi
        ;;
    esac
    return 1
  }

  # shellcheck source=../scripts/systems/borg.sh
  source "${ROOT}/scripts/systems/borg.sh"

  ghalo_system_setup_build mpi-hip
  assert_eq rocm/6.2.4 "${GHALO_LOADED_ROCM_MODULE}" \
    "Borg mpi-hip selects ROCm 6.2.4"
  assert_eq "${rocm624}" "${ROCM_PATH}" "Borg mpi-hip ROCM_PATH"
  if [[ -n "${OLCF_OFI_NCCL_ROOT+x}" ]]; then
    printf 'Borg mpi-hip setup should not load rccl-net-plugin/1.0\n' >&2
    exit 1
  fi

  ghalo_system_setup_build rccl
  assert_eq rocm/6.4.2 "${GHALO_LOADED_ROCM_MODULE}" \
    "Borg rccl selects ROCm 6.4.2"
  assert_eq "${rocm642}" "${ROCM_PATH}" "Borg rccl ROCM_PATH"
  assert_eq "${rocm642}" "${RCCL_ROOT}" "Borg rccl RCCL_ROOT"
  assert_eq "${plugin_root}" "${OLCF_OFI_NCCL_ROOT}" \
    "Borg rccl loads OFI plugin"
  assert_eq "${rocm642}/bin/hipcc" "${GHALO_RESOLVED_HIP_COMPILER}" \
    "Borg rccl resolved hipcc"
  assert_eq "${rocm642}/lib/librccl.so" "${GHALO_RESOLVED_RCCL_LIBRARY}" \
    "Borg rccl resolved librccl"
  assert_contains 'load rccl-net-plugin/1.0' "${module_log}" \
    "Borg rccl loads rccl-net-plugin"
)

# This test intentionally mutates PATH and ROCm-related variables inside a
# subshell while validating mixed-version failure handling.
# shellcheck disable=SC2030,SC2031
test_borg_rccl_mixed_rocm_versions_fail() (
  local bin_dir="${GHALO_TEST_TMPDIR}/ghalo-borg-mixed-bin"
  local rocm624="${GHALO_TEST_TMPDIR}/mixed/rocm-6.2.4"
  local rocm642="${GHALO_TEST_TMPDIR}/mixed/rocm-6.4.2"
  local plugin_root="${GHALO_TEST_TMPDIR}/mixed/rccl-net-plugin-1.0"
  local output="${GHALO_TEST_TMPDIR}/ghalo-borg-mixed-rocm.txt"
  local base_path
  mkdir -p "${bin_dir}" "${rocm624}/bin" "${rocm642}/include/rccl" \
    "${rocm642}/lib" "${plugin_root}"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${bin_dir}/CC"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${rocm624}/bin/hipcc"
  : >"${rocm642}/include/rccl/rccl.h"
  : >"${rocm642}/lib/librccl.so"
  chmod +x "${bin_dir}/CC" "${rocm624}/bin/hipcc"
  export PATH="${bin_dir}:${rocm624}/bin:${PATH}"
  base_path="${PATH}"

  # The mock is called indirectly by scripts/systems/borg.sh.
  # shellcheck disable=SC2317
  module() {
    case "$1" in
      load)
        case "$2" in
          PrgEnv-cray | craype-accel-amd-gfx90a)
            return 0
            ;;
          rocm/6.4.2)
            export ROCM_PATH="${rocm642}"
            export PATH="${base_path}"
            return 0
            ;;
          rccl-net-plugin/1.0)
            export OLCF_OFI_NCCL_ROOT="${plugin_root}"
            return 0
            ;;
        esac
        ;;
      unload)
        return 0
        ;;
    esac
    return 1
  }

  # shellcheck source=../scripts/systems/borg.sh
  source "${ROOT}/scripts/systems/borg.sh"

  if (ghalo_system_setup_build rccl) >"${output}" 2>&1; then
    printf 'expected mixed Borg RCCL ROCm setup to fail\n' >&2
    exit 1
  fi
  assert_contains 'mixed ROCm configuration' "${output}" \
    "Borg rccl rejects mixed ROCm versions"
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
    "${ROOT}" frontier mpi TEST123 batch 1 4 4 0.1 0 0 0 smoke "" submit out err \
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
test_backend_validation_accepts_rccl
test_borg_rccl_uses_frontier_build_alias
test_frontier_rccl_cmake_args
test_frontier_backend_specific_rocm_selection
test_frontier_rccl_mixed_rocm_versions_fail
test_borg_environment_setup
test_borg_backend_specific_rocm_selection
test_borg_rccl_mixed_rocm_versions_fail
test_submit_dry_run
test_submit_rank_layout_failure
test_batch_job_requires_slurm
test_batch_job_uses_explicit_repo_root_from_spool_copy
