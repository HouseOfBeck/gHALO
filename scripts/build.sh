#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/build.sh --backend mpi|mpi-hip [options]

Options:
  --system NAME              System configuration name. GHALO_SYSTEM_NAME wins if set.
  --backend mpi|mpi-hip      Backend build to configure.
  --build-type TYPE          CMake build type: Release or Debug. Default: Release.
  --clean                    Remove the selected build directory before configuring.
  --jobs N                   Parallel build jobs passed to cmake --build.
  -h, --help                 Show this help.
EOF
}

root="$(ghalo_repo_root)"
requested_system=""
backend=""
build_type="Release"
clean=false
jobs=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      [[ $# -ge 2 ]] || ghalo_die "--system requires a value"
      requested_system="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || ghalo_die "--backend requires a value"
      backend="$2"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || ghalo_die "--build-type requires a value"
      build_type="$2"
      shift 2
      ;;
    --clean)
      clean=true
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || ghalo_die "--jobs requires a value"
      jobs="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      ghalo_die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${backend}" ]] || ghalo_die "--backend is required"
ghalo_validate_backend "${backend}"
case "${build_type}" in
  Release | Debug) ;;
  *) ghalo_die "--build-type must be Release or Debug" ;;
esac
if [[ -n "${jobs}" && ( ! "${jobs}" =~ ^[0-9]+$ || "${jobs}" -lt 1 ) ]]; then
  ghalo_die "--jobs must be a positive integer"
fi

system="$(ghalo_resolve_system "${requested_system}")"
ghalo_validate_system_name "${system}"
ghalo_load_system_config "${root}" "${system}"

build_dir="$(ghalo_build_dir "${root}" "${system}" "${backend}")"
info_dir="${build_dir}/build-info"

if [[ "${clean}" == true ]]; then
  rm -rf "${build_dir}"
fi
mkdir -p "${info_dir}"

export GHALO_ACTIVE_SYSTEM="${system}"
export GHALO_ACTIVE_BACKEND="${backend}"
ghalo_system_setup_build "${backend}"

cmake_args=(
  -S "${root}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE="${build_type}"
)

case "${backend}" in
  mpi)
    cmake_args+=(
      -DGHALO_ENABLE_MPI=ON
      -DGHALO_ENABLE_HIP=OFF
      -DGHALO_ENABLE_MPI_HIP=OFF
    )
    ;;
  mpi-hip)
    cmake_args+=(
      -DGHALO_ENABLE_MPI=ON
      -DGHALO_ENABLE_HIP=ON
      -DGHALO_ENABLE_MPI_HIP=ON
    )
    ;;
esac

while IFS= read -r arg; do
  [[ -n "${arg}" ]] && cmake_args+=("${arg}")
done < <(ghalo_system_cmake_args "${backend}")

printf '%q ' cmake "${cmake_args[@]}" >"${info_dir}/configure-command.txt"
printf '\n' >>"${info_dir}/configure-command.txt"

ghalo_capture_modules "${info_dir}/modules.txt"
ghalo_capture_compiler "${info_dir}/compiler.txt"
ghalo_capture_cmake "${info_dir}/cmake.txt"
ghalo_capture_environment "${info_dir}/environment.txt"
ghalo_capture_git "${info_dir}/git.txt"
printf '%s\n' "${system}" >"${info_dir}/system.txt"
printf '%s\n' "${backend}" >"${info_dir}/backend.txt"

echo "Configuring gHALO ${backend} for system '${system}' in ${build_dir}"
cmake "${cmake_args[@]}"

build_args=(--build "${build_dir}")
if [[ -n "${jobs}" ]]; then
  build_args+=(--parallel "${jobs}")
fi

echo "Building gHALO ${backend}"
cmake "${build_args[@]}"

[[ -x "${build_dir}/ghalo" ]] ||
  ghalo_die "expected executable was not created: ${build_dir}/ghalo"

echo "Build complete: ${build_dir}/ghalo"
