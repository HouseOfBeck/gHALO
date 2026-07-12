cmake_minimum_required(VERSION 3.24)

project(
  gHALO
  VERSION 0.3.0
  DESCRIPTION "GPU-native halo exchange benchmark and diagnostic suite"
  HOMEPAGE_URL "https://github.com/gHALO/gHALO"
  LANGUAGES CXX
)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GHALO_ENABLE_HIP "Enable HIP language support for GPU kernels" OFF)
option(GHALO_ENABLE_MPI "Enable MPI support for distributed halo exchange" OFF)
option(GHALO_ENABLE_MPI_HIP "Enable GPU-aware MPI backend using HIP device buffers" OFF)
option(GHALO_ENABLE_RCCL "Enable RCCL communication backend support" OFF)
option(GHALO_ENABLE_UCX "Enable UCX communication backend support" OFF)
option(GHALO_ENABLE_WARNINGS "Enable project warning flags where supported" ON)

if(GHALO_ENABLE_MPI_HIP AND NOT GHALO_ENABLE_MPI)
  message(FATAL_ERROR "GHALO_ENABLE_MPI_HIP requires GHALO_ENABLE_MPI=ON")
endif()

if(GHALO_ENABLE_MPI_HIP AND NOT GHALO_ENABLE_HIP)
  message(FATAL_ERROR "GHALO_ENABLE_MPI_HIP requires GHALO_ENABLE_HIP=ON")
endif()

if(GHALO_ENABLE_RCCL AND NOT GHALO_ENABLE_MPI)
  message(FATAL_ERROR "GHALO_ENABLE_RCCL requires GHALO_ENABLE_MPI=ON")
endif()

if(GHALO_ENABLE_RCCL AND NOT GHALO_ENABLE_HIP)
  message(FATAL_ERROR "GHALO_ENABLE_RCCL requires GHALO_ENABLE_HIP=ON")
endif()

add_library(ghalo_project_options INTERFACE)
target_compile_features(ghalo_project_options INTERFACE cxx_std_20)

if(GHALO_ENABLE_WARNINGS)
  target_compile_options(
    ghalo_project_options
    INTERFACE
      $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
endif()

if(GHALO_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
endif()

if(GHALO_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(GHALO_ENABLE_UCX)
  find_package(UCX REQUIRED)
endif()

if(GHALO_ENABLE_RCCL)
  set(GHALO_RCCL_ROOT_HINTS)
  foreach(GHALO_RCCL_ROOT_ENV RCCL_ROOT RCCL_PATH ROCM_PATH)
    if(DEFINED ENV{${GHALO_RCCL_ROOT_ENV}})
      list(APPEND GHALO_RCCL_ROOT_HINTS "$ENV{${GHALO_RCCL_ROOT_ENV}}")
    endif()
  endforeach()

  find_package(rccl CONFIG QUIET)
  if(TARGET rccl::rccl OR TARGET rccl)
    if(TARGET rccl::rccl)
      set(GHALO_RCCL_PACKAGE_TARGET rccl::rccl)
    else()
      set(GHALO_RCCL_PACKAGE_TARGET rccl)
    endif()
    get_target_property(
      GHALO_RCCL_TARGET_INCLUDE_DIRS
      ${GHALO_RCCL_PACKAGE_TARGET}
      INTERFACE_INCLUDE_DIRECTORIES
    )
    get_target_property(
      GHALO_RCCL_TARGET_LIBRARY
      ${GHALO_RCCL_PACKAGE_TARGET}
      IMPORTED_LOCATION
    )
    if(NOT GHALO_RCCL_TARGET_INCLUDE_DIRS)
      set(
        GHALO_RCCL_TARGET_INCLUDE_DIRS
        "provided by ${GHALO_RCCL_PACKAGE_TARGET}"
      )
    endif()
    if(NOT GHALO_RCCL_TARGET_LIBRARY)
      set(GHALO_RCCL_TARGET_LIBRARY "provided by ${GHALO_RCCL_PACKAGE_TARGET}")
    endif()
    message(STATUS "  RCCL CMake package: found")
    message(STATUS "  RCCL target: ${GHALO_RCCL_PACKAGE_TARGET}")
    message(STATUS "  RCCL include directory: ${GHALO_RCCL_TARGET_INCLUDE_DIRS}")
    message(STATUS "  RCCL library: ${GHALO_RCCL_TARGET_LIBRARY}")
    message(STATUS "  RCCL version hint: provided by CMake package")
  else()
    find_path(
      GHALO_RCCL_INCLUDE_DIR
      NAMES rccl/rccl.h rccl.h nccl.h
      HINTS ${GHALO_RCCL_ROOT_HINTS}
      PATH_SUFFIXES include include/rccl
    )
    find_library(
      GHALO_RCCL_LIBRARY
      NAMES rccl
      HINTS ${GHALO_RCCL_ROOT_HINTS}
      PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    )
    if(NOT GHALO_RCCL_INCLUDE_DIR OR NOT GHALO_RCCL_LIBRARY)
      message(
        FATAL_ERROR
        "GHALO_ENABLE_RCCL=ON requires RCCL headers and librccl. "
        "Set RCCL_ROOT, RCCL_PATH, or ROCM_PATH if RCCL is installed in a "
        "non-standard location."
      )
    endif()
    add_library(ghalo_rccl_external INTERFACE)
    target_include_directories(
      ghalo_rccl_external
      INTERFACE
        ${GHALO_RCCL_INCLUDE_DIR}
    )
    target_link_libraries(
      ghalo_rccl_external
      INTERFACE
        ${GHALO_RCCL_LIBRARY}
    )
    message(STATUS "  RCCL include directory: ${GHALO_RCCL_INCLUDE_DIR}")
    message(STATUS "  RCCL library: ${GHALO_RCCL_LIBRARY}")
    if(EXISTS "${GHALO_RCCL_INCLUDE_DIR}/rccl/rccl.h")
      file(
        STRINGS "${GHALO_RCCL_INCLUDE_DIR}/rccl/rccl.h"
        GHALO_RCCL_VERSION_LINES
        REGEX "RCCL_VERSION|NCCL_VERSION"
      )
    elseif(EXISTS "${GHALO_RCCL_INCLUDE_DIR}/rccl.h")
      file(
        STRINGS "${GHALO_RCCL_INCLUDE_DIR}/rccl.h"
        GHALO_RCCL_VERSION_LINES
        REGEX "RCCL_VERSION|NCCL_VERSION"
      )
    elseif(EXISTS "${GHALO_RCCL_INCLUDE_DIR}/nccl.h")
      file(
        STRINGS "${GHALO_RCCL_INCLUDE_DIR}/nccl.h"
        GHALO_RCCL_VERSION_LINES
        REGEX "RCCL_VERSION|NCCL_VERSION"
      )
    endif()
    if(GHALO_RCCL_VERSION_LINES)
      list(GET GHALO_RCCL_VERSION_LINES 0 GHALO_RCCL_VERSION_LINE)
      message(STATUS "  RCCL version hint: ${GHALO_RCCL_VERSION_LINE}")
    else()
      message(STATUS "  RCCL version hint: not discovered")
    endif()
  endif()
  message(STATUS "  HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

message(STATUS "gHALO configuration")
message(STATUS "  HIP:  ${GHALO_ENABLE_HIP}")
message(STATUS "  MPI:  ${GHALO_ENABLE_MPI}")
message(STATUS "  MPI+HIP: ${GHALO_ENABLE_MPI_HIP}")
message(STATUS "  RCCL: ${GHALO_ENABLE_RCCL}")
message(STATUS "  UCX:  ${GHALO_ENABLE_UCX}")

add_library(
  ghalo_core
  src/core/cli.cpp
  src/core/benchmark.cpp
  src/core/output.cpp
)
target_link_libraries(ghalo_core PUBLIC ghalo_project_options)
target_include_directories(
  ghalo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(GHALO_ENABLE_MPI)
  add_library(ghalo_mpi_backend src/backends/mpi/mpi_backend.cpp)
  target_link_libraries(
    ghalo_mpi_backend
    PUBLIC
      ghalo_core
      MPI::MPI_CXX
  )
  target_include_directories(
    ghalo_mpi_backend
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mpi
  )

  if(GHALO_ENABLE_MPI_HIP)
    add_library(ghalo_mpi_hip_backend src/backends/mpi_hip/mpi_hip_backend.hip)
    target_link_libraries(
      ghalo_mpi_hip_backend
      PUBLIC
        ghalo_core
        MPI::MPI_CXX
    )

    set(GHALO_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
    get_target_property(
      GHALO_MPI_TARGET_INCLUDE_DIRS
      MPI::MPI_CXX
      INTERFACE_INCLUDE_DIRECTORIES
    )
    if(GHALO_MPI_TARGET_INCLUDE_DIRS)
      list(APPEND GHALO_MPI_INCLUDE_DIRS ${GHALO_MPI_TARGET_INCLUDE_DIRS})
    endif()
    if(NOT GHALO_MPI_INCLUDE_DIRS
       AND DEFINED ENV{MPICH_DIR}
       AND EXISTS "$ENV{MPICH_DIR}/include/mpi.h")
      list(APPEND GHALO_MPI_INCLUDE_DIRS "$ENV{MPICH_DIR}/include")
    endif()
    if(NOT GHALO_MPI_INCLUDE_DIRS)
      message(
        FATAL_ERROR
        "GHALO_ENABLE_MPI_HIP requires an explicit MPI header directory for "
        "HIP compilation. CMake did not expose MPI_CXX_INCLUDE_DIRS or "
        "MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES, and MPICH_DIR/include/mpi.h "
        "was not available."
      )
    endif()
    list(REMOVE_DUPLICATES GHALO_MPI_INCLUDE_DIRS)
    set(GHALO_MPI_HEADER_FOUND OFF)
    foreach(GHALO_MPI_INCLUDE_DIR IN LISTS GHALO_MPI_INCLUDE_DIRS)
      if(EXISTS "${GHALO_MPI_INCLUDE_DIR}/mpi.h")
        set(GHALO_MPI_HEADER_FOUND ON)
      endif()
    endforeach()
    if(NOT GHALO_MPI_HEADER_FOUND)
      message(
        FATAL_ERROR
        "GHALO_ENABLE_MPI_HIP found MPI include directories, but none contain "
        "mpi.h: ${GHALO_MPI_INCLUDE_DIRS}"
      )
    endif()
    message(STATUS "  MPI headers for HIP: ${GHALO_MPI_INCLUDE_DIRS}")

    target_include_directories(
      ghalo_mpi_hip_backend
      PRIVATE
        ${GHALO_MPI_INCLUDE_DIRS}
    )
    target_compile_options(
      ghalo_mpi_hip_backend
      PRIVATE
        ${MPI_CXX_COMPILE_OPTIONS}
    )
    if(TARGET hip::host)
      target_link_libraries(ghalo_mpi_hip_backend PUBLIC hip::host)
    elseif(TARGET hip::device)
      target_link_libraries(ghalo_mpi_hip_backend PUBLIC hip::device)
    endif()
    target_include_directories(
      ghalo_mpi_hip_backend
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mpi_hip
    )
  endif()

  if(GHALO_ENABLE_RCCL)
    add_library(ghalo_rccl_backend src/rccl_backend.cpp)
    target_link_libraries(
      ghalo_rccl_backend
      PUBLIC
        ghalo_core
        MPI::MPI_CXX
    )
    if(TARGET rccl::rccl)
      target_link_libraries(ghalo_rccl_backend PUBLIC rccl::rccl)
    elseif(TARGET rccl)
      target_link_libraries(ghalo_rccl_backend PUBLIC rccl)
    else()
      target_link_libraries(ghalo_rccl_backend PUBLIC ghalo_rccl_external)
    endif()
    if(TARGET hip::host)
      target_link_libraries(ghalo_rccl_backend PUBLIC hip::host)
    elseif(TARGET hip::device)
      target_link_libraries(ghalo_rccl_backend PUBLIC hip::device)
    endif()
    target_compile_definitions(ghalo_rccl_backend PUBLIC GHALO_HAVE_RCCL=1)
  endif()

  add_executable(ghalo src/app/main.cpp)
  target_link_libraries(ghalo PRIVATE ghalo_mpi_backend)
  if(GHALO_ENABLE_MPI_HIP)
    target_link_libraries(ghalo PRIVATE ghalo_mpi_hip_backend)
    target_compile_definitions(ghalo PRIVATE GHALO_HAVE_MPI_HIP=1)
  endif()
  if(GHALO_ENABLE_RCCL)
    target_link_libraries(ghalo PRIVATE ghalo_rccl_backend)
    target_compile_definitions(ghalo PRIVATE GHALO_HAVE_RCCL=1)
  endif()
else()
  message(STATUS "gHALO executable target is disabled because GHALO_ENABLE_MPI=OFF")
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
