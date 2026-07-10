cmake_minimum_required(VERSION 3.24)

project(
  gHALO
  VERSION 0.1.0
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

if(GHALO_ENABLE_RCCL)
  find_package(rccl REQUIRED)
endif()

if(GHALO_ENABLE_UCX)
  find_package(UCX REQUIRED)
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

  add_executable(ghalo src/app/main.cpp)
  target_link_libraries(ghalo PRIVATE ghalo_mpi_backend)
  if(GHALO_ENABLE_MPI_HIP)
    target_link_libraries(ghalo PRIVATE ghalo_mpi_hip_backend)
    target_compile_definitions(ghalo PRIVATE GHALO_HAVE_MPI_HIP=1)
  endif()
else()
  message(STATUS "gHALO executable target is disabled because GHALO_ENABLE_MPI=OFF")
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
