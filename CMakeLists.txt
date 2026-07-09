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
option(GHALO_ENABLE_RCCL "Enable RCCL communication backend support" OFF)
option(GHALO_ENABLE_UCX "Enable UCX communication backend support" OFF)
option(GHALO_ENABLE_WARNINGS "Enable project warning flags where supported" ON)

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
  find_package(hip REQUIRED)
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
message(STATUS "  RCCL: ${GHALO_ENABLE_RCCL}")
message(STATUS "  UCX:  ${GHALO_ENABLE_UCX}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

add_library(
  ghalo_core
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

  add_executable(ghalo src/app/main.cpp)
  target_link_libraries(ghalo PRIVATE ghalo_mpi_backend)
else()
  message(STATUS "gHALO executable target is disabled because GHALO_ENABLE_MPI=OFF")
endif()
