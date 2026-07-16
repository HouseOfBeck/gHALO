add_executable(ghalo_core_tests core_smoke.cpp)
target_link_libraries(ghalo_core_tests PRIVATE ghalo_core)
add_test(NAME ghalo_core_smoke COMMAND ghalo_core_tests)

find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE)
  add_test(
    NAME ghalo_workflow_smoke
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/workflow_smoke.sh
  )
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME ghalo_analysis_smoke
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_ghalo_analyze.py
  )
  add_test(
    NAME ghalo_iteration_stall_analysis_smoke
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_iteration_stalls.py
  )
endif()

if(GHALO_ENABLE_RCCL)
  add_executable(ghalo_rccl_smoke rccl_smoke.cpp)
  target_link_libraries(
    ghalo_rccl_smoke
    PRIVATE
      ghalo_project_options
      MPI::MPI_CXX
  )
  if(TARGET rccl::rccl)
    target_link_libraries(ghalo_rccl_smoke PRIVATE rccl::rccl)
  elseif(TARGET rccl)
    target_link_libraries(ghalo_rccl_smoke PRIVATE rccl)
  else()
    target_link_libraries(ghalo_rccl_smoke PRIVATE ghalo_rccl_external)
  endif()
  if(TARGET hip::host)
    target_link_libraries(ghalo_rccl_smoke PRIVATE hip::host)
  elseif(TARGET hip::device)
    target_link_libraries(ghalo_rccl_smoke PRIVATE hip::device)
  endif()
endif()
