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
