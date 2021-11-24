cmake_minimum_required(VERSION 3.0)
project(codegen)

set(sources
  ${CMAKE_CURRENT_LIST_DIR}/output/parameters.hpp
  ${CMAKE_CURRENT_LIST_DIR}/output/parameters.cpp)
add_custom_command(OUTPUT
    ${sources}
    COMMAND ./generate_parameters.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS parameter_definitions.c parameters_base.cpp parameters_base.hpp generate_parameters.py)
add_custom_target(generated_parameters DEPENDS ${sources})

# These could be conditionally compiled BUILD_CLI_TOOLS, but there
# are some unfixed issues with the cmd parameter classes being used
# from non-command line code.
set(cmd_sources
  ${CMAKE_CURRENT_LIST_DIR}/output/cmd_parameters.hpp
  ${CMAKE_CURRENT_LIST_DIR}/output/cmd_parameters.cpp)
add_custom_command(OUTPUT
    ${cmd_sources}
    COMMAND ./generate_parameters.py cmd
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS cmd_parameter_definitions.json cmd_parameters_base.cpp cmd_parameters_base.hpp generate_parameters.py)
add_custom_target(cmd_generated_parameters DEPENDS ${cmd_sources})

set(generated_sources ${sources} ${cmd_sources} PARENT_SCOPE)
