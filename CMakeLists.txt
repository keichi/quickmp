cmake_minimum_required(VERSION 3.15...4.1)
project(quickmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Detect VEDA
list(APPEND CMAKE_MODULE_PATH /opt/nec/ve/share/veoffload-veda/cmake)
list(APPEND CMAKE_PREFIX_PATH /opt/nec/ve3/lib)
find_package(VEDA QUIET)

# Python/nanobind configuration
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

if(VEDA_FOUND)
  message(STATUS "VEDA found - Building for VE")
  enable_language(VEDA_CXX)
  include_directories(${VEDA_INCLUDE_DIRS})

  add_library(quickmp-device SHARED
    src/kernel/dot_product.vcpp
    src/kernel/stats.vcpp
    src/kernel/stomp.vcpp)
  target_include_directories(quickmp-device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  nanobind_add_module(_quickmp src/bindings_ve.cpp)
  target_link_libraries(_quickmp PRIVATE ${VEDA_LIBRARY})

  install(TARGETS quickmp-device DESTINATION quickmp)
  install(TARGETS _quickmp DESTINATION quickmp)
else()
  message(STATUS "VEDA not found - Building for CPU")

  if(DEFINED ENV{RELEASE_FLAGS})
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} $ENV{RELEASE_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
  endif()

  add_library(quickmp src/cpu/dot_product.cpp src/cpu/stats.cpp src/cpu/stomp.cpp)
  target_include_directories(quickmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set_target_properties(quickmp PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set_source_files_properties(src/cpu/stomp.cpp PROPERTIES COMPILE_FLAGS "-qopt-zmm-usage=high")
  endif()

  add_executable(bench src/bench.cpp)
  target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(bench PRIVATE quickmp)

  nanobind_add_module(_quickmp src/bindings_cpu.cpp)
  target_include_directories(_quickmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(_quickmp PRIVATE quickmp)

  install(TARGETS _quickmp DESTINATION quickmp)
endif()
