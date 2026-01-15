cmake_minimum_required(VERSION 3.18...4.1)
project(quickmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{RELEASE_FLAGS})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} $ENV{RELEASE_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
endif()

# if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=max -qopt-report-phase=vec")
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized-missed")
# endif()

# Detect VEDA
list(APPEND CMAKE_MODULE_PATH /opt/nec/ve/share/veoffload-veda/cmake)
list(APPEND CMAKE_PREFIX_PATH /opt/nec/ve3/lib)
find_package(VEDA QUIET)

# Python/nanobind configuration
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

if(VEDA_FOUND)
  message(STATUS "VEDA found - Building for VE")
  enable_language(VEDA_CXX)
  include_directories(${VEDA_INCLUDE_DIRS})

  add_library(quickmp-device SHARED
    src/ve/dot_product.vcpp
    src/ve/stats.vcpp
    src/ve/stomp.vcpp)
  target_include_directories(quickmp-device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  nanobind_add_module(_quickmp src/bindings.cpp src/ve/backend.cpp)
  target_include_directories(_quickmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(_quickmp PRIVATE ${VEDA_LIBRARY})

  install(TARGETS quickmp-device DESTINATION quickmp)
  install(TARGETS _quickmp DESTINATION quickmp)
else()
  message(STATUS "VEDA not found - Building for CPU")

  add_library(quickmp-core
    src/cpu/dot_product.cpp
    src/cpu/stats.cpp
    src/cpu/stomp.cpp
    src/cpu/backend.cpp)
  target_include_directories(quickmp-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set_target_properties(quickmp-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set_source_files_properties(src/cpu/stomp.cpp PROPERTIES COMPILE_FLAGS "-qopt-zmm-usage=high")
  endif()

  add_executable(bench src/bench.cpp)
  target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(bench PRIVATE quickmp-core)

  nanobind_add_module(_quickmp src/bindings.cpp)
  target_include_directories(_quickmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(_quickmp PRIVATE quickmp-core)

  install(TARGETS _quickmp DESTINATION quickmp)
endif()
