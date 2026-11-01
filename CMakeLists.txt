cmake_minimum_required(VERSION 3.20)
project(vecq VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VECQ_BUILD_TESTS "Build unit tests" ON)
option(VECQ_BUILD_BENCH "Build micro-benchmarks" ON)
option(VECQ_WERROR "Treat warnings as errors" OFF)
set(VECQ_SANITIZE "" CACHE STRING "Sanitizers, e.g. address,undefined or thread")

find_package(Threads REQUIRED)

set(VECQ_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(VECQ_WERROR)
  list(APPEND VECQ_WARNINGS -Werror)
endif()

if(VECQ_SANITIZE)
  add_compile_options(-fsanitize=${VECQ_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${VECQ_SANITIZE})
endif()

add_library(vecq_core
  src/table.cpp
  src/kernels.cpp
  src/kernels_scalar.cpp
  src/kernels_neon.cpp
  src/csv.cpp
  src/sql.cpp
  src/exec.cpp)
target_include_directories(vecq_core PUBLIC include)
target_compile_options(vecq_core PRIVATE ${VECQ_WARNINGS})
target_link_libraries(vecq_core PUBLIC Threads::Threads)

add_executable(vecq tools/vecq_main.cpp)
target_link_libraries(vecq PRIVATE vecq_core)
target_compile_options(vecq PRIVATE ${VECQ_WARNINGS})

add_executable(vecq-gen tools/gen_trips.cpp)
target_compile_options(vecq-gen PRIVATE ${VECQ_WARNINGS})

include(FetchContent)

if(VECQ_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_executable(vecq_tests
    tests/kernels_test.cpp
    tests/csv_test.cpp
    tests/sql_test.cpp
    tests/exec_test.cpp
    tests/differential_test.cpp)
  target_link_libraries(vecq_tests PRIVATE vecq_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(vecq_tests DISCOVERY_TIMEOUT 60)
endif()

if(VECQ_BUILD_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_executable(vecq_bench bench/kernels_bench.cpp)
  target_link_libraries(vecq_bench PRIVATE vecq_core benchmark::benchmark)
endif()
