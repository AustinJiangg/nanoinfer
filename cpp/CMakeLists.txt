cmake_minimum_required(VERSION 3.16)
project(nanoinfer_cpp LANGUAGES CXX)

# A from-scratch CPU engine. Release (-O3) by default; stage C5 adds SIMD
# (AVX2/FMA) and OpenMP threads, both of which degrade gracefully — the kernels
# fall back to scalar / serial when the ISA or OpenMP is unavailable.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# C5: target the host ISA so the AVX2/FMA intrinsics in simd.hpp compile in (they
# are guarded by __AVX2__/__FMA__, which -march=native defines) and the compiler
# can auto-vectorize the elementwise ops. Set -DNI_NATIVE=OFF for a portable
# binary — the inner products then use the scalar fallback, same numbers.
option(NI_NATIVE "Build for the host CPU (-march=native; enables AVX2/FMA)" ON)
if(NI_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" NI_HAS_MARCH_NATIVE)
  if(NI_HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# C5: OpenMP for multithreading the output-channel loops. Optional — without it
# the kernels run serially (the #pragma omp lines compile out).
find_package(OpenMP COMPONENTS CXX)

# Core library. Kept as a plain C++ lib with a simple API so it can be wrapped by
# pybind11 later (the F-stage fusion) without reshaping it.
add_library(nicore
  src/tensor.cpp src/ops.cpp src/serialize.cpp src/cache.cpp src/quant.cpp
  src/config.cpp src/model.cpp src/sampling.cpp src/generate.cpp)
target_include_directories(nicore PUBLIC src)
if(OpenMP_CXX_FOUND)
  # PUBLIC so the -fopenmp flag (and _OPENMP define) and runtime propagate to the
  # test/bench executables that include parallel.hpp.
  target_link_libraries(nicore PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()

# Self-contained unit tests (no external inputs).
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE nicore)
target_include_directories(test_tensor PRIVATE tests)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE nicore)
target_include_directories(test_ops PRIVATE tests)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_io tests/test_io.cpp)
target_link_libraries(test_io PRIVATE nicore)
target_include_directories(test_io PRIVATE tests)
add_test(NAME test_io COMMAND test_io)

add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE nicore)
target_include_directories(test_sampling PRIVATE tests)
add_test(NAME test_sampling COMMAND test_sampling)

add_executable(test_cache tests/test_cache.cpp)
target_link_libraries(test_cache PRIVATE nicore)
target_include_directories(test_cache PRIVATE tests)
add_test(NAME test_cache COMMAND test_cache)

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE nicore)
target_include_directories(test_quant PRIVATE tests)
add_test(NAME test_quant COMMAND test_quant)

# C5: the SIMD dot helpers match a scalar reference (incl. the non-multiple-of-8
# tail) and any threaded run is deterministic across thread counts.
add_executable(test_simd tests/test_simd.cpp)
target_link_libraries(test_simd PRIVATE nicore)
target_include_directories(test_simd PRIVATE tests)
add_test(NAME test_simd COMMAND test_simd)

# numpy parity: a setup test generates fixtures, then ops_parity checks against them.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(ops_parity tests/ops_parity.cpp)
target_link_libraries(ops_parity PRIVATE nicore)
target_include_directories(ops_parity PRIVATE tests)

add_test(NAME gen_fixtures
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_fixtures.py
                 ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(gen_fixtures PROPERTIES FIXTURES_SETUP fixtures)

add_test(NAME ops_parity COMMAND ops_parity ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(ops_parity PROPERTIES FIXTURES_REQUIRED fixtures)

# End-to-end vs nanoinfer. Not wired into ctest: they need the ~2 GB weight
# export + a reference dump first (see the file headers).
add_executable(run_parity tests/run_parity.cpp)
target_link_libraries(run_parity PRIVATE nicore)
target_include_directories(run_parity PRIVATE tests)

add_executable(run_generate tests/run_generate.cpp)
target_link_libraries(run_generate PRIVATE nicore)
target_include_directories(run_generate PRIVATE tests)

add_executable(run_cache tests/run_cache.cpp)
target_link_libraries(run_cache PRIVATE nicore)
target_include_directories(run_cache PRIVATE tests)

add_executable(run_quant tests/run_quant.cpp)
target_link_libraries(run_quant PRIVATE nicore)
target_include_directories(run_quant PRIVATE tests)

# C5: time prefill + decode (tokens/sec) and report the SIMD target + thread count.
add_executable(run_bench tests/run_bench.cpp)
target_link_libraries(run_bench PRIVATE nicore)
target_include_directories(run_bench PRIVATE tests)
