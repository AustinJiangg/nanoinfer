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
  src/config.cpp src/model.cpp src/sampling.cpp src/generate.cpp src/paged.cpp
  src/backend.cpp)
target_include_directories(nicore PUBLIC src)
# PIC so the static core can be linked into the pybind11 shared module (F6); the
# executables that also link nicore just become PIE, which is the modern default.
set_target_properties(nicore PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  # PUBLIC so the -fopenmp flag (and _OPENMP define) and runtime propagate to the
  # test/bench executables that include parallel.hpp.
  target_link_libraries(nicore PUBLIC OpenMP::OpenMP_CXX)
endif()

# G0/G1: the CUDA backend seam. OFF by default — the C++ core builds and runs on CPU
# with no CUDA toolkit (the same graceful-degrade as OpenMP / -march=native). When ON,
# enable the CUDA language, define NI_CUDA so the CudaBackend compiles in, and (G1+)
# add src/cuda/*.cu to nicore. The Backend interface keeps model.cpp unchanged either way.
option(NI_CUDA "Build the CUDA backend (requires the NVIDIA CUDA toolkit; G1+)" OFF)
if(NI_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(nicore PUBLIC NI_CUDA)
  target_sources(nicore PRIVATE src/cuda/cuda_backend.cu)
  target_link_libraries(nicore PUBLIC CUDA::cudart)
  # sm_89 = Ada Lovelace (RTX 4070 Super); CUDA_STANDARD 17 matches the C++ side.
  set_target_properties(nicore PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 89)
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

# F8a: batched decode parity (forward_batch == per-sequence forward) + throughput.
add_executable(run_batch tests/run_batch.cpp)
target_link_libraries(run_batch PRIVATE nicore)
target_include_directories(run_batch PRIVATE tests)

# F8b: paged KV cache parity (paged forward == contiguous forward) + pool utilization.
add_executable(run_paged tests/run_paged.cpp)
target_link_libraries(run_paged PRIVATE nicore)
target_include_directories(run_paged PRIVATE tests)

# G1: CUDA backend parity — the hand-written GEMM matches the CPU backend within float
# tolerance. Self-contained (synthetic data) so it joins ctest; skips at runtime if no
# GPU is present. Only built when -DNI_CUDA=ON.
if(NI_CUDA)
  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE nicore)
  target_include_directories(test_cuda PRIVATE tests)
  add_test(NAME test_cuda COMMAND test_cuda)

  # G2: full GPU forward vs the nanoinfer reference (logits + greedy). Needs the weight
  # export + ref dump (like run_parity), so it is built but not auto-run by ctest.
  add_executable(run_cuda_parity tests/run_cuda_parity.cpp)
  target_link_libraries(run_cuda_parity PRIVATE nicore)
  target_include_directories(run_cuda_parity PRIVATE tests)

  # G3: device-resident KV cache — cached==uncached + real incremental greedy decode.
  add_executable(run_cuda_cache tests/run_cuda_cache.cpp)
  target_link_libraries(run_cuda_cache PRIVATE nicore)
  target_include_directories(run_cuda_cache PRIVATE tests)
endif()

# F6: pybind11 module exposing the C++ core to Python — the vLLM-shape pivot
# (Python orchestration over our own kernels). Optional and auto-located via the
# already-found Python3 (`pip install pybind11`); skipped cleanly when pybind11 is
# absent, the same graceful degrade as OpenMP / -march=native. The .so lands in the
# build dir; tests/tools add it to sys.path (a pip-installable wheel is a later step).
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE NI_PYBIND11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NI_PYBIND11_QUERY
  ERROR_QUIET)
if(NI_PYBIND11_QUERY EQUAL 0)
  find_package(pybind11 CONFIG QUIET PATHS "${NI_PYBIND11_DIR}")
endif()
# pybind11_add_module needs the Development.Module component (the Python3 find above
# requested only Interpreter, to keep the C++-only build dependency-light); add it
# here, optionally, so python3_add_library is defined when the headers are present.
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(pybind11_FOUND AND Python3_Development.Module_FOUND)
  pybind11_add_module(nicpp bindings/nicpp.cpp)
  target_link_libraries(nicpp PRIVATE nicore)
  message(STATUS "pybind11 ${pybind11_VERSION} found: building Python module 'nicpp'")
else()
  message(STATUS "pybind11 / Python dev headers not found: skipping Python module")
endif()
