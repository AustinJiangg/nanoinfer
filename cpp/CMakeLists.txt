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
  # G6 (CUDA graphs): compile the .cu with the PER-THREAD default stream so the implicit
  # <<<>>> launches land on a CAPTURABLE stream — the legacy stream 0 can't be graph-captured.
  # The engine is single-threaded, so this doesn't change ordering (the one thread's default
  # stream stays in-order); it only makes the decode forward recordable into a cudaGraph.
  target_compile_options(nicore PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
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

  # G4a: batched decode on GPU — forward_batch == N standalone forwards + throughput.
  add_executable(run_cuda_batch tests/run_cuda_batch.cpp)
  target_link_libraries(run_cuda_batch PRIVATE nicore)
  target_include_directories(run_cuda_batch PRIVATE tests)

  # G4b: paged KV cache on GPU — paged-attention kernel == contiguous cache.
  add_executable(run_cuda_paged tests/run_cuda_paged.cpp)
  target_link_libraries(run_cuda_paged PRIVATE nicore)
  target_include_directories(run_cuda_paged PRIVATE tests)

  # G5a: micro-benchmark the naive linear() kernel vs cuBLAS sgemm (the yardstick) on
  # the real projection shapes. cuBLAS is only linked here, never in nicore — the engine
  # runs our own kernels (the golden rule); this tool just measures the gap to close.
  add_executable(run_cuda_bench tests/run_cuda_bench.cpp)
  target_link_libraries(run_cuda_bench PRIVATE nicore CUDA::cublas)
  target_include_directories(run_cuda_bench PRIVATE tests)

  # G5b: end-to-end decode tok/s, A/B'ing the GEMV vs the forced-naive GEMM in one process.
  add_executable(run_cuda_decode_bench tests/run_cuda_decode_bench.cpp)
  target_link_libraries(run_cuda_decode_bench PRIVATE nicore)
  target_include_directories(run_cuda_decode_bench PRIVATE tests)

  # G5f: isolated attention() micro-bench — naive vs non-tiled online vs shared-mem tiled, on
  # synthetic data (no weights), so the kernel change isn't diluted by the full step.
  add_executable(run_cuda_attn_bench tests/run_cuda_attn_bench.cpp)
  target_link_libraries(run_cuda_attn_bench PRIVATE nicore)
  target_include_directories(run_cuda_attn_bench PRIVATE tests)
endif()

# R0/R1: the weight-dependent parity/golden binaries as a labelled `weights` ctest suite, so the
# whole gate is one `ctest` (the other tests need only the in-tree fixtures). Gated on the weight
# export + reference dump being present — skip-clean when absent, the same graceful degrade as the
# GPU tests. The parity binaries return nonzero on regression; run_quant is informational (always
# exits 0), so its baseline match-count is pinned via PASS_REGULAR_EXPRESSION. Numbers in
# tests/BASELINE.md. Run just this suite with `ctest -L weights` (or exclude it with `ctest -LE weights`).
set(NI_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/weights/qwen2.5-0.5b)
if(EXISTS ${NI_WEIGHTS_DIR}/ref_ids.txt)
  add_test(NAME run_parity   COMMAND run_parity   ${NI_WEIGHTS_DIR})
  add_test(NAME run_generate COMMAND run_generate ${NI_WEIGHTS_DIR})
  add_test(NAME run_cache    COMMAND run_cache    ${NI_WEIGHTS_DIR})
  add_test(NAME run_batch    COMMAND run_batch    ${NI_WEIGHTS_DIR})
  add_test(NAME run_paged    COMMAND run_paged    ${NI_WEIGHTS_DIR})
  set(NI_WEIGHTS_TESTS run_parity run_generate run_cache run_batch run_paged)

  # run_quant reports but doesn't gate (always exits 0) — pin its printed match-count so a quant
  # regression fails ctest, including q4's expected-lossy 0/12 (locks even the lossy baseline).
  add_test(NAME run_quant_none COMMAND run_quant ${NI_WEIGHTS_DIR} none)
  add_test(NAME run_quant_q8   COMMAND run_quant ${NI_WEIGHTS_DIR} q8)
  add_test(NAME run_quant_q4   COMMAND run_quant ${NI_WEIGHTS_DIR} q4)
  add_test(NAME run_quant_q4g  COMMAND run_quant ${NI_WEIGHTS_DIR} q4g)
  set_tests_properties(run_quant_none PROPERTIES PASS_REGULAR_EXPRESSION "matching fp32: 12/12")
  set_tests_properties(run_quant_q8   PROPERTIES PASS_REGULAR_EXPRESSION "matching fp32: 11/12")
  set_tests_properties(run_quant_q4   PROPERTIES PASS_REGULAR_EXPRESSION "matching fp32: 0/12")
  set_tests_properties(run_quant_q4g  PROPERTIES PASS_REGULAR_EXPRESSION "matching fp32: 5/12")
  list(APPEND NI_WEIGHTS_TESTS run_quant_none run_quant_q8 run_quant_q4 run_quant_q4g)

  if(NI_CUDA)
    add_test(NAME run_cuda_parity COMMAND run_cuda_parity ${NI_WEIGHTS_DIR})
    add_test(NAME run_cuda_cache  COMMAND run_cuda_cache  ${NI_WEIGHTS_DIR})
    add_test(NAME run_cuda_paged  COMMAND run_cuda_paged  ${NI_WEIGHTS_DIR})
    add_test(NAME run_cuda_batch  COMMAND run_cuda_batch  ${NI_WEIGHTS_DIR})
    list(APPEND NI_WEIGHTS_TESTS run_cuda_parity run_cuda_cache run_cuda_paged run_cuda_batch)
  endif()

  set_tests_properties(${NI_WEIGHTS_TESTS} PROPERTIES LABELS weights)
else()
  message(STATUS "weights/qwen2.5-0.5b/ref_ids.txt absent: skipping the `weights` ctest suite "
                 "(run tools/export_weights.py + tools/dump_reference.py to enable)")
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
