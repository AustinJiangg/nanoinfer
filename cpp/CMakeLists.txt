cmake_minimum_required(VERSION 3.16)
project(nanoinfer_cpp LANGUAGES CXX)

# C0: a from-scratch CPU engine. Reference build (-O2); SIMD/threads come in C5.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Core library. Kept as a plain C++ lib with a simple API so it can be wrapped by
# pybind11 later (the F-stage fusion) without reshaping it.
add_library(nicore
  src/tensor.cpp src/ops.cpp src/serialize.cpp
  src/config.cpp src/model.cpp src/sampling.cpp src/generate.cpp)
target_include_directories(nicore PUBLIC src)

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

add_executable(run_generate tests/run_generate.cpp)
target_link_libraries(run_generate PRIVATE nicore)
