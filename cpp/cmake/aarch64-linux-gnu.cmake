# Cross-compile the C++ core for aarch64 (Apple-M-class ARM64) and run the test
# binaries under qemu-user — so the NEON path in simd.hpp is parity-tested on the
# x86 dev box, the same "test the seam" discipline as every other backend.
#
# This proves CORRECTNESS (the NEON dots reproduce the scalar/AVX2 oracle); it does
# NOT benchmark — qemu emulates the ISA, not the M4's timing, so real tok/s waits
# for the actual M4. Correctness-first is the project's spine; the NEON intrinsics
# are the hard part and they ARE verifiable here.
#
# Usage (run from the repo root):
#   cmake -S cpp -B cpp/build-aarch64 \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake \
#         -DNI_NATIVE=OFF \
#         -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON \
#         -DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON
#   cmake --build cpp/build-aarch64 --target test_simd ops_parity test_ops test_quant
#   ctest --test-dir cpp/build-aarch64 -R 'test_simd|test_ops|test_quant|ops_parity' -V
#
# NI_NATIVE=OFF: -march=native is an x86 host flag, meaningless for the cross
# compiler. aarch64's base ISA mandates NEON, so <arm_neon.h> and the float64x2
# intrinsics are available with no extra -march flag.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Run cross-built test executables through qemu. ctest picks this up via each
# executable's CROSSCOMPILING_EMULATOR property (initialized from this variable);
# -L points qemu at the aarch64 sysroot for the dynamic loader + libstdc++/libc.
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64-static -L /usr/aarch64-linux-gnu)

# Target libs/headers come from the cross sysroot; host programs (python3, for the
# numpy fixture generator that feeds ops_parity) are found on the host.
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
