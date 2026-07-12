// Tiny tensor serialization — the parity bridge between Python and C++.
//
// Format "NIT0" (nanoinfer tensor v0), little-endian:
//   bytes 0..3   magic  "NIT0"
//   int32        ndim
//   int32 * ndim shape
//   float32 * N  data (row-major, N = product(shape))
//
// Format "NIT1" (v1, B1) adds a payload dtype so a bf16 export halves the file:
//   bytes 0..3   magic  "NIT1"
//   int32        dtype  (0 = float32, 1 = bfloat16)
//   int32        ndim
//   int32 * ndim shape
//   payload      float32*N or bfloat16(u16)*N (row-major)
//
// The loader reads both; a bf16 payload is inflated to fp32 on load (bits << 16 —
// exact, bf16 is fp32's top half), so the host Tensor and the CPU oracle stay fp32
// and a bf16-of-a-bf16-shipped-checkpoint export loads to BYTE-IDENTICAL fp32
// weights as the fp32 export. C++ save_bin still writes NIT0 (fp32); the bf16
// writer is Python-side (ni/nit0.py, used by the weight exporter).
//
// numpy writes these (cpp/tools/gen_fixtures.py and the nanoinfer weight
// exporter); C++ reads them. Assumes a little-endian host (x86/ARM as used here).
#pragma once

#include <string>

#include "tensor.hpp"

namespace ni {

Tensor load_bin(const std::string& path);
void save_bin(const std::string& path, const Tensor& t);

}  // namespace ni
