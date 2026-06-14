// Tiny tensor serialization — the parity bridge between Python and C++.
//
// Format "NIT0" (nanoinfer tensor v0), little-endian:
//   bytes 0..3   magic  "NIT0"
//   int32        ndim
//   int32 * ndim shape
//   float32 * N  data (row-major, N = product(shape))
//
// numpy writes it (cpp/tools/gen_fixtures.py and, later, the nanoinfer weight
// exporter); C++ reads it. Assumes a little-endian host (x86/ARM as used here).
#pragma once

#include <string>

#include "tensor.hpp"

namespace ni {

Tensor load_bin(const std::string& path);
void save_bin(const std::string& path, const Tensor& t);

}  // namespace ni
