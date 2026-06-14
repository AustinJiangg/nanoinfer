// Parity check against numpy: load inputs + numpy-computed expected outputs
// (written by cpp/tools/gen_fixtures.py), run the C++ ops, assert allclose.
// This is the C++ analogue of nanoinfer's "parity with HuggingFace" floor.
//
// Usage: ops_parity <fixtures_dir>   (ctest passes the build-dir fixtures path).
#include <string>

#include "ops.hpp"
#include "serialize.hpp"
#include "test_util.hpp"

using ni::load_bin;

// eps used for the rmsnorm fixture — must match gen_fixtures.py.
static constexpr float kRmsEps = 1e-6f;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ops_parity <fixtures_dir>\n");
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/";
    int fails = 0;

    try {
        // matmul
        {
            ni::Tensor a = load_bin(dir + "matmul_a.bin");
            ni::Tensor b = load_bin(dir + "matmul_b.bin");
            ni::Tensor expected = load_bin(dir + "matmul_expected.bin");
            fails += compare_tensors(ni::matmul(a, b), expected, 1e-3, "matmul");
        }
        // rmsnorm
        {
            ni::Tensor x = load_bin(dir + "rmsnorm_x.bin");
            ni::Tensor w = load_bin(dir + "rmsnorm_weight.bin");
            ni::Tensor expected = load_bin(dir + "rmsnorm_expected.bin");
            fails += compare_tensors(ni::rmsnorm(x, w, kRmsEps), expected, 1e-4, "rmsnorm");
        }
        // softmax
        {
            ni::Tensor x = load_bin(dir + "softmax_x.bin");
            ni::Tensor expected = load_bin(dir + "softmax_expected.bin");
            fails += compare_tensors(ni::softmax(x), expected, 1e-5, "softmax");
        }
        // add
        {
            ni::Tensor a = load_bin(dir + "add_a.bin");
            ni::Tensor b = load_bin(dir + "add_b.bin");
            ni::Tensor expected = load_bin(dir + "add_expected.bin");
            fails += compare_tensors(ni::add(a, b), expected, 1e-6, "add");
        }
    } catch (const std::exception& e) {
        std::printf("ops_parity: exception: %s\n", e.what());
        return 1;
    }

    std::printf(fails ? "ops_parity: %d failures\n" : "ops_parity: ok\n", fails);
    return fails ? 1 : 0;
}
