// Parity check against numpy: load inputs + numpy-computed expected outputs
// (written by cpp/tools/gen_fixtures.py), run the C++ ops, assert allclose.
// This is the C++ analogue of nanoinfer's "parity with HuggingFace" floor.
//
// Usage: ops_parity <fixtures_dir>   (ctest passes the build-dir fixtures path).
#include <string>
#include <vector>

#include "ops.hpp"
#include "serialize.hpp"
#include "test_util.hpp"

using ni::load_bin;

// Constants shared with gen_fixtures.py.
static constexpr float kRmsEps = 1e-6f;
static constexpr int64_t kRopeSeq = 4;
static constexpr int64_t kRopeHeadDim = 8;
static constexpr float kRopeTheta = 10000.0f;

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
        // linear
        {
            ni::Tensor x = load_bin(dir + "linear_x.bin");
            ni::Tensor w = load_bin(dir + "linear_w.bin");
            ni::Tensor b = load_bin(dir + "linear_bias.bin");
            ni::Tensor expected = load_bin(dir + "linear_expected.bin");
            fails += compare_tensors(ni::linear(x, w, &b), expected, 1e-3, "linear");
        }
        // silu
        {
            ni::Tensor x = load_bin(dir + "silu_x.bin");
            ni::Tensor expected = load_bin(dir + "silu_expected.bin");
            fails += compare_tensors(ni::silu(x), expected, 1e-5, "silu");
        }
        // mul
        {
            ni::Tensor a = load_bin(dir + "mul_a.bin");
            ni::Tensor b = load_bin(dir + "mul_b.bin");
            ni::Tensor expected = load_bin(dir + "mul_expected.bin");
            fails += compare_tensors(ni::mul(a, b), expected, 1e-6, "mul");
        }
        // embedding (ids mirror gen_fixtures.py)
        {
            ni::Tensor table = load_bin(dir + "embedding_table.bin");
            ni::Tensor expected = load_bin(dir + "embedding_expected.bin");
            std::vector<int64_t> ids = {3, 0, 19, 7};
            fails += compare_tensors(ni::embedding(table, ids), expected, 1e-6, "embedding");
        }
        // rope cache: built tables match numpy's
        {
            ni::RopeCache rc = ni::build_rope_cache(kRopeSeq, kRopeHeadDim, kRopeTheta);
            fails += compare_tensors(rc.cos, load_bin(dir + "rope_cos.bin"), 1e-5, "rope_cos");
            fails += compare_tensors(rc.sin, load_bin(dir + "rope_sin.bin"), 1e-5, "rope_sin");
        }
        // rope apply: use the loaded cache so this isolates apply from build
        {
            ni::Tensor q = load_bin(dir + "rope_q.bin");
            ni::Tensor cos = load_bin(dir + "rope_cos.bin");
            ni::Tensor sin = load_bin(dir + "rope_sin.bin");
            ni::Tensor expected = load_bin(dir + "rope_applied_expected.bin");
            fails += compare_tensors(ni::apply_rope(q, cos, sin), expected, 1e-5, "rope_apply");
        }
        // split_heads / merge_heads (shapes mirror gen_fixtures.py)
        {
            ni::Tensor x = load_bin(dir + "split_x.bin");
            ni::Tensor expected = load_bin(dir + "split_expected.bin");
            fails += compare_tensors(ni::split_heads(x, 2, 4), expected, 1e-6, "split_heads");
        }
        {
            ni::Tensor x = load_bin(dir + "merge_x.bin");
            ni::Tensor expected = load_bin(dir + "merge_expected.bin");
            fails += compare_tensors(ni::merge_heads(x), expected, 1e-6, "merge_heads");
        }
        // repeat_kv (n_rep mirrors gen_fixtures.py)
        {
            ni::Tensor x = load_bin(dir + "repeatkv_x.bin");
            ni::Tensor expected = load_bin(dir + "repeatkv_expected.bin");
            fails += compare_tensors(ni::repeat_kv(x, 3), expected, 1e-6, "repeat_kv");
        }
        // causal attention
        {
            ni::Tensor q = load_bin(dir + "attn_q.bin");
            ni::Tensor k = load_bin(dir + "attn_k.bin");
            ni::Tensor v = load_bin(dir + "attn_v.bin");
            ni::Tensor expected = load_bin(dir + "attn_expected.bin");
            fails += compare_tensors(ni::attention(q, k, v, true), expected, 1e-5, "attention");
        }
    } catch (const std::exception& e) {
        std::printf("ops_parity: exception: %s\n", e.what());
        return 1;
    }

    std::printf(fails ? "ops_parity: %d failures\n" : "ops_parity: ok\n", fails);
    return fails ? 1 : 0;
}
